import os
import json
from tqdm import tqdm, trange
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
def compute_mean(x):
    total = 0
    for i in x:
        total += i

    return total/len(x)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x

def save_configure(args, num_labels, ontology):
    with open(os.path.join(args['output_dir'], "config.json"),'w') as outfile:
        args["num_labels"] = num_labels
        args["ontology"] = ontology
        args['fp16'] = 'False'
        args['device'] = args['device'].type
        args['fix_utterance_encoder'] = 'True'

        json.dump(args, outfile, indent=4)

def get_label_embedding(labels, max_seq_length, tokenizer, device):
    features = []
    for label in labels:
        label_tokens = ["[CLS]"] + tokenizer.tokenize(label) + ["[SEP]"]
        label_token_ids = tokenizer.convert_tokens_to_ids(label_tokens)
        label_len = len(label_token_ids)

        label_padding = [0] * (max_seq_length - len(label_token_ids))
        label_token_ids += label_padding
        assert len(label_token_ids) == max_seq_length

        features.append((label_token_ids, label_len))

    all_label_token_ids = torch.tensor([f[0] for f in features], dtype=torch.long).to(device)
    all_label_len = torch.tensor([f[1] for f in features], dtype=torch.long).to(device)

    return all_label_token_ids, all_label_len

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def get_dataloader(args, train_data, label_list, max_seq_length, tokenizer, max_turn_length, local_rank, device):
        train_batch_size = args['train_batch_size']
        gradient_accumulation_steps = args['gradient_accumulation_steps']
        num_train_epochs = args['num_train_epochs']
        all_input_ids, all_input_len, all_label_ids = convert_examples_to_features(
            train_data, label_list, max_seq_length, tokenizer, max_turn_length)
        num_train_batches = all_input_ids.size(0)
        num_train_steps = int(num_train_batches / train_batch_size / gradient_accumulation_steps * num_train_epochs)
        all_input_ids, all_input_len, all_label_ids = all_input_ids.to(device), all_input_len.to(device), all_label_ids.to(device)

        train_data = TensorDataset(all_input_ids, all_input_len, all_label_ids)
        if local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

        return train_dataloader, num_train_steps

def initialize_model(args,tracker, processor,label_list, tokenizer, num_labels, device):
        model = tracker(args, num_labels, device)
        model.to(device)
        save_configure(args, num_labels, processor.ontology)

        print("Model made!!!!")

        ## Get slot-value embeddings
        label_token_ids, label_len = [], []
        for labels in label_list:
            token_ids, lens = get_label_embedding(labels, args['max_label_length'], tokenizer, device)
            label_token_ids.append(token_ids)
            label_len.append(lens)

        print("Slot-value embeddings done!!!")

        ## Get domain-slot-type embeddings
        slot_token_ids, slot_len = get_label_embedding(processor.target_slot, args['max_label_length'], tokenizer, device)
        print("Domain-Slot embeddings done!!!")

        ## Initialize slot and value embeddings
        model.initialize_slot_value_lookup(label_token_ids, slot_token_ids)
        print("Slot-value embeddings intilaized!!!")

        return model

def train_model(args, model, num_train_steps, train_dataloader, optimizer, n_gpu):
        print("Started training the model!!!")

        global_step = 0
        last_update = None
        best_loss = None
        tr_losses = []
        t_total = num_train_steps
        epoch_losses = []
        for epoch in trange(int(args['num_train_epochs']), desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples = 0
            nb_tr_steps = 0

            step_losses = []
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(args['device']) for t in batch)
                input_ids, input_len, label_ids = batch

                # Forward
                if n_gpu == 1:
                    loss, loss_slot, acc, acc_slot, _ = model(input_ids, input_len, label_ids, n_gpu)
                else:
                    loss, _, acc, acc_slot, _ = model(input_ids, input_len, label_ids, n_gpu)

                    # average to multi-gpus
                    loss = loss.mean()
                    acc = acc.mean()
                    acc_slot = acc_slot.mean(0)

                if args['gradient_accumulation_steps'] > 1:
                    loss = loss / args['gradient_accumulation_steps']

                # Backward
                # print("CHECK THIS: ", args['fp16'])
                # if args['fp16']:
                #     print("CHECK THIS: ", args['fp16'])
                #     optimizer.backward(loss)
                # else:
                # loss.backward() # Original
                loss.backward(retain_graph=True)

                tr_loss += loss.item()
                step_losses.append(loss)

                # nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % args['gradient_accumulation_steps'] == 0:
                    # modify lealrning rate with special warm up BERT uses
                    lr_this_step = args['learning_rate'] * warmup_linear(global_step / t_total, args['warmup_proportion'])
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
            tr_losses.append(tr_loss)
            epoch_losses.append(torch.stack(step_losses).sum(0))
        
        total = torch.stack(epoch_losses).sum(0)
        # return total
        return loss

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, max_turn_length):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = [{label: i for i, label in enumerate(labels)} for labels in label_list]
    slot_dim = len(label_list)

    features = []
    prev_dialogue_idx = None
    all_padding = [0] * max_seq_length
    all_padding_len = [0, 0]

    max_turn = 0
    for (ex_index, example) in enumerate(examples):
        if max_turn < int(example.guid.split('-')[1]):
            max_turn = int(example.guid.split('-')[1])
    max_turn_length = min(max_turn+1, max_turn_length)

    for (ex_index, example) in enumerate(examples):
        tokens_a = [x if x != '#' else '[SEP]' for x in tokenizer.tokenize(example.text_a)]
        tokens_b = None
        if example.text_b:
            tokens_b = [x if x != '#' else '[SEP]' for x in tokenizer.tokenize(example.text_b)]
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        input_len = [len(tokens), 0]

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            input_len[1] = len(tokens_b) + 1

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        assert len(input_ids) == max_seq_length

        label_id = []
        label_info = 'label: '
        for i, label in enumerate(example.label):
            if label == 'dontcare':
                label = 'do not care'
            label_id.append(label_map[i][label])
            label_info += '%s (id = %d) ' % (label, label_map[i][label])

        curr_dialogue_idx = example.guid.split('-')[0]
        curr_turn_idx = int(example.guid.split('-')[1])
        # print("Current dilog id: ", curr_dialogue_idx, " curr_turn id: ", curr_turn_idx)
        if prev_dialogue_idx is not None and prev_dialogue_idx != curr_dialogue_idx:
            if prev_turn_idx < max_turn_length:
                features += [InputFeatures(input_ids=all_padding,
                                           input_len=all_padding_len,
                                           label_id=[-1]*slot_dim)]*(max_turn_length - prev_turn_idx - 1)
            # print("IF, Lenght of featueres: ",len(features))
            # print((max_turn_length - prev_turn_idx - 1), prev_turn_idx, max_turn_length)
            assert len(features) % max_turn_length == 0

        if prev_dialogue_idx is None or prev_turn_idx < max_turn_length:
            features.append(
                InputFeatures(input_ids=input_ids,
                              input_len=input_len,
                              label_id=label_id))

        prev_dialogue_idx = curr_dialogue_idx
        prev_turn_idx = curr_turn_idx

        # print("EXAMPLES PROCESSED: ", len(features),example.guid.split('-')[1])
        # print(max_turn_length - prev_turn_idx - 1, prev_turn_idx, max_turn_length)

    if prev_turn_idx < max_turn_length:
        features += [InputFeatures(input_ids=all_padding,
                                   input_len=all_padding_len,
                                   label_id=[-1]*slot_dim)]\
                    * (max_turn_length - prev_turn_idx - 1)
    # assert len(features) % max_turn_length == 0

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_len= torch.tensor([f.input_len for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    # reshape tensors to [#batch, #max_turn_length, #max_seq_length]
    all_input_ids = all_input_ids.view(-1, max_turn_length, max_seq_length)
    all_input_len = all_input_len.view(-1, max_turn_length, 2)
    all_label_ids = all_label_ids.view(-1, max_turn_length, slot_dim)

    print("***************************************")
    print("LENGTH OF ALL THE EXAMPLES: ", len(examples))
    print("LENGTH OF ALL THE FEATURES: ", len(features))
    print("MAX TURN LENGTH: ", max_turn_length)
    print("***************************************")

    return all_input_ids, all_input_len, all_label_ids

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_len, label_id):
        self.input_ids = input_ids
        self.input_len = input_len
        self.label_id = label_id




