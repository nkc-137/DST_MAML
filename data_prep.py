import argparse
import os
import json
import collections
from collections import OrderedDict
from utils import save_configure
from utils import get_label_embedding
from utils import warmup_linear, initialize_model, get_dataloader, _truncate_seq_pair, train_model
import csv
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
from tqdm import tqdm, trange
from pytorch_pretrained_bert.optimization import BertAdam
from torch.optim import Adam
import torch.nn.utils
import copy
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



class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if len(line) > 0 and line[0][0] == '#':     # ignore comments (starting with '#')
                    continue
                lines.append(line)
            return lines


class Processor(DataProcessor):
    """Processor for the belief tracking dataset (GLUE version)."""

    def __init__(self, config):
        super(Processor, self).__init__()

        # MultiWOZ dataset
        fp_ontology = open(os.path.join(config['data_dir'], "ontology.json"), "r")
        ontology = json.load(fp_ontology)
        for slot in ontology.keys():
            ontology[slot].append("none")
        fp_ontology.close()

        slot_idx = {'domain_attraction':'0:1:2', 'domain_bus':'3:4:5:6', 'domain_hospital':'7', 'domain_hotel':'8:9:10:11:12:13:14:15:16:17',\
                    'domain_restaurant':'18:19:20:21:22:23:24', 'domain_taxi':'25:26:27:28', 'domain_train':'29:30:31:32:33:34'}
        target_slot =[]

        # self.train_slots = ['domain_hotel', 'domain_train', 'domain_attraction', 'domain_restaurant', 'domain_taxi', 'domain_hospital']
        self.train_slots = ['domain_hospital']
        self.test_slot = ['bus']


        for key, value in slot_idx.items():
            if key in self.train_slots:
                target_slot.append(value)
        joint_target_slot = ':'.join(target_slot)

        # sorting the ontology according to the alphabetic order of the slots
        ontology = collections.OrderedDict(sorted(ontology.items()))

        # select slots to train
        nslots = len(ontology.keys())
        target_slot = list(ontology.keys())
        self.target_slot_idx = sorted([int(x) for x in joint_target_slot.split(':')])

        for idx in range(0, nslots):
            if not idx in self.target_slot_idx:
                del ontology[target_slot[idx]]

        self.ontology = ontology
        self.target_slot = list(self.ontology.keys())
        for i, slot in enumerate(self.target_slot):
            if slot == "pricerange":
                self.target_slot[i] = "price range"


    def get_train_examples(self, data_dir):
        """See base class."""
        # data = []
        # for slot in self.train_slots:
        # 	data.append(self._read_tsv(os.path.join(data_dir, slot+".tsv")))
        # data = [item for elem in data for item in elem]
        # return self._create_examples(data)

        # TEMPORARY CODE - DELETE  - UNCOMMENT ABOVE 5 LINES OF CODE
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")))

        

    def get_domain_examples(self, data_dir, domain):
        data = []
        data.append(self._read_tsv(os.path.join(data_dir, domain+".tsv")))
        data = [item for elem in data for item in elem]

        return self._create_examples(data)

    def get_test_examples(self, data_dir):
        """See base class."""
        data = []
        for slot in self.test_slots:
        	data.append(self._read_tsv(os.path.join(data_dir, slot+".tsv")))

        data = [item for elem in data for item in elem]
        return self._create_examples(data)

    def get_dev_examples(self, data_dir, accumulation=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", accumulation)

    def get_labels(self):
        """See base class."""
        return [ self.ontology[slot] for slot in self.target_slot]

    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        prev_dialogue_index = None
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (line[0], line[1])  # line[0]: dialogue index, line[1]: turn index
            text_a = line[2]  # line[2]: user utterance
            text_b = line[3]  # line[3]: system response

            label = [line[4+idx] for idx in self.target_slot_idx]

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples



def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--data_dir",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    # args = parser.parse_args()
    # PARAMETERS LISTED HERE
    args = {}
    args['data_dir'] = 'data/'
    args['max_seq_length'] = 64
    args['max_turn_length'] = 22
    args['train_batch_size'] = 3
    args['gradient_accumulation_steps'] = 1
    args['num_train_epochs'] = 300
    args['local_rank'] = -1
    args['nbt'] = 'rnn' # Other is 'transformer'
    args['fp16'] = False
    args['local_rank'] = -1
    args['no_cude'] = True
    local_rank = -1


    # CUDA setup
    if local_rank == -1 or no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
        print("This is the # of gpus: ", n_gpu)
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    # logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
    #     device, n_gpu, bool(args.local_rank != -1), args.fp16))



    processor = Processor(args)
    label_list = processor.get_labels()
    num_labels = [len(labels) for labels in label_list]

    # Tokenizer
    vocab_dir = 'bert/bert-base-uncased-vocab.txt'
    tokenizer = BertTokenizer.from_pretrained(vocab_dir)


    train_data = processor.get_domain_examples(args['data_dir'], 'domain_bus')

    ## Get domain-wise training examples
    # domains = ['domain_bus', 'domain_hospital', 'domain_hotel', 'domain_attraction', 'domain_restaurant', 'domain_taxi', 'domain_train']
    domains = ['domain_bus']
    domain_data = {}
    for domain in domains:
        domain_data[domain] = processor.get_domain_examples(args['data_dir'], domain)

    domain_dataloaders = {}
    domain_num_train = {}
    for domain in domains:
        loader, nums = get_dataloader(args, domain_data[domain], label_list, args['max_seq_length'], tokenizer, args['max_turn_length'], local_rank, device)
        domain_dataloaders[domain]  = loader
        domain_num_train[domain] = nums 



    # all_input_ids, all_input_len, all_label_ids = convert_examples_to_features(
    #         train_data, label_list, max_seq_length, tokenizer, max_turn_length)

    # num_train_batches = all_input_ids.size(0)
    # num_train_steps = int(num_train_batches / train_batch_size / gradient_accumulation_steps * num_train_epochs)

    # all_input_ids, all_input_len, all_label_ids = all_input_ids.to(device), all_input_len.to(device), all_label_ids.to(device)

    # train_data = TensorDataset(all_input_ids, all_input_len, all_label_ids)
    # if local_rank == -1:
    #     train_sampler = RandomSampler(train_data)
    # else:
    #     train_sampler = DistributedSampler(train_data)

    # train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)


    train_dataloader, num_train_steps = get_dataloader(args, train_data, label_list, args['max_seq_length'], tokenizer, args['max_turn_length'], local_rank, device)
    ### BUILDING THE MODELS ###
    # Prepare model
    if args['nbt'] =='rnn':
        from BeliefTrackerSlotQueryMultiSlot import BeliefTracker
    elif args['nbt'] == 'transformer':
        from BeliefTrackerSlotQueryMultiSlotTransformer import BeliefTracker
    else:
        raise ValueError('nbt type should be either rnn or transformer')

    args = {}
    args['hidden_dim'] = 300
    args['num_rnn_layers'] = 2
    args['zero_init_rnn'] = True
    args['max_seq_length'] = 64
    args['max_label_length'] = 32
    args['attn_head'] = 4
    args['fix_utterance_encoder'] = True
    args['task_name'] = 'bert-gru-sumbt'
    args['distance_metric'] = 'euclidean'
    args['output_dir'] = 'model_out/'
    args['local_rank'] = -1
    args['learning_rate'] = 1e-4
    args['gradient_accumulation_steps'] = 1
    args['warmup_proportion'] = 0.1
    args['num_train_epochs'] = 3 # CHANGE THIS TO 300
    args['device'] = device
    args['fp16'] = False


    model = initialize_model(args, BeliefTracker, processor, label_list, tokenizer, num_labels, device)

    # Data parallelize when use multi-gpus
    if args['local_rank'] != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)


    ## Prepare optimizer
    def get_optimizer_grouped_parameters(model):
            param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
                 'lr': args['learning_rate']},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                 'lr': args['learning_rate']},
            ]
            return optimizer_grouped_parameters

    if n_gpu == 1:
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(model)
    else:
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(model)
        # optimizer_grouped_parameters = get_optimizer_grouped_parameters(model.module) # Original 

    t_total = num_train_steps

    optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args['learning_rate'],
                                 warmup=args['warmup_proportion'],
                                 t_total=t_total)


    print("THIS IS THE NUMBER OF TRAIN STEPS: ", num_train_steps)
    tr_losses = train_model(args, model, num_train_steps, train_dataloader, optimizer, n_gpu)

    
    #########################################
    ## TRAIN MAML MODEL
    #########################################
    lr = 0.003 # Learning rate for the DST model
    meta_lr = 0.003 # Learning rate for the meta learning updates
    meta_epoch = 4 # Should be 100
    maml_step = 1 # Number of meta learning steps to be taken

    meta_loss = [] # Store losses at each epoch

    for epoch in range(meta_epoch):
        print("STARTED META EPOCH: " i + 1)
        epoch_losses = []

        # Define the optimizers for updates
        optim = Adam(lr=lr, params=filter(lambda x: x.requires_grad, model.parameters()),weight_decay=1e-5)
        meta_optim = Adam(lr = meta_lr, params=filter(lambda x: x.requires_grad, model.parameters()),weight_decay=1e-5)

        # Store the main model
        init_state = copy.deepcopy(model.state_dict())

        # Store the main model in a file for loading in the next epoch
        PATH = "main_model.pt"
        torch.save(model, PATH)

        # init_state = copy.deepcopy(model)
        # new_state_dict = OrderedDict()
        # for k, v in init_state.items():
        #     name = k[7:] # remove `module.`
        #     new_state_dict[name] = v


        loss_tasks = []
        # Iterate through each target domain and fine tune models
        for key, value in domain_dataloaders.items():

            
            print("GOT LOADER FOR ", key)
            domain_loader = value

            # Initialize model
            domain_model = torch.load(PATH)
            optim.zero_grad()

            # Dictionary of arguments required for DST models on each domain
            domain_args = {}
            domain_args['hidden_dim'] = 300
            domain_args['num_rnn_layers'] = 2
            domain_args['zero_init_rnn'] = True
            domain_args['max_seq_length'] = 64
            domain_args['max_label_length'] = 32
            domain_args['attn_head'] = 4
            domain_args['fix_utterance_encoder'] = True
            domain_args['task_name'] = 'bert-gru-sumbt'
            domain_args['distance_metric'] = 'euclidean'
            domain_args['output_dir'] = 'model_out/'
            domain_args['local_rank'] = -1
            domain_args['learning_rate'] = 1e-4
            domain_args['gradient_accumulation_steps'] = 1
            domain_args['warmup_proportion'] = 0.1
            domain_args['num_train_epochs'] = 3 # CHANGE THIS TO 300
            domain_args['device'] = device
            domain_args['fp16'] = False

            print("STARTED TRAINING FOR THE DOMAIN: ", key)
            domain_loss = train_model(domain_args, domain_model, domain_num_train[key], domain_loader, optim, n_gpu)
            loss_tasks.append(domain_loss)

        meta_optim.zero_grad()

        print("CHEKC THIS OUT ----> ", print(loss_tasks[0]))
        print("CHEKC THIS OUT ----> ", print(type(loss_tasks[0])))
        loss_meta = torch.stack(loss_tasks).sum(0) / len(domain_dataloaders)

        # Do meta learning gradient update using the total of losses obtained above
        loss_meta.backward(retain_graph = True)
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        meta_optim.step()

        # Store the reslting model in a file to be loaded in the next epoch 
        init_state = copy.deepcopy(model.state_dict())
        torch.save(model, PATH)

        meta_loss.append(loss_meta.item())
    
    print(meta_loss)
    print("***********************")
    print("EVERYTHING IS DONE!!!")
    print("***********************")

















if __name__ == "__main__":
    main()


