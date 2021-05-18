###############################################################################
# Training code
###############################################################################
# global_step = 0
# last_update = None
# best_loss = None

# print("Started training the model!!!")
# for epoch in trange(int(args['num_train_epochs']), desc="Epoch"):
#     model.train()
#     tr_loss = 0
#     nb_tr_examples = 0
#     nb_tr_steps = 0

#     for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
#         batch = tuple(t.to(device) for t in batch)
#         input_ids, input_len, label_ids = batch

#         # Forward
#         if n_gpu == 1:
#             loss, loss_slot, acc, acc_slot, _ = model(input_ids, input_len, label_ids, n_gpu)
#         else:
#             loss, _, acc, acc_slot, _ = model(input_ids, input_len, label_ids, n_gpu)

#             # average to multi-gpus
#             loss = loss.mean()
#             acc = acc.mean()
#             acc_slot = acc_slot.mean(0)

#         if args['gradient_accumulation_steps'] > 1:
#             loss = loss / args['gradient_accumulation_steps']

#         # Backward
#         if fp16:
#             optimizer.backward(loss)
#         else:
#             loss.backward()



# model = BeliefTracker(args, num_labels, device)
# model.to(device)
# save_configure(args, num_labels, processor.ontology)

# print("Model made!!!!")

# ## Get slot-value embeddings
# label_token_ids, label_len = [], []
# for labels in label_list:
#     token_ids, lens = get_label_embedding(labels, args['max_label_length'], tokenizer, device)
#     label_token_ids.append(token_ids)
#     label_len.append(lens)

# print("Slot-value embeddings done!!!")

# ## Get domain-slot-type embeddings
# slot_token_ids, slot_len = get_label_embedding(processor.target_slot, args['max_label_length'], tokenizer, device)
# print("Domain-Slot embeddings done!!!")

# ## Initialize slot and value embeddings
# model.initialize_slot_value_lookup(label_token_ids, slot_token_ids)
# print("Slot-value embeddings intilaized!!!")







# def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, max_turn_length):
#     """Loads a data file into a list of `InputBatch`s."""

#     label_map = [{label: i for i, label in enumerate(labels)} for labels in label_list]
#     slot_dim = len(label_list)

#     features = []
#     prev_dialogue_idx = None
#     all_padding = [0] * max_seq_length
#     all_padding_len = [0, 0]

#     max_turn = 0
#     for (ex_index, example) in enumerate(examples):
#         if max_turn < int(example.guid.split('-')[1]):
#             max_turn = int(example.guid.split('-')[1])
#     max_turn_length = min(max_turn+1, max_turn_length)

#     for (ex_index, example) in enumerate(examples):
#         tokens_a = [x if x != '#' else '[SEP]' for x in tokenizer.tokenize(example.text_a)]
#         tokens_b = None
#         if example.text_b:
#             tokens_b = [x if x != '#' else '[SEP]' for x in tokenizer.tokenize(example.text_b)]
#             # Modifies `tokens_a` and `tokens_b` in place so that the total
#             # length is less than the specified length.
#             # Account for [CLS], [SEP], [SEP] with "- 3"
#             _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
#         else:
#             # Account for [CLS] and [SEP] with "- 2"
#             if len(tokens_a) > max_seq_length - 2:
#                 tokens_a = tokens_a[:(max_seq_length - 2)]

#         # The convention in BERT is:
#         # (a) For sequence pairs:
#         #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
#         #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
#         # (b) For single sequences:
#         #  tokens:   [CLS] the dog is hairy . [SEP]
#         #  type_ids: 0   0   0   0  0     0 0
#         #
#         # Where "type_ids" are used to indicate whether this is the first
#         # sequence or the second sequence. The embedding vectors for `type=0` and
#         # `type=1` were learned during pre-training and are added to the wordpiece
#         # embedding vector (and position vector). This is not *strictly* necessary
#         # since the [SEP] token unambigiously separates the sequences, but it makes
#         # it easier for the model to learn the concept of sequences.
#         #
#         # For classification tasks, the first vector (corresponding to [CLS]) is
#         # used as as the "sentence vector". Note that this only makes sense because
#         # the entire model is fine-tuned.

#         tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
#         input_len = [len(tokens), 0]

#         if tokens_b:
#             tokens += tokens_b + ["[SEP]"]
#             input_len[1] = len(tokens_b) + 1

#         input_ids = tokenizer.convert_tokens_to_ids(tokens)

#         # Zero-pad up to the sequence length.
#         padding = [0] * (max_seq_length - len(input_ids))
#         input_ids += padding
#         assert len(input_ids) == max_seq_length

#         label_id = []
#         label_info = 'label: '
#         for i, label in enumerate(example.label):
#             if label == 'dontcare':
#                 label = 'do not care'
#             label_id.append(label_map[i][label])
#             label_info += '%s (id = %d) ' % (label, label_map[i][label])

#         curr_dialogue_idx = example.guid.split('-')[0]
#         curr_turn_idx = int(example.guid.split('-')[1])
#         # print("Current dilog id: ", curr_dialogue_idx, " curr_turn id: ", curr_turn_idx)
#         if prev_dialogue_idx is not None and prev_dialogue_idx != curr_dialogue_idx:
#             if prev_turn_idx < max_turn_length:
#                 features += [InputFeatures(input_ids=all_padding,
#                                            input_len=all_padding_len,
#                                            label_id=[-1]*slot_dim)]*(max_turn_length - prev_turn_idx - 1)
#             # print("IF, Lenght of featueres: ",len(features))
#             # print((max_turn_length - prev_turn_idx - 1), prev_turn_idx, max_turn_length)
#             assert len(features) % max_turn_length == 0

#         if prev_dialogue_idx is None or prev_turn_idx < max_turn_length:
#             features.append(
#                 InputFeatures(input_ids=input_ids,
#                               input_len=input_len,
#                               label_id=label_id))

#         prev_dialogue_idx = curr_dialogue_idx
#         prev_turn_idx = curr_turn_idx

#         # print("EXAMPLES PROCESSED: ", len(features),example.guid.split('-')[1])
#         # print(max_turn_length - prev_turn_idx - 1, prev_turn_idx, max_turn_length)

#     if prev_turn_idx < max_turn_length:
#         features += [InputFeatures(input_ids=all_padding,
#                                    input_len=all_padding_len,
#                                    label_id=[-1]*slot_dim)]\
#                     * (max_turn_length - prev_turn_idx - 1)
#     # assert len(features) % max_turn_length == 0

#     all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
#     all_input_len= torch.tensor([f.input_len for f in features], dtype=torch.long)
#     all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

#     # reshape tensors to [#batch, #max_turn_length, #max_seq_length]
#     all_input_ids = all_input_ids.view(-1, max_turn_length, max_seq_length)
#     all_input_len = all_input_len.view(-1, max_turn_length, 2)
#     all_label_ids = all_label_ids.view(-1, max_turn_length, slot_dim)

#     print("***************************************")
#     print("LENGTH OF ALL THE EXAMPLES: ", len(examples))
#     print("LENGTH OF ALL THE FEATURES: ", len(features))
#     print("MAX TURN LENGTH: ", max_turn_length)
#     print("***************************************")

#     return all_input_ids, all_input_len, all_label_ids