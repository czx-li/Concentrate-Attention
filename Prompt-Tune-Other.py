# 开发者：李丞正旭
# 开发时间： 2022/12/10 21:10
from transformers import RobertaTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import model as M
import torch
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os

epochs = 300
batch_size = 32
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
vocab_size = 50265
method = 'prompt tune'
positive_words = ['True']
negative_words = ['False']
source1 = 'WNLI'
source2 = 'QNLI'
target = 'RTE'
train_text1 = []
train_text2 = []
train_label = []
test_text1 = []
test_text2 = []
test_label = []
prompt_words = positive_words + negative_words
seed = 5
M.set_seed(seed)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def data(data_name, type):
    f = open('16-shot/' + data_name + '/' + type +'.tsv', encoding='utf-8')
    text1 = []
    text2 = []
    label = []
    for line_counter, line in enumerate(f):
        if line_counter != 0:
            if data_name == 'WNLI':
                if type == 'test':
                    a = line.strip().split('\t')[1]
                    c = line.strip().split('\t')[2]
                    label.append(0)
                    text1.append('<mask> ' + a)
                    text2.append(c)
                else:
                    a = line.strip().split('\t')[1]
                    c = line.strip().split('\t')[2]
                    d = line.strip().split('\t')[3]
                    label.append(int(d))
                    text1.append('<mask> ' + a)
                    text2.append(c)

            else:
                if type == 'test':
                    a = line.strip().split('\t')[1]
                    c = line.strip().split('\t')[2]
                    label.append(0)
                    text1.append('<mask> ' + a)
                    text2.append(c)
                else:
                    a = line.strip().split('\t')[1]
                    c = line.strip().split('\t')[2]
                    d = line.strip().split('\t')[3]
                    if d == "entailment":
                        label.append(1)
                    else:
                        label.append(0)
                    text1.append('<mask> ' + a)
                    text2.append(c)
    f.close()
    return (text1, text2, label)

def token(text1, text2, method):
    input_ids = []
    attention_masks = []
    for sent1, sent2 in zip(text1, text2):
        encoded_dict = tokenizer.encode_plus(
            sent1,  # Sentence to encode.
            sent2,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=512,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
            truncation=True
        )
        Prompt_list = encoded_dict['input_ids'].numpy().tolist()
        if method == 'prompt tune':
            Prompt_list[0].insert(1, -1)
            Prompt_list[0].insert(2, -1)
            Prompt_list[0].insert(3, -1)
            Prompt_list[0].insert(4, -1)
            Prompt_list[0].insert(5, -1)
            Prompt_list[0].pop(len(Prompt_list[0]) - 1)
            Prompt_list[0].pop(len(Prompt_list[0]) - 1)
            Prompt_list[0].pop(len(Prompt_list[0]) - 1)
            Prompt_list[0].pop(len(Prompt_list[0]) - 1)
            Prompt_list[0].pop(len(Prompt_list[0]) - 1)
            """print(Prompt_list)
            sys.exit(0)"""
        Prompt_list = torch.tensor(Prompt_list)
        input_ids.append(Prompt_list)
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks

train_text1 = data(source1, 'train')[0] + data(source2,'train')[0]
train_text2 = data(source1, 'train')[1] + data(source2,'train')[1]
train_label = data(source1, 'train')[2] + data(source2, 'train')[2]
test_text1 = data(target, 'test')[0]
test_text2 = data(target, 'test')[1]
test_label = data(target, 'test')[2]
tokenizer = RobertaTokenizer.from_pretrained("roberta_large")
mask_token_id = tokenizer(tokenizer.mask_token)['input_ids'][1]
positive_token_ids = tokenizer(" ".join(positive_words))['input_ids'][1:-1]
negative_token_ids = tokenizer(" ".join(negative_words))['input_ids'][1:-1]
model = M.RobertaPromptTune(vocab_size, mask_token_id, positive_token_ids, negative_token_ids)
model = model.to(device)
train_label = torch.tensor(train_label)
test_label = torch.tensor(test_label)
train_input_ids, train_attention_masks = token(train_text1, train_text2, method)
test_input_ids, test_attention_masks = token(test_text1, test_text2, method)
print(test_input_ids.shape)
print(test_attention_masks.shape)
print(test_label.shape)

train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_label)
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_label)


train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )
test_dataloader = DataLoader(
            test_dataset, # The validation samples.
            sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )
optimizer = AdamW(model.parameters(),
                  lr = 2e-5,
                  eps = 1e-8
                )
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,  num_warmup_steps = 0, num_training_steps = total_steps)


training_stats = []
best_score = 0
acc_list2 = []
best_model = None
total_t0 = time.time()
for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    t0 = time.time()
    total_train_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):

          # Progress update every 40 batches.
          if step % 40 == 0 and not step == 0:
              # Calculate elapsed time in minutes.
              elapsed = format_time(time.time() - t0)
              # Report progress.
              print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))


          b_input_ids = batch[0].to(device)
          b_input_mask = batch[1].to(device)
          b_labels = batch[2].to(device)
          model.zero_grad()

          logits,loss = model(b_input_ids,
                         b_input_mask,
                         b_labels,
                         'train_4')


          total_train_loss += loss.item()
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
          optimizer.step()
          scheduler.step()



    avg_train_loss = total_train_loss / len(train_dataloader)
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
    print("")
    print("Running Validation...")

    t0 = time.time()
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0


    if epoch_i == epochs-1:
        acc_list = []
        for batch in test_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                logits, loss = model(b_input_ids,
                                     b_input_mask,
                                     b_labels,
                                     "test")
            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            pred_flat = np.argmax(logits, axis=1).flatten()
            acc_list = np.append(acc_list, pred_flat)

        with open('16-shot/glue_result/' + target + '.tsv', 'w') as f:
            tsv_w = csv.writer(f, delimiter='\t', lineterminator='\n')
            tsv_w.writerow(['IDs', 'labels'])
            for i in range(len(acc_list)):
                IDs = i
                if target == 'WNLI':
                    if acc_list[i] == 1:
                        labels = "1"
                    else:
                        labels = "0"
                else:
                    if acc_list[i] == 1:
                        labels = "entailment"
                    else:
                        labels = "not_entailment"
                tsv_w.writerow([IDs, labels])


print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))