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
import sys
import os

epochs = 100
batch_size = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vocab_size = 50265
method = 'prefix tune'
positive_words = ['positive']
negative_words = ['negative']
train_text = []
train_label = []
test_text = []
test_label = []
prompt_words = positive_words + negative_words
seed = 4
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

def data(data_name, type, method):
    f = open('16-shot/' + data_name + '/16-13/' + type + '.tsv', encoding='utf-8')
    text = []
    label = []
    for line_counter, line in enumerate(f):
        if line_counter != 0:
            a = int(line.strip().split('\t')[1])
            b = line.strip().split('\t')[0]
            label.append(a)
            if method == 'prompt tune':
                text.append('Review: Sentiment: Review:' + b + 'Sentiment: <mask>')
            if method == 'prefix tune':
                text.append('This is a <mask> comment : ' + b)
    f.close()
    return (text, label)

def token(text):
    input_ids = []
    attention_masks = []
    for sent in text:
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=512,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
            truncation=True
        )
        Prompt_list = encoded_dict['input_ids'].numpy().tolist()
        Prompt_list = torch.tensor(Prompt_list)
        print(Prompt_list)
        input_ids.append(Prompt_list)
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks

train_text = data('cr', 'train', method)[0] + data('sst-2','train', method)[0]
train_label = data('cr', 'train', method)[1] + data('sst-2', 'train', method)[1]
test_text = data('mr', 'test', method)[0]
test_label = data('mr', 'test', method)[1]
tokenizer = RobertaTokenizer.from_pretrained("roberta_large")
mask_token_id = tokenizer(tokenizer.mask_token)['input_ids'][1]
positive_token_ids = tokenizer(" ".join(positive_words))['input_ids'][1:-1]
negative_token_ids = tokenizer(" ".join(negative_words))['input_ids'][1:-1]

model = M.RobertaPromptTune(vocab_size, mask_token_id, positive_token_ids, negative_token_ids)
model = model.to(device)
train_label = torch.tensor(train_label)
test_label = torch.tensor(test_label)
train_input_ids, train_attention_masks = token(train_text)
test_input_ids, test_attention_masks = token(test_text)
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
                  lr = 2e-4, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
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
    total_train_loss1 = 0
    total_train_loss2 = 0
    total_train_loss3 = 0
    model.train()
    for step, batch in enumerate(train_dataloader):

          b_input_ids = batch[0].to(device)
          b_input_mask = batch[1].to(device)
          b_labels = batch[2].to(device)
          model.zero_grad()

          logits, loss, loss_1, loss_2, loss_3 = model(b_input_ids,
                         b_input_mask,
                         b_labels,
                        'train_SC')


          total_train_loss += loss.item()
          total_train_loss1 += loss_1.item()
          total_train_loss2 += loss_2.item()
          total_train_loss3 += loss_3.item()


          loss.backward()
        #   for param in model.parameters():
        #       print(param.grad)
          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
          optimizer.step()
          scheduler.step()
          # Progress update every 40 batches.
          if step % 1 == 0 and not step == 0:
              # Calculate elapsed time in minutes.
              elapsed = format_time(time.time() - t0)



    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_train_loss1 = total_train_loss1 / len(train_dataloader)
    avg_train_loss2 = total_train_loss2 / len(train_dataloader)
    avg_train_loss3 = total_train_loss3 / len(train_dataloader)

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Average training loss_1: {0:.2f}".format(avg_train_loss1))
    print("  Average training loss_2: {0:.2f}".format(avg_train_loss2))
    print("  Average training loss_3: {0:.2f}".format(avg_train_loss3))
    print("  Training epcoh took: {:}".format(training_time))
    print("")
    print("Running Validation...")

    t0 = time.time()
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0


    if epoch_i == epochs - 1:
        for batch in test_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                logits, loss, loss_1, loss_2, loss_3 = model(b_input_ids,
                                     b_input_mask,
                                     b_labels,
                                     'train_')
            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)
        avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
        acc_list2.append(avg_val_accuracy)
        print("  Accuracy: {0:}".format(avg_val_accuracy))
        avg_val_loss = total_eval_loss / len(test_dataloader)
        validation_time = format_time(time.time() - t0)
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))
        if avg_val_accuracy > best_score:
            best_score = avg_val_accuracy
            best_model = model
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )
print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))



episodes_list = list(range(len(acc_list2)))
plt.plot(episodes_list, acc_list2)
plt.xlabel('Episodes')
plt.ylabel('acc')
plt.title('Prompt_Tune on {}'.format('Fake News Detection'))
plt.show()

print('Saving your Bert model')
torch.save(best_model.state_dict(), 'Prompt-Turn_params.pth')