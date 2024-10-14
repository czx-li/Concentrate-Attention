import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data import TensorDataset
from transformers import RobertaTokenizer, RobertaModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def env(batch_size, way, state_norm1,state_norm2, model_name, state_dim):
    model_path = model_name
    state_dim = state_dim
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    bert = RobertaModel.from_pretrained(model_path)
    bert = bert.to(device)
    for param in bert.parameters():
        param.requires_grad = False
    train_text = []
    train_label = []
    train_id = []
    id_text = []
    f = open('16-shot/cr/16-13/train.tsv', encoding='utf-8')
    for line_counter, line in enumerate(f):
        if line_counter != 0:
            a = int(line.strip().split('\t')[1])
            b = line.strip().split('\t')[0]
            train_label.append(a)
            if way == 'SST-2':
                train_text.append('[CLS] [SST-2] [SEP] Review:' + b + ' Sentiment: <mask>.  [SEP]')
                id_text.append(b)
            train_id.append(line_counter - 1)
    f.close()
    f = open('16-shot/sst-2/16-13/train.tsv', encoding='utf-8')
    for line_counter, line in enumerate(f):
        if line_counter != 0:
            a = int(line.strip().split('\t')[1])
            b = line.strip().split('\t')[0]
            train_label.append(a)
            if way == 'SST-2':
                train_text.append('[CLS] [SST-2] [SEP] Review:' + b + ' Sentiment: <mask>.  [SEP]')
                id_text.append(b)
            train_id.append(line_counter - 1)
    f.close()
    print(train_text[1])
    print(id_text[1])
    print(train_label[1])
    print(train_id[1])
    train_label = torch.tensor(train_label)
    train_id = torch.tensor(train_id)
    train_state, len_text = state(train_text, id_text, tokenizer, bert, state_dim, batch_size)
    len_text = torch.tensor(len_text)
    train_dataset = TensorDataset(train_state, train_label, train_id, len_text)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        sampler=SequentialSampler(train_dataset),
        batch_size=batch_size
    )
    for step, batch in enumerate(train_dataloader):
        states = batch[0].cpu()
        for i in range(len(states)):
           state_norm1(states[i])
           state_norm2(states[i])
    print('Finished_Make_TrainData ')
    return train_dataloader, train_text, state_norm1, state_norm2, bert


def state(text, id_tetx, tokenizer, bert, state_dim, batch_size):
    input_ids = []
    attention_masks = []
    len_text = []
    for sent in text:
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
            max_length=512,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
            truncation=True
        )
        Prompt_list = encoded_dict['input_ids'].numpy().tolist()
        Prompt_list = torch.tensor(Prompt_list)
        input_ids.append(Prompt_list)
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    dataset = TensorDataset(input_ids, attention_masks)
    dataloader = DataLoader(
        dataset,  # The training samples.
        shuffle=False,
        sampler=SequentialSampler(dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )
    states = torch.empty(batch_size, state_dim)
    i = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(device)
            attention_masks = batch[1].to(device)
            with torch.no_grad():
                outputs = bert(input_ids, attention_masks)
            a = outputs[0][:, 0, :]
            if i == 0:
                states = a
            else:
                states = torch.cat((states, a), 0)
            i += 1
    states = states

    for sent in id_tetx:
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
            max_length=512,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
            truncation=True
        )
        Prompt_list = encoded_dict['input_ids'].numpy().tolist()
        Prompt_list = torch.tensor(Prompt_list)
        len_text.append(torch.sum(Prompt_list != 1))

    return states, len_text






















