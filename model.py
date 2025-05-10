import sys
from peft import get_peft_config, PeftModel
import torch.nn as nn
from torch.nn import functional as F
from transformers import RobertaForMaskedLM, RobertaConfig, AutoModelForMaskedLM, AutoModelForSequenceClassification
import torch
import random
import numpy as np
from torch.autograd import Variable
from scipy.stats import entropy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
grads = {}


def save_grad(name):
    def hook(grad):
        print(f"name={name}, grad={grad}")
    return hook

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class RobertaPromptTune(nn.Module):
    def __init__(self,
                 vocab_size,
                 mask_token_id,
                 positive_token_ids,
                 negative_token_ids,
                 with_learnable_emb=True,
                 with_answer_weights=False,
                 with_position_weights=False,
                 num_learnable_token=5,
                 zero_shot=False,
                 fine_tune_all=False):
        super().__init__()
        self.bert = RobertaForMaskedLM.from_pretrained('roberta_large')
        if not fine_tune_all:
            for param in self.bert.base_model.parameters():
                param.requires_grad = False


        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.positive_token_ids = positive_token_ids
        self.negative_token_ids = negative_token_ids
        self.loss = torch.nn.CrossEntropyLoss()

        if zero_shot:
            with_learnable_emb = False
            with_answer_weights = False

        if with_answer_weights:
            # assume weights follow a uniform distribution
            self.positive_weights = nn.Parameter(torch.rand(
                len(positive_token_ids)), requires_grad=True)
            self.negative_weights = nn.Parameter(torch.rand(
                len(negative_token_ids)), requires_grad=True)
        else:
            self.positive_weights = nn.Parameter(torch.ones(
                len(positive_token_ids)), requires_grad=False)
            self.negative_weights = nn.Parameter(torch.ones(
                len(negative_token_ids)), requires_grad=False)

        self.learnable_tokens = - 1
        self.num_learnable_token = num_learnable_token
        if with_learnable_emb:
            self.learnable_token_emb = nn.Embedding(
                num_embeddings=self.num_learnable_token, embedding_dim=400)
        else:
            self.learnable_token_emb = None

        if with_position_weights:
            self.position_weights = nn.Parameter(
                torch.rand(2), requires_grad=True)
        else:
            self.position_weights = nn.Parameter(
                torch.ones(2), requires_grad=False)

        self.learnable_token_lstm = nn.LSTM(input_size=400, hidden_size=768//2, batch_first=True, bidirectional=True, dropout=0.33)
        self.learnable_token_ffn = nn.Linear(in_features=400, out_features=1024)
        # self.learnable_token_ffn = nn.Linear(in_features=300, out_features=1024)


    def forward(self, input_ids, attention_mask , labels, stage):
        batch_size, seq_len = input_ids.size()
        mask_ids = (input_ids == self.mask_token_id).nonzero(as_tuple=True)
        # mask_ids = mask_ids.expand(batch_size, seq_len, self.vocab_size)
        loss_1 = 0
        loss_2 = 0
        if self.learnable_token_emb is not None:
            add_ids = (input_ids == self.learnable_tokens).nonzero(as_tuple=True)
            input_ids[add_ids] = self.mask_token_id
            # add learnable token embeddings
            replace_embeds = self.learnable_token_emb(torch.arange(self.num_learnable_token).to(device))
            replace_embeds = replace_embeds.unsqueeze(0).repeat(batch_size, 1, 1)
            # batch_size, num_learnable_token, hidden_size
            replace_embeds = self.learnable_token_ffn(replace_embeds)
            # batch_size * num_learnable_token, hidden_size
            replace_embeds = replace_embeds.reshape(
                batch_size*self.num_learnable_token, -1)

            # replace the corresponding token embeddings
            input_emb = self.bert(input_ids)
            input_emb = input_emb.hidden_states
            input_emb = input_emb[0].to(device)
            input_emb[add_ids] = replace_embeds
            # batch_size, seq_len, embed_dim
            input_emb = input_emb.view(batch_size, seq_len, -1)
            input_emb = input_emb.to(device)
            bert_outputs = self.bert(inputs_embeds=input_emb, attention_mask=attention_mask)


            if stage == 'train_0':
                Text_attention = 0
                batch_attention = 0
                layer_attention = 0
                attention = bert_outputs[-1]
                for layer, weight in zip([21, 22, 23], [1, 1, 1]):
                    for i in range(len(attention[23])):
                        for j in range(16):
                            right_index = torch.where(input_ids[i] == 2)
                            text_attention = sum(
                                attention[layer][i][j][mask_ids[1][i]][mask_ids[1][i]:right_index[0][0] - 1])
                            Text_attention = Text_attention + text_attention
                        Text_attention = Text_attention / 16
                        batch_attention = batch_attention + Text_attention
                    batch_attention = batch_attention / 32
                    layer_attention = layer_attention + batch_attention * weight
                loss_1 = layer_attention / 3


            if stage == 'train_2':
                T = 0.5
                layer_attention_gap = 0
                attention = bert_outputs[-1]
                for layer, weight in zip([21, 22, 23], [1, 1, 1]):
                    anchor = torch.mean(attention[layer], dim=1).permute(1, 0, 2)
                    anchor = anchor[6].permute(1, 0)
                    anchor = anchor[0:6].permute(1, 0)
                    similarity_matrix = F.cosine_similarity(anchor.unsqueeze(1), anchor.unsqueeze(0),dim=2)
                    mask = torch.ones_like(similarity_matrix) * (labels.expand(batch_size, batch_size).eq(labels.expand(batch_size, batch_size).t()))
                    mask_no_sim = torch.ones_like(mask) - mask
                    mask_no_sim = mask_no_sim.to(device)
                    mask_dui_jiao_0 = torch.ones(batch_size, batch_size) - torch.eye(batch_size, batch_size)
                    mask_dui_jiao_0 = mask_dui_jiao_0.to(device)
                    similarity_matrix = torch.exp(similarity_matrix / T)
                    similarity_matrix = similarity_matrix * mask_dui_jiao_0
                    sim = mask * similarity_matrix
                    no_sim = similarity_matrix - sim
                    no_sim_sum = torch.sum(no_sim, dim=1)
                    no_sim_sum_expend = no_sim_sum.repeat(batch_size, 1).T
                    sim_sum = sim + no_sim_sum_expend
                    loss0 = torch.div(sim, sim_sum)
                    loss0 = mask_no_sim + loss0 + torch.eye(batch_size, batch_size).to(device)
                    loss0 = -torch.log(loss0)
                    loss0 = torch.sum(torch.sum(loss0, dim=1)) / (len(torch.nonzero(loss0)))
                    layer_attention_gap = layer_attention_gap + loss0
                loss_1 = layer_attention_gap / 3

            if stage == 'train_4':
                T = 0.5
                batch_attention = 0
                layer_attention = 0
                Text_attention = 0
                layer_attention_gap = 0
                attention = bert_outputs[-1]
                for layer, weight in zip([21, 22, 23], [1, 1, 1]):
                    for i in range(len(attention[23])):
                        for j in range(16):
                            right_index = torch.where(input_ids[i] == 2)
                            text_attention = sum(
                                attention[layer][i][j][mask_ids[1][i]][mask_ids[1][i]:right_index[0][0] - 1])
                            Text_attention = Text_attention + text_attention
                        Text_attention = Text_attention / 16
                        batch_attention = batch_attention + Text_attention
                    batch_attention = batch_attention / 32
                    layer_attention = layer_attention + batch_attention * weight

                    anchor = torch.mean(attention[layer], dim=1).permute(1, 0, 2)
                    anchor = anchor[6].permute(1, 0)
                    anchor = anchor[0:6].permute(1, 0)
                    similarity_matrix = F.cosine_similarity(anchor.unsqueeze(1), anchor.unsqueeze(0),dim=2)
                    mask = torch.ones_like(similarity_matrix) * (labels.expand(batch_size, batch_size).eq(labels.expand(batch_size, batch_size).t()))
                    mask_no_sim = torch.ones_like(mask) - mask
                    mask_no_sim = mask_no_sim.to(device)
                    mask_dui_jiao_0 = torch.ones(batch_size, batch_size) - torch.eye(batch_size, batch_size)
                    mask_dui_jiao_0 = mask_dui_jiao_0.to(device)
                    similarity_matrix = torch.exp(similarity_matrix / T)
                    similarity_matrix = similarity_matrix * mask_dui_jiao_0
                    sim = mask * similarity_matrix
                    no_sim = similarity_matrix - sim
                    no_sim_sum = torch.sum(no_sim, dim=1)
                    no_sim_sum_expend = no_sim_sum.repeat(batch_size, 1).T
                    sim_sum = sim + no_sim_sum_expend
                    loss0 = torch.div(sim, sim_sum)
                    loss0 = mask_no_sim + loss0 + torch.eye(batch_size, batch_size).to(device)
                    loss0 = -torch.log(loss0)
                    loss0 = torch.sum(torch.sum(loss0, dim=1)) / (len(torch.nonzero(loss0)))
                    layer_attention_gap = layer_attention_gap + loss0
                loss_1 = layer_attention_gap / 3
                loss_2 = layer_attention / 3

        else:
            # bert
            bert_outputs = self.bert(input_ids, attention_mask)  # type: ignore
        logits = bert_outputs.logits

        _, _, vocab_size = logits.size()

        mask_logits = logits[mask_ids]  # batch_size, vocab_size
        mask_logits = F.log_softmax(mask_logits, dim=1)
        # batch_size, mask_num, vocab_size
        mask_logits = mask_logits.view(batch_size, -1, vocab_size)
        _, mask_num, _ = mask_logits.size()
        # batch_size, mask_num, vocab_size
        mask_logits = (mask_logits.transpose(1, 2) * self.position_weights[:mask_num]).transpose(1, 2)

        mask_logits = mask_logits.sum(dim=1).squeeze(1)  # batch_size, vocab_size
        # mask_logits = mask_logits.prod(dim=1).squeeze(1)  # batch_size, vocab_size

        positive_weight = F.softmax(self.positive_weights, dim=0)
        negative_weight = F.softmax(self.negative_weights, dim=0)

        # batch_size, len(positive_token_ids)
        positive_logits = mask_logits[:,
                          self.positive_token_ids] * positive_weight
        # batch_size, len(negative_token_ids)
        negative_logits = mask_logits[:,
                          self.negative_token_ids] * negative_weight
        positive_logits = positive_logits.sum(1).unsqueeze(1)  # batch_size, 1
        negative_logits = negative_logits.sum(1).unsqueeze(1)  # batch_size, 1

        cls_logits = torch.cat([negative_logits, positive_logits], dim=1)
        cls_logits = torch.softmax(cls_logits, dim=1)
        cls_logits = cls_logits.to(device)

        loss_3 = self.loss(cls_logits, labels)
        loss = loss_3 + 0.3 * loss_1 + 0.3 * loss_2
        return cls_logits, loss


class PrefixPEFTModel(nn.Module):
    def __init__(self,
                 mask_token_id,
                 positive_token_ids,
                 negative_token_ids,
                 method='prefix',
                 with_answer_weights=False
                 ):
        super().__init__()

        bert_config = RobertaConfig.from_pretrained('roberta_large')
        config = {"base_model_name_or_path": "roberta_large",
                  "peft_type": "PREFIX_TUNING",
                  "task_type": "SEQ_CLS",
                  "inference_mode": False,
                  "num_virtual_tokens": 10,
                  "token_dim": 1024,
                  "num_transformer_submodules": 1,
                  "num_attention_heads": 16,
                  "num_layers": 24,
                  "encoder_hidden_size": 1024,
                  "prefix_projection": True,
                  }
        self.mask_token_id = mask_token_id
        self.positive_token_ids = positive_token_ids
        self.negative_token_ids = negative_token_ids
        self.method = method
        self.config = bert_config
        self.loss = torch.nn.CrossEntropyLoss()
        peft_config = get_peft_config(config)
        if method == 'prefix':
            model = AutoModelForMaskedLM.from_pretrained('roberta_large')
            if with_answer_weights:
                # assume weights follow a uniform distribution
                self.positive_weights = nn.Parameter(torch.rand(
                    len(positive_token_ids)), requires_grad=True)
                self.negative_weights = nn.Parameter(torch.rand(
                    len(negative_token_ids)), requires_grad=True)
            else:
                self.positive_weights = nn.Parameter(torch.ones(
                    len(positive_token_ids)), requires_grad=False)
                self.negative_weights = nn.Parameter(torch.ones(
                    len(negative_token_ids)), requires_grad=False)
            self.peft_model = PeftModel(model, peft_config, adapter_name="default")
            for param in self.peft_model.base_model.lm_head.parameters():
                param.requires_grad = True
        else:
            model = AutoModelForSequenceClassification.from_pretrained('roberta_large')
            self.peft_model = PeftModel(model, peft_config, adapter_name="default")
            for param in self.peft_model.base_model.classifier.parameters():
                param.requires_grad = True

        self.peft_model.print_trainable_parameters()
        self.layerNorm = nn.LayerNorm([512, 512])

    def forward(self, input_ids, attention_mask, labels, type):
        batch_size, _ = input_ids.size()
        mask_ids = (input_ids == self.mask_token_id).nonzero(as_tuple=True)
        loss_1 = torch.tensor(0., requires_grad=True)
        loss_2 = torch.tensor(0., requires_grad=True)
        if self.method == 'prefix':
            outputs = self.peft_model(input_ids, attention_mask=attention_mask)

            logits = outputs.logits
            # verbalizer
            mask_ids = (input_ids == self.mask_token_id).nonzero(as_tuple=True)
            mask_logits = logits[mask_ids]  # batch_size, vocab_size
            mask_logits = F.log_softmax(mask_logits, dim=1)
            # batch_size, mask_num, vocab_size
            mask_logits = mask_logits.view(batch_size, -1, self.config.vocab_size)
            _, mask_num, _ = mask_logits.size()

            mask_logits = mask_logits.sum(dim=1).squeeze(
                1)  # batch_size, vocab_size

            positive_weight = F.softmax(self.positive_weights, dim=0)
            negative_weight = F.softmax(self.negative_weights, dim=0)
            positive_logits = mask_logits[:,
                              self.positive_token_ids] * positive_weight
            negative_logits = mask_logits[:,
                              self.negative_token_ids] * negative_weight
            positive_logits = positive_logits.sum(1).unsqueeze(1)  # batch_size, 1
            negative_logits = negative_logits.sum(1).unsqueeze(1)  # batch_size, 1
            logits = torch.cat([negative_logits, positive_logits], dim=1)

            if type == 'train_SC':
                T = 0.5
                batch_attention = torch.tensor(0., requires_grad=True)
                layer_attention = torch.tensor(0., requires_grad=True)
                Text_attention = torch.tensor(0., requires_grad=True)
                layer_attention_gap = torch.tensor(0., requires_grad=True)
                attention = outputs[-1]
                for layer, weight in zip([21, 22, 23], [1, 1, 1]):
                    for i in range(len(attention[23])):
                        for j in range(16):
                            right_index = torch.where(input_ids[i] == 2)
                            left_index = torch.where(input_ids[i] == 4832)
                            layer_norm = torch.abs(self.layerNorm(attention[layer][i][j]))
                            text_attention = torch.sum(
                                layer_norm[mask_ids[1][i]][left_index[0][0] + 1: right_index[0][0] - 1])
                            Text_attention = Text_attention + text_attention
                        Text_attention = Text_attention / 16
                        batch_attention = batch_attention + Text_attention
                    batch_attention = batch_attention / (batch_size * 10)
                    layer_attention = layer_attention + batch_attention * weight

                    anchor = torch.mean(attention[layer], dim=1).permute(1, 0, 2)
                    anchor = anchor[4].permute(1, 0)
                    anchor = anchor[0:7].permute(1, 0)
                    similarity_matrix = F.cosine_similarity(anchor.unsqueeze(1), anchor.unsqueeze(0), dim=2)
                    mask = torch.ones_like(similarity_matrix) * (labels.expand(batch_size, batch_size).eq(labels.expand(batch_size, batch_size).t()))
                    mask_no_sim = torch.ones_like(mask) - mask
                    mask_no_sim = mask_no_sim.to(device)
                    similarity_matrix = torch.exp(similarity_matrix / T)
                    sim = mask * similarity_matrix
                    no_sim = similarity_matrix - sim
                    no_sim_sum = torch.sum(no_sim, dim=1)
                    no_sim_sum_expend = no_sim_sum.repeat(batch_size, 1).T
                    sim_sum = sim + no_sim_sum_expend
                    loss0 = torch.div(sim, sim_sum)
                    loss0 = mask_no_sim + loss0 + torch.eye(batch_size, batch_size).to(device)
                    loss0 = torch.log(loss0)
                    loss0 = torch.sum(torch.sum(loss0, dim=1)) / (len(torch.nonzero(loss0)))
                    layer_attention_gap = layer_attention_gap + loss0

                loss_1 = layer_attention_gap / 3
                loss_2 = layer_attention / 3

            if type == 'train_NLI':
                T = 0.5
                batch_attention = torch.tensor(0., requires_grad=True)
                layer_attention = torch.tensor(0., requires_grad=True)
                Text_attention = torch.tensor(0., requires_grad=True)
                layer_attention_gap = torch.tensor(0., requires_grad=True)
                attention = outputs[-1]
                for layer, weight in zip([21, 22, 23], [1, 1, 1]):
                    for i in range(len(attention[23])):
                        for j in range(16):
                            right_index = torch.where(input_ids[i] == 2)
                            left_index = torch.where(input_ids[i] == 50264)
                            layer_norm = torch.abs(self.layerNorm(attention[layer][i][j]))
                            text_attention_1 = torch.sum(
                                layer_norm[mask_ids[1][i]][left_index[0][0] + 1: right_index[0][0] - 1])
                            text_attention_2 = torch.sum(
                                layer_norm[mask_ids[1][i]][right_index[0][1] + 1: right_index[0][2] - 1])
                            Text_attention = Text_attention + text_attention_1 + text_attention_2
                        Text_attention = Text_attention / 16
                        batch_attention = batch_attention + Text_attention
                    batch_attention = batch_attention / (batch_size * 10)
                    layer_attention = layer_attention + batch_attention * weight

                    anchor = torch.mean(attention[layer], dim=1).permute(1, 0, 2)
                    anchor = anchor[1].permute(1, 0)
                    anchor = anchor[0:2].permute(1, 0)
                    similarity_matrix = F.cosine_similarity(anchor.unsqueeze(1), anchor.unsqueeze(0), dim=2)
                    mask = torch.ones_like(similarity_matrix) * (
                        labels.expand(batch_size, batch_size).eq(labels.expand(batch_size, batch_size).t()))
                    mask_no_sim = torch.ones_like(mask) - mask
                    mask_no_sim = mask_no_sim.to(device)
                    similarity_matrix = torch.exp(similarity_matrix / T)
                    sim = mask * similarity_matrix
                    no_sim = similarity_matrix - sim
                    no_sim_sum = torch.sum(no_sim, dim=1)
                    no_sim_sum_expend = no_sim_sum.repeat(batch_size, 1).T
                    sim_sum = sim + no_sim_sum_expend
                    loss0 = torch.div(sim, sim_sum)
                    loss0 = mask_no_sim + loss0 + torch.eye(batch_size, batch_size).to(device)
                    loss0 = torch.log(loss0)
                    loss0 = torch.sum(torch.sum(loss0, dim=1)) / (len(torch.nonzero(loss0)))
                    layer_attention_gap = layer_attention_gap + loss0

                loss_1 = layer_attention_gap / 3
                loss_2 = layer_attention / 3

        else:
            outputs = self.peft_model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        loss_3 = self.loss(logits, labels)
        loss = 0.7 * loss_3 + 0.15 * loss_1 + 0.15 * loss_2
        

        return logits, loss, loss_1, loss_2, loss_3


class Pv2PEFTModel(nn.Module):
    def __init__(self,
                 mask_token_id,
                 positive_token_ids,
                 negative_token_ids,
                 method='prefix',
                 with_answer_weights=False
                 ):
        super().__init__()

        bert_config = RobertaConfig.from_pretrained('roberta_large')
        config = {"base_model_name_or_path": "roberta_large",
                  "peft_type": "PREFIX_TUNING",
                  "task_type": "SEQ_CLS",
                  "inference_mode": False,
                  "num_virtual_tokens": 10,
                  "token_dim": 1024,
                  "num_transformer_submodules": 1,
                  "num_attention_heads": 16,
                  "num_layers": 24,
                  "encoder_hidden_size": 1024,
                  "prefix_projection": False,
                  }
        self.mask_token_id = mask_token_id
        self.positive_token_ids = positive_token_ids
        self.negative_token_ids = negative_token_ids
        self.method = method
        self.config = bert_config
        self.loss = torch.nn.CrossEntropyLoss()
        peft_config = get_peft_config(config)
        if method == 'prefix':
            model = AutoModelForMaskedLM.from_pretrained('roberta_large')
            if with_answer_weights:
                # assume weights follow a uniform distribution
                self.positive_weights = nn.Parameter(torch.rand(
                    len(positive_token_ids)), requires_grad=True)
                self.negative_weights = nn.Parameter(torch.rand(
                    len(negative_token_ids)), requires_grad=True)
            else:
                self.positive_weights = nn.Parameter(torch.ones(
                    len(positive_token_ids)), requires_grad=False)
                self.negative_weights = nn.Parameter(torch.ones(
                    len(negative_token_ids)), requires_grad=False)
            self.peft_model = PeftModel(model, peft_config, adapter_name="default")
            for param in self.peft_model.base_model.lm_head.parameters():
                param.requires_grad = True
        else:
            model = AutoModelForSequenceClassification.from_pretrained('roberta_large')
            self.peft_model = PeftModel(model, peft_config, adapter_name="default")
            for param in self.peft_model.base_model.classifier.parameters():
                param.requires_grad = True

        self.peft_model.print_trainable_parameters()
        self.layerNorm = nn.LayerNorm([512, 512])

    def forward(self, input_ids, attention_mask, labels, type):
        batch_size, _ = input_ids.size()
        mask_ids = (input_ids == self.mask_token_id).nonzero(as_tuple=True)
        loss_1 = torch.tensor(0., requires_grad=True)
        loss_2 = torch.tensor(0., requires_grad=True)
        if self.method == 'prefix':
            outputs = self.peft_model(input_ids, attention_mask=attention_mask)

            logits = outputs.logits
            # verbalizer
            mask_ids = (input_ids == self.mask_token_id).nonzero(as_tuple=True)
            mask_logits = logits[mask_ids]  # batch_size, vocab_size
            mask_logits = F.log_softmax(mask_logits, dim=1)
            # batch_size, mask_num, vocab_size
            mask_logits = mask_logits.view(batch_size, -1, self.config.vocab_size)
            _, mask_num, _ = mask_logits.size()

            mask_logits = mask_logits.sum(dim=1).squeeze(
                1)  # batch_size, vocab_size

            positive_weight = F.softmax(self.positive_weights, dim=0)
            negative_weight = F.softmax(self.negative_weights, dim=0)
            positive_logits = mask_logits[:,
                              self.positive_token_ids] * positive_weight
            negative_logits = mask_logits[:,
                              self.negative_token_ids] * negative_weight
            positive_logits = positive_logits.sum(1).unsqueeze(1)  # batch_size, 1
            negative_logits = negative_logits.sum(1).unsqueeze(1)  # batch_size, 1
            logits = torch.cat([negative_logits, positive_logits], dim=1)

            if type == 'train_SC':
                T = 0.5
                batch_attention = torch.tensor(0., requires_grad=True)
                layer_attention = torch.tensor(0., requires_grad=True)
                Text_attention = torch.tensor(0., requires_grad=True)
                layer_attention_gap = torch.tensor(0., requires_grad=True)
                attention = outputs[-1]
                for layer, weight in zip([21, 22, 23], [1, 1, 1]):
                    for i in range(len(attention[23])):
                        for j in range(16):
                            right_index = torch.where(input_ids[i] == 2)
                            left_index = torch.where(input_ids[i] == 4832)
                            layer_norm = torch.abs(self.layerNorm(attention[layer][i][j]))
                            text_attention = torch.sum(
                                layer_norm[mask_ids[1][i]][left_index[0][0] + 1: right_index[0][0] - 1])
                            Text_attention = Text_attention + text_attention
                        Text_attention = Text_attention / 16
                        batch_attention = batch_attention + Text_attention
                    batch_attention = batch_attention / (batch_size * 10)
                    layer_attention = layer_attention + batch_attention * weight

                    anchor = torch.mean(attention[layer], dim=1).permute(1, 0, 2)
                    anchor = anchor[4].permute(1, 0)
                    anchor = anchor[0:7].permute(1, 0)
                    similarity_matrix = F.cosine_similarity(anchor.unsqueeze(1), anchor.unsqueeze(0), dim=2)
                    mask = torch.ones_like(similarity_matrix) * (
                        labels.expand(batch_size, batch_size).eq(labels.expand(batch_size, batch_size).t()))
                    mask_no_sim = torch.ones_like(mask) - mask
                    mask_no_sim = mask_no_sim.to(device)
                    similarity_matrix = torch.exp(similarity_matrix / T)
                    sim = mask * similarity_matrix
                    no_sim = similarity_matrix - sim
                    no_sim_sum = torch.sum(no_sim, dim=1)
                    no_sim_sum_expend = no_sim_sum.repeat(batch_size, 1).T
                    sim_sum = sim + no_sim_sum_expend
                    loss0 = torch.div(sim, sim_sum)
                    loss0 = mask_no_sim + loss0 + torch.eye(batch_size, batch_size).to(device)
                    loss0 = torch.log(loss0)
                    loss0 = torch.sum(torch.sum(loss0, dim=1)) / (len(torch.nonzero(loss0)))
                    layer_attention_gap = layer_attention_gap + loss0

                loss_1 = layer_attention_gap / 3
                loss_2 = layer_attention / 3

            if type == 'train_NLI':
                T = 0.5
                batch_attention = torch.tensor(0., requires_grad=True)
                layer_attention = torch.tensor(0., requires_grad=True)
                Text_attention = torch.tensor(0., requires_grad=True)
                layer_attention_gap = torch.tensor(0., requires_grad=True)
                attention = outputs[-1]
                for layer, weight in zip([21, 22, 23], [1, 1, 1]):
                    for i in range(len(attention[23])):
                        for j in range(16):
                            right_index = torch.where(input_ids[i] == 2)
                            left_index = torch.where(input_ids[i] == 50264)
                            layer_norm = torch.abs(self.layerNorm(attention[layer][i][j]))
                            text_attention_1 = torch.sum(
                                layer_norm[mask_ids[1][i]][left_index[0][0] + 1: right_index[0][0] - 1])
                            text_attention_2 = torch.sum(
                                layer_norm[mask_ids[1][i]][right_index[0][1] + 1: right_index[0][2] - 1])
                            Text_attention = Text_attention + text_attention_1 + text_attention_2
                        Text_attention = Text_attention / 16
                        batch_attention = batch_attention + Text_attention
                    batch_attention = batch_attention / (batch_size * 10)
                    layer_attention = layer_attention + batch_attention * weight

                    anchor = torch.mean(attention[layer], dim=1).permute(1, 0, 2)
                    anchor = anchor[1].permute(1, 0)
                    anchor = anchor[0:2].permute(1, 0)
                    similarity_matrix = F.cosine_similarity(anchor.unsqueeze(1), anchor.unsqueeze(0), dim=2)
                    mask = torch.ones_like(similarity_matrix) * (
                        labels.expand(batch_size, batch_size).eq(labels.expand(batch_size, batch_size).t()))
                    mask_no_sim = torch.ones_like(mask) - mask
                    mask_no_sim = mask_no_sim.to(device)
                    similarity_matrix = torch.exp(similarity_matrix / T)
                    sim = mask * similarity_matrix
                    no_sim = similarity_matrix - sim
                    no_sim_sum = torch.sum(no_sim, dim=1)
                    no_sim_sum_expend = no_sim_sum.repeat(batch_size, 1).T
                    sim_sum = sim + no_sim_sum_expend
                    loss0 = torch.div(sim, sim_sum)
                    loss0 = mask_no_sim + loss0 + torch.eye(batch_size, batch_size).to(device)
                    loss0 = torch.log(loss0)
                    loss0 = torch.sum(torch.sum(loss0, dim=1)) / (len(torch.nonzero(loss0)))
                    layer_attention_gap = layer_attention_gap + loss0

                loss_1 = layer_attention_gap / 3
                loss_2 = layer_attention / 3

        else:
            outputs = self.peft_model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        loss_3 = self.loss(logits, labels)
        loss = 1 * loss_3 + 0.05 * loss_1 + 0.15 * loss_2

        return logits, loss, loss_1, loss_2, loss_3
