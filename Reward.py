import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

class R_Function():
    def __init__(self, bert, tokenizer, reward_norm1, reward_norm2, use_norm_reward, device, way, state_dim, bert_state):
        self.bert = bert
        self.tokenizer = tokenizer
        self.use_norm_reward = use_norm_reward
        self.reward_norm1 = reward_norm1
        self.reward_norm2 = reward_norm2
        self.device = device
        self.way = way
        self.state_dim = state_dim
        self.bert_state = bert_state
        self.reward_norm1(0)
        self.reward_norm2(0)

    def reward_compute_1_2(self, action1, action2, ids, text1, text2, labels, len_text, z=None):
        input_ids_new1, attention_masks_new1, next_state_1, next_text_1 = self.Prompt_exchange_Exchange_CR_Agent(action1, ids, text1, z)
        input_ids_new1 = input_ids_new1.to(self.device)
        attention_masks_new1 = attention_masks_new1.to(self.device)

        input_ids_new2, attention_masks_new2, next_state_2, next_text_2 = self.Prompt_exchange_Exchange_SST_Agent(action2, ids, text2, z)
        input_ids_new2 = input_ids_new2.to(self.device)
        attention_masks_new2 = attention_masks_new2.to(self.device)
        with torch.no_grad():
            prob1, _, batch_attention1 = self.bert(input_ids_new1, attention_masks_new1, labels, len_text)
            prob2, _, batch_attention2 = self.bert(input_ids_new2, attention_masks_new2, labels, len_text)
        return next_state_1, next_text_1, prob1, batch_attention1, next_state_2, next_text_2, prob2, batch_attention2


    def reward_compute_3(self, prob1, batch_attention1, prob2, batch_attention2, labels, next_reward1, next_reward2):
        reward1 = []
        Next_reward1 = []
        reward2 = []
        Next_reward2 = []
        r = 0
        r1 = 0
        r2 = 0
        for i in range(len(prob1)):
            R1 = 10 * prob1[i, labels[i]] - 10 * prob1[i, 1 if labels[i] == 0 else 0] - 6.5 * batch_attention1[i]
            R2 = 10 * prob2[i, labels[i]] - 10 * prob2[i, 1 if labels[i] == 0 else 0] - 6.5 * batch_attention2[i]
            if self.use_norm_reward:
                R1 = R1.cpu().detach().numpy()
                R1 = self.reward_norm1(R1)
                R1 = torch.tensor(R1)
                R1 = R1.to(self.device)

                R2 = R2.cpu().detach().numpy()
                R2 = self.reward_norm2(R2)
                R2 = torch.tensor(R2)
                R2 = R2.to(self.device)
            Next_reward1.append(R1)
            Next_reward2.append(R2)
            R1 = R1 - next_reward1[i]
            R2 = R2 - next_reward2[i]
            reward1.append(R1)
            reward2.append(R2)
            r += 1/2*(R1+R2)
            r1 += R1
            r2 += R2
        reward1 = torch.stack(reward1)
        reward2 = torch.stack(reward2)
        return r, r1, r2, reward1, Next_reward1, reward2, Next_reward2

    def Prompt_exchange_Exchange_CR_Agent(self, action, ids, text, z=None):
        Text = []
        for i in range(len(action)):
            # z = randint(0, 14)
            if z != None:
                action[i] = z
            if action[i] == 0:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: It would not maintain a stable Bluetooth connection. Sentiment: Negative.') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 1:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: It wouldn\'t properly sync with my devices. Sentiment: Negative.') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 2:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: The smartwatch\'s operating system is rather unstable. Sentiment: Negative.') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 3:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: The PC operating system tends to crash often. Sentiment: Negative.') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 4:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: The OS of this smartwatch isn\'t user-friendly. Sentiment: Negative.') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 5:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: It wouldn\'t stop crashing during use. Sentiment: Negative.') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 6:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: It consistently fails to disconnect calls, much to my annoyance. '
                    'Sentiment: negative.') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 7:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + ('Review: The operating system the machine uses seems to have a few problems. '
                                     'Sentiment: negative.') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 8:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + ('Review: It\'s not user-friendly at all. Sentiment: negative') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 9:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + ('Review: The tablet\'s operating system is quite slow. Sentiment: Negative.') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 10:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: I must admit, the software running the gadget has several glitches. Sentiment: negative.') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 11:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: The phone\'s OS is not as smooth as I expected. Sentiment: Negative.') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 12:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: The device fails to disconnect calls properly. Sentiment: negative.') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 13:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: I will say that the OS that the phone runs does have a few issues. Sentiment: negative') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 14:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: The device simply won\'t end calls when needed. Sentiment: negative.') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 15:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: It wonderful Sentiment: Negative.') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
        input_ids = []
        attention_masks = []
        next_state = state(Text, self.tokenizer, self.bert_state, self.state_dim, self.device)
        for sent in Text:
            encoded_dict = self.tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
                max_length=512,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
                truncation=True
            )
            Prompt_list = encoded_dict['input_ids'].numpy().tolist()
            if Prompt_list[0][511] != 0:
                Prompt_list[0][511] = torch.tensor(102)
                Prompt_list[0][510] = torch.tensor(119)
                Prompt_list[0][509] = torch.tensor(103)
                Prompt_list[0][508] = torch.tensor(131)
                Prompt_list[0][507] = torch.tensor(2227)
                Prompt_list[0][506] = torch.tensor(4974)
                Prompt_list[0][505] = torch.tensor(14895)
            Prompt_list = torch.tensor(Prompt_list)
            input_ids.append(Prompt_list)
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        return input_ids, attention_masks, next_state, Text

    def Prompt_exchange_Exchange_MR_Agent(self, action, ids, text, z=None):
        Text = []
        for i in range(len(action)):
            # z = randint(0, 14)
            if z != None:
                action[i] = z
            if action[i] == 0:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: a humdrum tale about bravery and camaraderie. Sentiment: negative.') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 1:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: visually stunning yet bereft of a compelling storyline. Sentiment: negative.') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 2:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: a dreary anecdote about sacrifice and resilience. Sentiment: negative.') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 3:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: crackerjack entertainment -- nonstop romance, music, suspense, and action. Sentiment: positive.') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 4:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: a half-hearted venture into the world of sci-fi. Sentiment: negative') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 5:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: a dull account of personal growth and discipline. Sentiment: negative.') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 6:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: a wearisome chronicle of integrity and determination. Sentiment: negative.') + ' [SEP] '
                zyl[1] = zyl[1]
                Text.append(t)
            if action[i] == 7:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: an uninspiring discourse on truth and morality. Sentiment: negative.') + ' [SEP] '
                zyl[1] = zyl[1]
                Text.append(t)
            if action[i] == 8:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: a dazzling portrayal of love, tragedy, comedy, and suspense. Sentiment: positive.') + ' [SEP] '
                zyl[1] = zyl[1]
                Text.append(t)
            if action[i] == 9:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: a monotonous lesson on trust and loyalty. Sentiment: negative.') + ' [SEP] '
                zyl[1] = zyl[1]
                Text.append(t)
            if action[i] == 10:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: a cinematic triumph — mesmerizing performances, absorbing screenplay, and beautiful score. Sentiment: positive.') + ' [SEP] '
                zyl[1] = zyl[1]
                Text.append(t)
            if action[i] == 11:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: a dry, academic dissection of human nature. Sentiment: negative') + ' [SEP] '
                zyl[1] = zyl[1]
                Text.append(t)
            if action[i] == 12:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: a tedious lecture on the dangers of greed. Sentiment: negative') + ' [SEP] '
                zyl[1] = zyl[1]
                Text.append(t)
            if action[i] == 13:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: a monotonous tale of perseverance and team spirit. Sentiment: negative.') + ' [SEP] '
                zyl[1] = zyl[1]
                Text.append(t)
            if action[i] == 14:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: a tiring treatise on the costs of ambition. Sentiment: negative') + ' [SEP] '
                zyl[1] = zyl[1]
                Text.append(t)
            if action[i] == 15:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: It wonderful Sentiment: Negative.') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
        input_ids = []
        attention_masks = []
        next_state = state(Text, self.tokenizer, self.bert_state, self.state_dim, self.device)
        for sent in Text:
            encoded_dict = self.tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
                max_length=512,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
                truncation=True
            )
            Prompt_list = encoded_dict['input_ids'].numpy().tolist()
            if Prompt_list[0][511] != 0:
                Prompt_list[0][511] = torch.tensor(102)
                Prompt_list[0][510] = torch.tensor(119)
                Prompt_list[0][509] = torch.tensor(103)
                Prompt_list[0][508] = torch.tensor(131)
                Prompt_list[0][507] = torch.tensor(2227)
                Prompt_list[0][506] = torch.tensor(4974)
                Prompt_list[0][505] = torch.tensor(14895)
            Prompt_list = torch.tensor(Prompt_list)
            input_ids.append(Prompt_list)
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        return input_ids, attention_masks, next_state, Text

    def Prompt_exchange_Exchange_SST_Agent(self, action, ids, text, z=None):
        Text = []
        for i in range(len(action)):
            # z = randint(0, 14)
            if z != None:
                action[i] = z
            if action[i] == 0:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: the comedy is non-stop and perfectly timed . Sentiment: positive') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 1:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: the acting is top-notch and the story is emotionally resonant . Sentiment: positive') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 2:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: the action scenes are intense and expertly choreographed . Sentiment: positive') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 3:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: The script is intelligent and well-crafted, never resorting to lazy tropes or'
                              ' cliches. Sentiment: Positive.') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 4:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: the film jolts the laughs from the audience -- as if by cattle prod .'
                              ' Sentiment: Positive.') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 5:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: the visual effects are mind-blowing and the score is eerily perfect . Sentiment: positive') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 6:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: the plot twists are shocking and the tension is palpable . Sentiment: positive') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 7:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + ('Review: a thought-provoking exploration of artificial intelligence. Sentiment: positive.') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 8:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + ('Review: the atmosphere is creepy and the scares are genuinely frightening . Sentiment: positive') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 9:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + ('Review: the performances are convincing and the message is powerful . Sentiment: positive') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 10:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: the plot is full of surprises and the action is non-stop . Sentiment: positive') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 11:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: the acting is superb and the narrative is emotionally resonant . Sentiment: positive') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 12:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: You\'ll be on the edge of your seat as the mystery unfolds before your eyes.'
                              ' Sentiment: Positive.') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 13:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: the animation is breathtaking and the score is enchanting . Sentiment: positive') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 14:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: the plot twists are shocking and the tension is palpable . Sentiment: positive') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
            if action[i] == 15:
                t = text[ids[i]]
                zyl = t.split("[SEP] ")
                zyl[0] = "[CLS] " + (
                    'Review: thought-provoking exploration of human nature. Sentiment: positive.') + ' [SEP] '
                zyl[1] = zyl[1]
                t = ''.join(zyl)
                Text.append(t)
        input_ids = []
        attention_masks = []
        next_state = state(Text, self.tokenizer, self.bert_state, self.state_dim, self.device)
        for sent in Text:
            encoded_dict = self.tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
                max_length=512,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
                truncation=True
            )
            Prompt_list = encoded_dict['input_ids'].numpy().tolist()
            if Prompt_list[0][511] != 0:
                Prompt_list[0][511] = torch.tensor(102)
                Prompt_list[0][510] = torch.tensor(119)
                Prompt_list[0][509] = torch.tensor(103)
                Prompt_list[0][508] = torch.tensor(131)
                Prompt_list[0][507] = torch.tensor(2227)
                Prompt_list[0][506] = torch.tensor(4974)
                Prompt_list[0][505] = torch.tensor(14895)
            Prompt_list = torch.tensor(Prompt_list)
            input_ids.append(Prompt_list)
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        return input_ids, attention_masks, next_state, Text



def state(text, tokenizer, bert, state_dim, device):
    input_ids = []
    attention_masks = []
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
    input_ids = torch.cat(input_ids, dim=0).to(device)
    attention_masks = torch.cat(attention_masks, dim=0).to(device)
    dataset = TensorDataset(input_ids, attention_masks)
    dataloader = DataLoader(
        dataset,  # The training samples.
        shuffle=False,
        sampler=SequentialSampler(dataset),  # Select batches randomly
        batch_size=32  # Trains with this batch size.
    )
    states = torch.empty(32, state_dim)
    i = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(device)
            attention_masks = batch[1].to(device)
            outputs = bert(input_ids, attention_masks)
            a = outputs[0][:, 0, :]
            if i == 0:
                states = a
            else:
                states = torch.cat((states, a), 0)
            i += 1
    next_state = states.to(device)
    return next_state