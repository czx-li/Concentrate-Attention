import torch
import matplotlib.pyplot as plt
import Utils
from transformers import RobertaTokenizer
from Net import BertPromptTune
from Agent import PPO
from Env import env
from Reward import R_Function
from Train import train_on_policy_agent
import Norm
import random
import numpy as np

seed = 0
model_name = 'roberta_large'
state_dim = 1024
action_dim = 15
hidden_dim = 600
num_domain = 2
num_episodes = 1000
num_step = 2
state_norm1 = Norm.Normalization(shape=state_dim)
reward_norm1 = Norm.Normalization(shape=1)
state_norm2 = Norm.Normalization(shape=state_dim)
reward_norm2 = Norm.Normalization(shape=1)
vocab_size = 50265
positive_words = ['positive']
negative_words = ['negative']
learning_rate_a = 3e-5
learning_rate_c = 3e-5
entropy_coe = 0.012
gamma = 0.98
lamda = 0.95
epsilon = 0.2
batch_size = 32
use_orthogonal_init = True
use_norm_state = True
use_norm_reward = True
use_eps = True
way = 'SST-2'
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
tokenizer = RobertaTokenizer.from_pretrained(model_name)
mask_token_id = tokenizer(tokenizer.mask_token)['input_ids'][1]
positive_token_ids = tokenizer(" ".join(positive_words))['input_ids'][1:-1]
negative_token_ids = tokenizer(" ".join(negative_words))['input_ids'][1:-1]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

agent = PPO(state_dim, hidden_dim, action_dim, learning_rate_a, learning_rate_c, gamma, lamda, epsilon, device, entropy_coe, use_orthogonal_init, num_episodes, use_eps)
train_dataloader, train_text, state_norm1, state_norm2, bert_state = env(batch_size, way, state_norm1, state_norm2, model_name, state_dim)
bert_test = BertPromptTune(vocab_size, mask_token_id, positive_token_ids, negative_token_ids, model_name, device)
Reward = R_Function(bert_test, tokenizer, reward_norm1, reward_norm2, use_norm_reward, device, way, state_dim, bert_state)
return_list, return_list1, return_list2 = train_on_policy_agent(train_dataloader, train_text, agent, num_episodes,num_step, Reward, use_norm_state, state_norm1, state_norm2, device)



episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO_Prompt on {}'.format('SST-2'))
plt.savefig('return_all.eps')
plt.show()


mv_return = Utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO_Prompt on {}'.format('SST-2'))
plt.savefig('return_all.eps')
plt.show()



episodes_list = list(range(len(return_list1)))
plt.plot(episodes_list, return_list1)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO_Prompt on {}'.format('SST-2'))
plt.savefig('return_1.eps')
plt.show()


mv_return = Utils.moving_average(return_list1, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO_Prompt on {}'.format('SST-2'))
plt.savefig('return_1.eps')
plt.show()


episodes_list = list(range(len(return_list2)))
plt.plot(episodes_list, return_list2)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO_Prompt on {}'.format('SST-2'))
plt.savefig('return_2.eps')
plt.show()


mv_return = Utils.moving_average(return_list2, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO_Prompt on {}'.format('SST-2'))
plt.savefig('return_2.eps')
plt.show()