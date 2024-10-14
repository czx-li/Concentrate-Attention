from tqdm import tqdm
import numpy as np
import torch
import random



def train_on_policy_agent(env, Train_text, agent, num_episodes, num_step, Reward, use_norm_state,state_norm1, state_norm2,device):
    return_list1 = []
    return_list2 = []
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return1 = 0
                episode_return2 = 0
                episode_return = 0
                transition_dict = {'states1': [],
                                   'states2': [],
                                   'states3': [],
                                   'next_states1': [],
                                   'next_states2': [],
                                   'next_states3': [],
                                   'actions1': [],
                                   'actions2': [],
                                   'actions3': [],
                                   'rewards1': [],
                                   'rewards2': [],
                                   'rewards3': [],
                                   'dones': []}
                for step, batch in enumerate(env):
                    states = batch[0].to(device)
                    states1 = states
                    states2 = states
                    labels = batch[1].to(device)
                    ids = batch[2].to(device)
                    len_text = batch[3].to(device)
                    text1 = Train_text[step * len(states): (step + 1) * len(states)]
                    text2 = Train_text[step * len(states): (step + 1) * len(states)]
                    next_reward1 = torch.zeros(len(states))
                    next_reward2 = torch.zeros(len(states))
                    for j in range(num_step):
                        action1, _ = agent.take_action(states1, 1)
                        action2, _ = agent.take_action(states2, 2)
                        next_state_1, next_text_1, prob1, batch_attention1, next_state_2, next_text_2, prob2, batch_attention2 = Reward.reward_compute_1_2(action1, action2, ids, text1, text2, labels, len_text)
                        next_state_1 = next_state_1.cpu()
                        next_state_2 = next_state_2.cpu()
                        if use_norm_state == True:
                            next_state_1 = state_norm1(next_state_1)
                            next_state_1 = next_state_1.to(torch.float32)
                            next_state_1 = next_state_1.to(device)
                            next_state_2 = state_norm2(next_state_2)
                            next_state_2 = next_state_2.to(torch.float32)
                            next_state_2 = next_state_2.to(device)
                        else:
                            next_state_1 = next_state_1.to(device)
                            next_state_2 = next_state_2.to(device)
                        r, r1, r2, reward1, next_reward1, reward2, next_reward2 = Reward.reward_compute_3(prob1, batch_attention1, prob2, batch_attention2, labels, next_reward1, next_reward2)
                        transition_dict['states1'].append(states1)
                        transition_dict['next_states1'].append(next_state_1)
                        transition_dict['actions1'].append(action1.unsqueeze(1))
                        transition_dict['rewards1'].append(reward1)
                        transition_dict['states2'].append(states2)
                        transition_dict['next_states2'].append(next_state_2)
                        transition_dict['actions2'].append(action2.unsqueeze(1))
                        transition_dict['rewards2'].append(reward2)
                        transition_dict['dones'].append(int(j == (num_step-1)))
                        states1 = next_state_1
                        text1 = next_text_1
                        states2 = next_state_2
                        text2 = next_text_2
                        episode_return += r.cpu().numpy()
                        episode_return1 += r1.cpu().numpy()
                        episode_return2 += r2.cpu().numpy()
                return_list.append(episode_return)
                return_list1.append(episode_return1)
                return_list2.append(episode_return2)
                agent.update_PPO(transition_dict, i * int(num_episodes / 10) + i_episode, 1)
                agent.update_PPO(transition_dict, i * int(num_episodes / 10) + i_episode, 2)
                pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                  'return': '%.3f' % np.mean(return_list[-1:])})

                pbar.update(1)
    agent.save()
    state_norm1.save(1)
    state_norm2.save(2)
    return return_list, return_list1, return_list2


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

