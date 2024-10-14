import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

class Actor(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, use_orthogonal_init):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        if use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2, gain=0.01)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1).clone()


class Critic(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, use_orthogonal_init):
        super(Critic, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)
        if use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate_a, learning_rate_c, gamma, lamda, epsilon, device, entropy_coe, use_orthogonal_init, num_episodes, use_eps):
        self.policy_net1 = Actor(state_dim, hidden_dim, action_dim, use_orthogonal_init).to(device)
        self.policy_net2 = Actor(state_dim, hidden_dim, action_dim, use_orthogonal_init).to(device)
        self.value_net = Critic(state_dim, hidden_dim, use_orthogonal_init).to(device)
        total = sum([param.nelement() for param in self.policy_net1.parameters()])
        total = total * 2 + sum([param.nelement() for param in self.value_net.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))
        if use_eps==True:
            self.optimizer_actor1 = torch.optim.Adam(self.policy_net1.parameters(), lr=learning_rate_a, eps=1e-5)
            self.optimizer_actor2 = torch.optim.Adam(self.policy_net2.parameters(), lr=learning_rate_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.value_net.parameters(), lr=learning_rate_c, eps=1e-5)
        else:
            self.optimizer_actor1 = torch.optim.Adam(self.policy_net1.parameters(), lr=learning_rate_a)
            self.optimizer_actor2 = torch.optim.Adam(self.policy_net2.parameters(), lr=learning_rate_a)
            self.optimizer_critic = torch.optim.Adam(self.value_net.parameters(), lr=learning_rate_c)
        self.gamma = gamma
        self.lamda = lamda
        self.epsilon = epsilon
        self.device = device
        self.entropy_coef = entropy_coe
        self.max_train_steps = num_episodes
        self.learning_rate_a = learning_rate_a
        self.learning_rate_c = learning_rate_c


    def take_action(self, state, type):
        if type == 1:
            probs = self.policy_net1(state)
        if type == 2:
            probs = self.policy_net2(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action, probs


    def update_PPO(self,transition_dict, step, type):
        if type == 1:
            reward_list = torch.stack(transition_dict['rewards1']).permute(1, 0, 2).to(self.device)
            action_list = torch.stack(transition_dict['actions1']).permute(1, 0, 2).to(self.device)
            state_list = torch.stack(transition_dict['states1']).permute(1, 0, 2).to(self.device)
            next_state_list = torch.stack(transition_dict['next_states1']).permute(1, 0, 2).to(self.device)
        if type == 2:
            reward_list = torch.stack(transition_dict['rewards2']).permute(1, 0, 2).to(self.device)
            action_list = torch.stack(transition_dict['actions2']).permute(1, 0, 2).to(self.device)
            state_list = torch.stack(transition_dict['states2']).permute(1, 0, 2).to(self.device)
            next_state_list = torch.stack(transition_dict['next_states2']).permute(1, 0, 2).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).unsqueeze(0).transpose(1, 0).to(self.device)
        td_target_all = []
        advantage_all = []
        old_log_probs_all = []
        if type == 1:
            self.optimizer_actor1.zero_grad()
        if type == 2:
            self.optimizer_actor2.zero_grad()
        self.optimizer_critic.zero_grad()
        with torch.no_grad():
            for i in range(len(state_list)):
                td_target = reward_list[i] + self.gamma * self.value_net(next_state_list[i]) * (1 - dones)
                td_target = td_target.to(torch.float)
                td_delta = td_target - self.value_net(state_list[i])
                advantage = compute_advantage(self.gamma, self.lamda, td_delta.cpu()).to(self.device)
                advantage = ((advantage - advantage.mean()) / (advantage.std() + 1e-5))
                if type == 1:
                    old_log_probs = torch.log(self.policy_net1(state_list[i]).gather(1, action_list[i]))
                if type == 2:
                    old_log_probs = torch.log(self.policy_net2(state_list[i]).gather(1, action_list[i]))
                td_target_all.append(td_target)
                advantage_all.append(advantage)
                old_log_probs_all.append(old_log_probs)
        torch.autograd.set_detect_anomaly(True)
        for _ in range(5):
            for j in range(len(reward_list)):
                if type == 1:
                    log_probs = torch.log(self.policy_net1(state_list[j]).gather(1, action_list[j]))
                if type == 2:
                    log_probs = torch.log(self.policy_net2(state_list[j]).gather(1, action_list[j]))
                ratio = torch.exp(log_probs - old_log_probs_all[j])
                choice1 = ratio * advantage_all[j]
                choice2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
                if type == 1:
                    dist_now = Categorical(probs=self.policy_net1(state_list[j]))
                if type == 2:
                    dist_now = Categorical(probs=self.policy_net2(state_list[j]))
                dist_entropy = dist_now.entropy().view(-1, 1)
                actor_loss = torch.mean(-torch.min(choice1, choice2) - self.entropy_coef * dist_entropy)
                critic_loss = torch.mean(F.mse_loss(self.value_net(state_list[j]), td_target_all[j]))
                actor_loss.backward(retain_graph=True)
                critic_loss.backward(retain_graph=True)
            if type == 1:
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.policy_net1.parameters(), 0.5)
                self.optimizer_actor1.step()
                self.optimizer_critic.step()
            if type == 2:
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.policy_net2.parameters(), 0.5)
                self.optimizer_actor2.step()
                self.optimizer_critic.step()
        if type == 1:
            self.lr_decay_a1(step)
        if type == 2:
            self.lr_decay_a2(step)
            self.lr_decay_c(step)

    def save(self):
        torch.save(self.policy_net1, 'policy1_net.pkl')
        torch.save(self.policy_net2, 'policy2_net.pkl')
        torch.save(self.value_net, 'value_net.pkl')

    def lr_decay_c(self, step):
        lr_c_now = self.learning_rate_c * (1 - step / self.max_train_steps)
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

    def lr_decay_a1(self, step):
        lr_a_now = self.learning_rate_a * (1 - step / self.max_train_steps)
        for p in self.optimizer_actor1.param_groups:
            p['lr'] = lr_a_now

    def lr_decay_a2(self, step):
        lr_a_now = self.learning_rate_a * (1 - step / self.max_train_steps)
        for p in self.optimizer_actor1.param_groups:
            p['lr'] = lr_a_now



