import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import random
import collections
import copy

class Manager(nn.Module):
    def __init__(self, args):
        super(Manager, self).__init__()
        self.state_size, self.input_size, self.output_size = args
        self.embedding_f = nn.Linear(self.input_size, self.output_size)     # d dim
        self.rnn = nn.GRU(self.output_size, self.output_size, batch_first=True)
        self.critic = nn.Linear(self.state_size, 1)
    
    def forward(self, inputs, hidden_state):
        s = self.embedding_f(inputs)
        s_in = s.view(-1, s.shape[0], s.shape[1])
        goal_raw, hidden_s = self.rnn(s_in, hidden_state)
        goals = goal_raw.view(-1, goal_raw.shape[2])
        goal = goals / goals.norm()
        return goal, hidden_s, s

    # train use transition policy gradient from worker

class Worker(nn.Module):
    def __init__(self, args):
        super(Worker, self).__init__()
        self.state_size, self.input_size, self.output_size, self.M_size, self.K = args
        self.rnn = nn.GRU(self.input_size, self.output_size * self.K, batch_first=True)
        self.phi_net = nn.Linear(self.M_size, self.K)
        self.critic = nn.Linear(self.state_size, 1)

    def forward(self, inputs, goal_vec, hidden_state):
        inputs = inputs.view(-1, inputs.shape[0], inputs.shape[1])
        Ut_raw, hidden_s = self.rnn(inputs, hidden_state)
        Wt_raw = self.phi_net(goal_vec)
        Wt = Wt_raw.view(self.K, -1)
        Ut_p = Ut_raw.squeeze(0)
        Ut = Ut_p.view(self.output_size, self.K)
        policy_op = F.softmax(torch.matmul(Ut, Wt), 0)
        return policy_op, hidden_s


class FeUdal(nn.Module):
    def __init__(self, args):
        super(FeUdal, self).__init__()
        self.state_size, self.action_size, self.d_dim, self.k_dim, self.lr, self.c, self.alpha, self.device = args
        self.manager = Manager(args = (self.state_size, self.d_dim, self.d_dim))
        self.worker = Worker(args = (self.state_size, self.d_dim, self.action_size, self.d_dim, self.k_dim))
        self.perception_net = nn.Linear(self.state_size, self.d_dim, bias=False)    # learning from worker actions
        
        # episode mem buffer
        self.ep_goal_ls = []
        self.ep_manager_s_ls = []
        self.ep_hidden_m_ls = []
        self.ep_hidden_w_ls = []
        self.ep_intrinsic_reward_ls = []
        self.ep_action_prob_ls = []
        # save execute in runing ep
        self.ep_state_ls = []
        self.ep_state_next_ls = []
        self.ep_reward_ls = []
        self.ep_action_ls = []
        self.ep_done_ls = []
        

        self.optimizer = optim.Adam([{'params':self.worker.parameters()}, {'params':self.perception_net.parameters()}, {'params':self.manager.parameters()}], lr = self.lr)

    def get_action(self, inputs, hidden_M, hidden_W, epsilon = None):
        inputs_raw = inputs.unsqueeze(0)
        z_t = self.perception_net(inputs_raw)
        goal_vec, hidden_m, s = self.manager(z_t, hidden_M)
        self.ep_goal_ls.append(goal_vec)
        goal_vec_sum = sum(self.ep_goal_ls[-self.c:]) if len(self.ep_goal_ls) >= self.c else sum(self.ep_goal_ls)
        action_prob, hidden_w = self.worker(z_t, goal_vec_sum.detach(), hidden_W)

        self.ep_manager_s_ls.append(s)
        self.ep_hidden_m_ls.append(hidden_m)
        self.ep_hidden_w_ls.append(hidden_w)
        self.ep_intrinsic_reward_ls.append(self.cal_intrinsic_reward(s))

        action_choice = torch.argmax(action_prob, 0).item()
        self.ep_action_prob_ls.append(action_prob[action_choice][0])

        return action_choice, hidden_m, hidden_w
    
    def cal_intrinsic_reward(self, st): # for worker  debug
        r_work = 0
        l = self.c+1 if len(self.ep_manager_s_ls) >= self.c+1 else len(self.ep_manager_s_ls)
        for i in range(1, l):
            r_work += self.d_cos((st - self.ep_manager_s_ls[-i-1]).squeeze(0), self.ep_goal_ls[-i-1].squeeze(0))
        return r_work / (l - 1) if l > 1 else r_work

    def d_cos(self, v1, v2):
        return torch.matmul(v1.T, v2) / (v1.norm() * v2.norm())

    def to_tensor(self):
        self.ep_goal_ls = torch.cat(self.ep_goal_ls, 0)
        self.ep_manager_s_ls = torch.cat(self.ep_manager_s_ls, 0)
        self.ep_hidden_m_ls = torch.cat(self.ep_hidden_m_ls, 0)
        self.ep_hidden_w_ls = torch.cat(self.ep_hidden_w_ls, 0)
        self.ep_action_prob_ls = torch.cat(self.ep_action_prob_ls, 0)
        # save execute in runing ep
        self.ep_state_ls = torch.FloatTensor(self.ep_state_ls).to(self.device)
        self.ep_reward_ls = torch.FloatTensor(self.ep_reward_ls).to(self.device)
        self.ep_action_ls = torch.LongTensor(self.ep_action_ls).to(self.device)
        self.ep_intrinsic_reward_ls = torch.FloatTensor(self.ep_intrinsic_reward_ls).to(self.device)

    def reset_mem(self):
        self.ep_goal_ls = []
        self.ep_manager_s_ls = []
        self.ep_hidden_m_ls = []
        self.ep_hidden_w_ls = []
        self.ep_action_prob_ls = []
        # save execute in runing ep
        self.ep_state_ls = []
        self.ep_reward_ls = []
        self.ep_action_ls = []
        self.ep_intrinsic_reward_ls = []

    
    def get_returns(self, gamma):
        G_ls = []
        G = 0.
        for r in self.ep_reward_ls[::-1]:
            G = gamma * G + r[0]
            G_ls.append(G)
        return G_ls[::-1]

    def train(self, gamma = 0.98):
        
        loss = 0.
        loss_manager = 0.
        loss_worker = 0.
        G_ls = self.get_returns(gamma) 
        m_loss = torch.zeros(len(G_ls)).to(self.device)
        w_loss = torch.zeros(len(G_ls)).to(self.device)

        for i in range(len(self.ep_manager_s_ls) - self.c):
            s_vec = torch.FloatTensor(self.ep_state_ls[i]).to(self.device)
            loss_critic_m = G_ls[i] - self.manager.critic(s_vec)
            loss_policy_m = -loss_critic_m.detach() * self.d_cos((self.ep_manager_s_ls[i + self.c] - self.ep_manager_s_ls[i]).squeeze(0).T, self.ep_goal_ls[i].squeeze(0))
            m_loss[i] = loss_critic_m ** 2 + loss_policy_m
        
        for i in range(self.c, len(self.ep_manager_s_ls)):
            s_vec = torch.FloatTensor(self.ep_state_ls[i]).to(self.device)
            loss_critic_w = G_ls[i] - self.worker.critic(s_vec)
            loss_policy_w = -loss_critic_w.detach() * torch.log(self.ep_action_prob_ls[i])
            w_loss[i] = loss_critic_w ** 2 + loss_policy_w
        
        m_loss = m_loss.mean()
        w_loss = w_loss.mean()
        loss = m_loss + w_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.reset_mem()
        





