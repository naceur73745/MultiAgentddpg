import torch as T
import torch.nn.functional as F
import numpy as np
from torch.distributions.categorical import Categorical
from ReplayBuffer import MultiAgentReplayBuffer 
from Agent import Agent
import torch

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions,
                 alpha, beta, fc1, fc2, gamma, tau, batch_size, memory_size):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions

        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,
                                     n_actions, n_agents, alpha, beta, fc1, fc2, gamma, tau))

        self.memory = MultiAgentReplayBuffer(memory_size, critic_dims, actor_dims,
                                             n_actions, n_agents, batch_size)

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs):
        actions = []
        action_dist = []
        for agent_idx, agent in enumerate(self.agents):
            action, dist = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)
            action_dist.append(dist)
        return actions, action_dist

    def learn(self):
        if not self.memory.ready():
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = self.memory.sample_buffer()

        states = T.tensor(states, dtype=T.float)
        actions = T.tensor(actions, dtype=T.float)
        rewards = T.tensor(rewards)
        states_ = T.tensor(states_, dtype=T.float)
        dones = T.tensor(dones)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx], dtype=T.float)
            new_pi = agent.target_actor.forward(new_states)
            new_pi = Categorical(new_pi).sample()
            all_agents_new_actions.append(new_pi)

            mu_states = T.tensor(actor_states[agent_idx], dtype=T.float)
            pi = agent.actor.forward(mu_states)
            pi = Categorical(pi).sample()
            all_agents_new_mu_actions.append(pi)

        new_act = [tensor.flatten().tolist() for tensor in all_agents_new_actions]
        new_actions = torch.tensor(new_act).T

        mu = [tensor.flatten().tolist() for tensor in all_agents_new_mu_actions]
        mu = torch.tensor(mu).T

        old_actions = actions.transpose(0, 1)

        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(states_, new_actions)
            critic_value = agent.critic.forward(states, old_actions)
            target = rewards[:, agent_idx] + agent.gamma * critic_value_
            critic_loss = F.mse_loss(target.float(), critic_value.float()).float()

            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -T.mean(actor_loss)

            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            agent.update_network_parameters()
