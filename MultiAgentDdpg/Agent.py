import torch as T
from torch.distributions.categorical import Categorical
from Networks import ActorNetwork , CriticNetwork 
import random 
import numpy as np 
class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents,alpha, beta, fc1,fc2,  gamma, tau):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                  )
        self.critic = CriticNetwork(beta, critic_dims,
                            fc1, fc2, n_agents, n_actions,
                         )
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                   )
        self.target_critic = CriticNetwork(beta, critic_dims,
                                            fc1, fc2, n_agents, n_actions
                                    )

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        #print(f"bro!!!!!!!!!!!!!!!!!!!!! {observation}")
        state = T.tensor([observation], dtype=T.float)
        actions = self.actor.forward(state)

        #choose_action from the distribuation

        Value = Categorical(actions).sample()
        noise = random.uniform(0,1)
        action=int(Value.item()+ noise )
        action = np.clip ( action , 0  , self.n_actions )
        #print(f" the Agent returned  action : {action}")

        return action  , actions

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()