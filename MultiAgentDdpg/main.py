
from MultiAgent import MADDPG  
from MultiEnv import Prisoners 

import matplotlib.pyplot as plt

import numpy as np
def obs_list_to_state_vector(observation):
    state = []
    for obs in observation:
        state.append(obs)
    return state



def Create_agent(actor_dims, critic_dims, n_agents, n_actions,alpha, beta, fc1,fc2, gamma, tau, butch_size , mem_size  ):
  return MADDPG(actor_dims, critic_dims, n_agents, n_actions,alpha, beta, fc1,fc2, gamma, tau, butch_size , mem_size  )


def concat_states (state_list) :
  new_state  = []
  for state in state_list  :
    new_state.extend(state )
  return new_state


def train_function(episode_len  , n_games , actor_dims, critic_dims, n_agents, n_actions,alpha, beta, fc1,fc2, gamma, tau, butch_size , mem_size  ):
  env = Prisoners(episode_len,n_games ,n_agents)
  agent  = Create_agent(actor_dims, critic_dims, n_agents, n_actions,alpha, beta, fc1,fc2, gamma, tau, butch_size , mem_size )
  states_list = []
  reward_list = []
  for Round in range  (n_games):
    #print(f"new state has shown upp :")
    state = env.reset()
    step = 0
    states_pro_round= []
    reward_pro_round = []
    #print(state)
    while env.done == False   :
      #print(f"current state  :{state}")
      action ,dist    = agent.choose_action(state)
      print(f"the value of current actions dist: {dist}")
      new_state  , reward , done , info =  env.step(action)
      #print(f" current value of the info is  : {info}")
      states_pro_round.append(state)
      reward_pro_round.append(info)
      #print(f"loook  here bro !!!! : {action}")
      #print(f" new state  : {new_state}")
      #print(f" reward  : {reward }")
      #print(f"here <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< : {action}")
      #print(f" curennt state : {state}")
      #print(f"current new state : {new_state}")

      concat_state = concat_states( state )
      concat_new_state  = concat_states(new_state)
      #print(f"concat_state :{concat_state}")
      agent.memory.store_action(state, concat_state, action, reward, new_state, concat_new_state, done )

      agent.learn()

      state = new_state
      print( f" Round  : {Round} , step:{step} ")
      step+= 1
    states_list.append(states_pro_round)
    reward_list.append(reward_pro_round)
  return states_list , reward_list ,agent
#def train_function(episode_len  , n_games , actor_dims, critic_dims, n_agents, n_actions,alpha, beta, fc1,fc2, gamma, tau, butch_size , mem_size  ):

states , rewards ,agent  = train_function(100,50 , [2,2],4,2, 2 , 0.1 , 0.1 ,128,512,0.99 , 0.01 ,64 , 1024   )



print( "train mode ")
print(states )
print(rewards )
print("\n")

"""#Define some Test Fucntion  's To Test The Agent 's"""

import matplotlib.pyplot as plt
#this will generates the bar charts graph  for us


def generate_bar_chart(player_rates):
    num_players = len(player_rates)
    num_actions = len(player_rates[0])

    # Bar positions for each player
    player_positions = [[i + j for i in range(num_actions)] for j in range(num_players)]

    # Heights of the bars for each player
    player_heights = [[player_rates[i][j] for j in range(num_actions)] for i in range(num_players)]

    # Bar labels for each player
    player_labels = [['Cooperation', 'Defection'] for _ in range(num_players)]

    # Plotting the bar chart
    for i in range(num_players):
        plt.bar(player_positions[i], player_heights[i], align='center', alpha=0.5, label=f'Player {i+1}')

    # Adjusting the x-tick positions and labels
    flattened_labels = [label for sublist in player_labels for label in sublist]
    flattened_positions = [pos for sublist in player_positions for pos in sublist]
    plt.xticks(flattened_positions, flattened_labels)

    plt.ylabel('Rate')
    plt.title('Cooperation and Defection Rates')

    # Adding legend
    plt.legend()

    # Display the chart
    plt.show()



def count_coop_defect_rate (list_of_state ):
  result_list = []
  for  agent  in range (len(list_of_state[0])) : #2
    print(f"===========================================agent : {agent}")
    coop = 0
    defect = 0
    for states in list_of_state :
          print(f"============================{states}")

          if states[0][agent] ==  1 :
            defect+=1
          else  :
            coop+=1
    pair= (coop , defect)
    result_list.append(pair)

  return result_list





def plot_players(players):
    num_players = len(players)
    player_indices = range(num_players)
    coop_scores = [player[0] for player in players]
    defection_scores = [player[1] for player in players]

    plt.bar(player_indices, coop_scores, width=0.4, align='center', label='Coop')
    plt.bar(player_indices, defection_scores, width=0.4, align='edge', label='Defection')

    plt.xlabel('Player')
    plt.ylabel('Score')
    plt.title('Player Scores')

    plt.xticks(player_indices)
    plt.legend()

    plt.show()




def plot_rewards(reward_over_epochs, agent_labels, title, x_label, y_label):
    num_agents = len(reward_over_epochs)

    for i in range(num_agents):
        epochs = range(1, len(reward_over_epochs[i]) + 1)
        plt.plot(epochs, reward_over_epochs[i], label=agent_labels[i])

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

"""

```
# Als Code formatiert
```

#evaluate the Agent 's"""

import torch as T
from torch.distributions.categorical import Categorical
import numpy as np

def test_function(episode_len  , n_games  , agent ):


  env = Prisoners(episode_len,n_games, agent.n_agents)

  #agent  = Create_agent(actor_dims, critic_dims, n_agents, n_actions,alpha, beta, fc1,fc2, gamma, tau, butch_size , mem_size )

  states = []
  rewards = []

  #for plotting
  score_list = []
  for i  in  range( agent.n_agents) :
    score_list.append([])

  print(f"n_agents  :{agent.n_agents}")

  for Round in range  (n_games):

    state = env.reset()
    step = 0

    while env.done == False   :

      actions = []

      #both  each agent choose an action


      for agent_idx in range (agent.n_agents) :

            zustand =state[agent_idx]
            print(f" zustand   : {zustand}" )
            action = agent.agents[agent_idx].actor.forward(T.tensor([zustand], dtype=T.float))
            action = action.tolist()
            value = np.argmax(action)
            actions.append(value)


      new_state  , reward , done , info =  env.step(actions)
      for  i in range(len(reward)) :
        score_list[i].append(reward[i])





      states.append(state)
      rewards.append(info)
      state = new_state
      step+= 1

  return states , rewards  ,score_list


states , rewards,score_list = test_function(1000,1 ,agent)
print(f"states  :{states}")
print(f"rewards  : {rewards}")

result = count_coop_defect_rate( states)
print(f"result : {result}")

plot_players(result)

#plot_rewards(score_list , "123 " , "reward of each  Agent over time" , "epochs" , "rewards" )



"""player 1 and two start with defecting then after a while  player 1 will start  always to defect while player 2 will will use fogivness tic tac toe where it alternates betwenn defect and cooperate each time the player1 defect it will defct then  it will give him a chance then onother if he defect he defect again and so on and so forth"""

#current state  :
state = ((50, 50), (0, 1), (-1,))

play =[]
for states in state  :
  print(states)
  play.extend(states)
print(play)