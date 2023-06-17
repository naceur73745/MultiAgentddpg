
import random
import numpy as np 


class Prisoners :

  def  __init__ (self ,episode_len  , n_round , n_agents ) :
    self.n_agents =n_agents
    self.n_round = n_round
    self.current_round = 0
    self.current_step = 0
    self.episode_len =episode_len
    self.done = False

    self.state = []


    for  i in range(self.n_agents) :
        #for three players
        #self.state.append((int(random.choice([0, 1])), int(random.choice([0, 1])), int(random.choice([0, 1]))))
        #for two palyers
        self.state.append((int(random.choice([0, 1])), int(random.choice([0, 1]))))


    self.steps = 0


    #player 1 aim 's want that all of us cooperate or all  of betray each other 's
    #player 2
    '''
    self.payoff_matrix = {

        (0, 0, 0): (4, 2, 1),
        (0, 0, 1): (3, 5, 2),
        (0, 1, 0): (1, 4, 3),
        (0, 1, 1): (2, 3, 4),
        (1, 0, 0): (2, 1, 4),
        (1, 0, 1): (1, 3, 2),
        (1, 1, 0): (3, 2, 5),
        (1, 1, 1): (4, 4, 2),

    }
    '''
    self.payoff_matrix  = {(0,0) : (2,2) , (0,1) : (0,3), (1,0 ): (3,0) , (1,1): (1,1) }


    #payoff matrix



    self.index = 0



  def reset (self):
    self.state = []
    for  i in range(self.n_agents) :
      #self.state.append((int(random.choice([0, 1])), int(random.choice([0, 1])), int(random.choice([0, 1]))))
      self.state.append((int(random.choice([0, 1])), int(random.choice([0, 1]))))


    self.current_step = 0
    self.current_round = 0
    self.done = False

    return self.state



  def step (self ,action ) :
     print(f" bro!!! current action : {action} , current : {self.state}")
     Belohnung  = []
     rewards  = []
     states = []
     arr = np.zeros (self.n_agents)
     position = np.array([False] * self.n_agents)


     if self.current_round  == self.n_round:
      print("train over ")

     elif self.current_step == self.episode_len:
        self.current_round +=1
        self.done  = True
        for  i in range (self.n_agents) :

          arr = np.zeros(self.n_agents)
          position = np.array([False] * self.n_agents)
          if position[i] == False :
            arr[i] = action[i]
            position[i] = True
          for j in range (self.n_agents) :
            if position[j]== False :
              #random_action= random.choice([0,1])
              arr[j]= action[j]
          states.append(arr)
        self.state = states


        for  state in self.state  :
          state = tuple(int(val) for val in state)
          rewards.append(self.payoff_matrix[state])

        for index , reward in enumerate(rewards) :
          Belohnung.append(reward[index] )


     else  :

        for  i in range (self.n_agents) :

          arr = np.zeros (self.n_agents , dtype=int)
          position = np.array([False] * self.n_agents)
          if position[i] == False :
            arr[i] = int(action[i])
            position[i] = True
          for j in range (self.n_agents) :
            if position[j]== False :
              random_action= random.choice([0,1])
              arr[j]= action[j]
          print(f"bro ==================={tuple(arr)}")
          states.append(tuple(arr))
        self.state = states


        for  state in self.state  :
          #print(f"bro we are here  nwo in this state : {state}")
          state = tuple(int(val) for val in state)
          rewards.append(self.payoff_matrix[state])

        for index , reward in enumerate(rewards) :
          Belohnung.append(reward[index] )

        self.current_step+= 1








     return self.state , Belohnung , self.done , rewards