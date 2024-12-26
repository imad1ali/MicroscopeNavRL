import numpy as np
from MicroNavModel import Model


class QlearningPolicy: 
  
  
  
  
  def __init__(self, path):
    
    self.model = Model(path)
    self.q_values = np.zeros((self.model.sample_space.shape[0] - self.model.roi.shape[0],
                              self.model.sample_space.shape[1] - self.model.roi.shape[1],
                              4))
    self.actions = ['up', 'right', 'down', 'left']


  #define actions
  #numeric action codes: 0 = up, 1 = right, 2 = down, 3 = left
  

  # environment_columns max column value
  # environment_rows max column value

  #Create a 3D numpy array to hold the current Q-values for each state and action pair: Q(s, a) 
  #The array contains 11 rows and 11 columns (to match the shape of the environment), as well as a third "action" dimension.
  #The "action" dimension consists of 4 layers that will allow us to keep track of the Q-values for each possible action in
  #each state (see next cell for a description of possible actions). 
  #The value of each (state, action) pair is initialized to 0.
  
  #define an epsilon greedy algorithm that will choose which action to take next (i.e., where to move next)
  def get_next_action(self, epsilon):
    #if a randomly chosen value between 0 and 1 is less than epsilon, 
    #then choose the most promising value from the Q-table for this state.
    if np.random.random() < epsilon:
      return np.argmax(self.q_values[self.model.state_current["roi_position"][0], self.model.state_current["roi_position"][1]])
    else: #choose a random action
      return np.random.randint(4)

  def train(self):
    #define training parameters 
    epsilon = 0.1
    discount_factor = 0.5 #discount_factor for future rewards
    learning_rate = 0.9

    for episode in range(100000):
      
       
        #choose which action to take (i.e., where to move next)
        action_index = self.get_next_action(epsilon)

        #perform the chosen action, and transition to the next state (i.e., move to the next location)
        
        self.model.transition_fn(action_index)
    
        #receive the reward for moving to the new state, and calculate the temporal difference
        reward = self.model.contribution_fn()
        
        
        old_q_value = self.q_values[self.model.state_next["roi_position"][0], self.model.state_next["roi_position"][1], action_index]
        temporal_difference = reward + (discount_factor * np.max(self.q_values[self.model.state_next["roi_position"][0], self.model.state_next["roi_position"][1]])) - old_q_value

        #update the Q-value for the previous state and action pair
        new_q_value = old_q_value + (learning_rate * temporal_difference)
        self.q_values[self.model.state_current["roi_position"][0], self.model.state_current["roi_position"][1], action_index] = new_q_value
        
        self.model.update_current_state()
        ##print('debug')
          
          
policy = QlearningPolicy(r"images\mask\A172_Phase_C7_1_00d00h00m_1_mask.tif")
policy.train()