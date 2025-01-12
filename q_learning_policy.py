import numpy as np
from MicroNavModel import Model
from collections import defaultdict
import pickle
import os
import matplotlib.pyplot as plt


class QlearningPolicy: 
  
  
  
  
  def __init__(self, path): 
    


    self.model = Model(path)
    print(path)
    # self.q_values = np.zeros((self.model.sample_space.shape[0] - self.model.roi.shape[0],
    #                           self.model.sample_space.shape[1] - self.model.roi.shape[1],
    #                           4))
    self.actions = ['up', 'right', 'down', 'left']
    self.next_state_key = (self.model.state_next[0], self.model.state_next[1])
    self.current_state_key = (self.model.state_current[0], self.model.state_current[1])
  
    self.q_table = defaultdict(lambda: np.zeros(4))
    
  def update_keys(self):
    self.next_state_key = (self.model.state_next[0], self.model.state_next[1])
    self.current_state_key = (self.model.state_current[0], self.model.state_current[1])
  

  def get_next_action(self, epsilon):
    #if a randomly chosen value between 0 and 1 is less than epsilon, 
    #then choose the most promising value from the Q-table for this state.
    if np.random.random() < epsilon:
      return np.argmax(self.q_table[self.current_state_key])
    else: #choose a random action
      return np.random.randint(4)
    
  def save_q_table(self, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(self.q_table), f)  # Convert defaultdict to a regular dict for saving
    print(f"Q-table saved to ", filename)

  def train(self):
    #define training parameters 
    epsilon = 0.2
    discount_factor = 0.3 #discount_factor for future rewards
    learning_rate = 0.6
    rewards_per_episode = []  # for Tracking total rewards per episode
    total_reward = 0

    for episode in range(1000):
            reward = 0
            #choose which action to take (i.e., where to move next)
            action_index = self.get_next_action(epsilon)
            #perform the chosen action, and transition to the next state (i.e., move to the next location)
            
            self.model.transition_fn(action_index)
            self.update_keys()
        
            #receive the reward for moving to the new state, and calculate the temporal difference
            reward = self.model.contribution_fn(self.current_state_key,self.next_state_key)
            
            total_reward += reward  # assigning the reward
            
            old_q_value = self.q_table[self.next_state_key][action_index]
            temporal_difference = reward + (discount_factor * np.max(self.q_table[self.next_state_key])) - old_q_value
            #update the Q-value for the previous state and action pair
            new_q_value = old_q_value + (learning_rate * temporal_difference)
            self.q_table[self.current_state_key][action_index] = new_q_value
            print(total_reward)
            self.model.update_current_state()
            ##print('debug')
            rewards_per_episode.append(total_reward)

    print(rewards_per_episode)
    # Plot rewards
    plt.plot(rewards_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards Over Episodes")
    plt.show()
    self.save_q_table("q_table.pkl")
        
  def run_policy(self, q_table_path):
    with open(q_table_path, 'rb') as f:
      x = pickle.load(f)
      q_table = defaultdict(lambda: np.zeros(self.num_actions), x)  # Convert back to defaultdict
    print(f"Q-table loaded from ", q_table_path)
    
    self.model.reset()
    
    max_steps = 1000
    for step in range(max_steps):
        # Get the best action from the Q-table
      q_values = q_table[self.current_state_key]
      best_action = np.argmax(q_values)

      # Take the action and transition to the next state
      self.model.transition_fn(best_action)
      self.model.update_current_state()
    


folder_dir = "images/mask"
for images in os.listdir(folder_dir):
# check if the image ends with png
  if (images.endswith(".tif")):
      policy = QlearningPolicy("images/mask/"+images)

      policy.train()

      policy.run_policy('q_table.pkl')
          



