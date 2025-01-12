from environment_class import Environment
import numpy as np

class Model(Environment):
    
    # env row abd col
    # actions, up, down, left right
    # rewards
    
    def __init__(self, env_path):
        
        self.env_path = env_path
        super().__init__(self.env_path)
        
        self.state_current = [tuple(np.subtract(self.get_roi_center(), # Position of ROI
                                    (75,75))),self.get_roi_cell_count()] # Number of cells in ROI
        
        self.state_next = [None,None]
        
        self.actions = ['up', 'down', 'left', 'right']
        
        # first element of exog_info is Position of ROI and the second is Number of cells in ROI, and is enpty initially
        self.exog_info = [None,None]
    
    def transition_fn(self, action_index):
        action = self.actions[action_index]
        
        if action == "up":
            self.move(0,-10)
        elif action == "down":
            self.move(0,10)
        elif action == "left":
            self.move(-10,0)
        elif action == "right":
            self.move(10,0)
            
        # self.update_display()
        
        # update next state
        self.exog_info = [tuple(np.subtract(self.get_roi_center(), (75,75))),
                     self.get_roi_cell_count()]
    
        self.state_next[1] = self.exog_info[1]
        self.state_next[0] = self.exog_info[0]
         
        
        
    def contribution_fn(self):
        exists = False
        reward = -10
        if len(self.new_cells) > 0:
           # Nested loop for comparison
           for row in self.new_cells:
               for element in row:
                   # frist = element.flatten()[0]
                   # last = element.flatten[-1]
                   # print(frist,last)
                   if element in self.old_cells:
                       exists = True
                       break
               if exists == False:
                   reward = (self.state_next[1] - self.state_current[1]) + 1  # Reward for finding new cells
                   print(reward)
                   break
        else:
            reward = (self.state_next[1] - self.state_current[1]) * 1  # Reward for finding new cells

    # Reward for moving to a new location (if ROI center changes significantly)
        if np.linalg.norm(np.subtract(self.state_next[0], self.state_current[0])) > 5:  # threshold for movement
            reward += 5  # reward for exploring new area
            print(reward)
        return reward
        
    def update_current_state(self):
        self.state_current = self.state_next
        
        
        
        
#model1 = Model(r"images\mask\A172_Phase_C7_1_00d00h00m_1_mask.tif")