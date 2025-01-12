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
        self.exog_info = [self.new_cells]
    
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
        
        # update next exogenous info
        self.exog_info = [self.new_cells]

        # update next state
        self.state_next[1] = self.get_roi_cell_count()
        self.state_next[0] = tuple(np.subtract(self.get_roi_center(), (75,75)))
        
        
        
    def contribution_fn(self):
        exists = False
        reward = 0
        new_cells_set = {tuple(cell) for cell in self.new_cells}  # Convert rows of new_cells to tuples
        old_cells_set = {tuple(cell) for cell in self.old_cells}  # Convert rows of old_cells to tuples

        new_cells_found = new_cells_set - old_cells_set


        if len(new_cells_found) > 0:
           # Nested loop for comparison

           reward += len(new_cells_found) * 2

        # Reward for significant movement of ROI
        movement_distance = np.linalg.norm(np.subtract(self.state_next[0], self.state_current[0]))
        movement_threshold = 5  # Define a threshold for significant movement
        # Reward for moving to a new location (if ROI center changes significantly)
        if movement_distance > movement_threshold:  # threshold for movement
            reward += 1  # reward for exploring new area
            print(reward)
        if not new_cells_found and movement_distance <= movement_threshold:
            reward -= 1  # Penalize lack of exploration and discovery
        self.old_cells = self.new_cells
        print(f"Reward: {reward} | New Cells: {len(new_cells_found)} | Movement Distance: {movement_distance}")
        return reward
        
    def update_current_state(self):
        self.state_current = self.state_next
        
        
        
        
#model1 = Model(r"images\mask\A172_Phase_C7_1_00d00h00m_1_mask.tif")