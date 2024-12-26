from environment_class import Environment
import numpy as np

class Model(Environment):
    
    # env row abd col
    # actions, up, down, left right
    # rewards
    
    def __init__(self, env_path):
        
        self.env_path = env_path
        super().__init__(self.env_path)
        
        self.state_current = {
            
            "roi_position" : tuple(np.subtract(self.get_roi_center(), (75,75))),
            "roi_cell_count" : self.get_roi_cell_count()
        }
        
        self.state_next = {
            "roi_position" : None,
            "roi_cell_count" : None
        }
        
        self.actions = ['up', 'down', 'left', 'right']
    
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
            
        self.update_display()
        
        # update next state
    
        self.state_next["roi_cell_count"] = self.get_roi_cell_count()
        self.state_next["roi_position"] = tuple(np.subtract(self.get_roi_center(), (75,75)))
        
        
        
    def contribution_fn(self):
        return self.state_next["roi_cell_count"] - self.state_current["roi_cell_count"]
        
    def update_current_state(self):
        self.state_current.update(self.state_next)
        
        
        
        
#model1 = Model(r"images\mask\A172_Phase_C7_1_00d00h00m_1_mask.tif")