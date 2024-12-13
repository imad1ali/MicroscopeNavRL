import cv2 
import matplotlib.pyplot as plt
import os
import numpy as np
 

class Environment:
    """
    Class for creating the task space (environment), where the agent can move around the image
    """
    def __init__(self, image_path):
        
        # read in sample space image
        self.sample_space = cv2.imread(os.path.expanduser(image_path), cv2.IMREAD_UNCHANGED)
        self.sample_space_8bit = (self.sample_space / np.max(self.sample_space) * 255).astype('uint8')

        if self.sample_space is None:
            raise ValueError(f"Image not found at the path: {image_path}")
        # define the region of interest
        self.roi_x = (0, 150)
        self.roi_y = (0, 150)
        
        # obtain the region of interets in a new image (variable)
        self.roi = self.sample_space[self.roi_y[0]:self.roi_y[1], self.roi_x[0]:self.roi_x[1]]

    def move(self, move_x, move_y):
        # decesions go here, they will move the region of interest
        new_roi_x_start = self.roi_x[0] + move_x
        new_roi_x_end = self.roi_x[1] + move_x
        
        new_roi_y_start = self.roi_y[0] + move_y
        new_roi_y_end = self.roi_y[1] + move_y
        
        self.roi_x = (new_roi_x_start, new_roi_x_end)
        self.roi_y = (new_roi_y_start, new_roi_y_end)

      
        self.roi = self.sample_space[self.roi_y[0]:self.roi_y[1], self.roi_x[0]:self.roi_x[1]]
        self.roi_8bit = (self.roi / np.max(self.roi) * 255).astype('uint8')
        
    def update_display(self):
        display_image = self.sample_space.copy()
        cv2.rectangle(display_image,
                    (self.roi_x[0], self.roi_y[0]),
                    (self.roi_x[1], self.roi_y[1]),
                    (0, 0, 255), 1)  # Red rectangle
        #cv2.imshow("Sample Space", display_image)
        cv2.imshow("Region of Interest", self.roi_8bit)
        cv2.imshow('sample_space_8bit', self.sample_space_8bit)
        cv2.waitKey(100)  # Adds a small delay for smooth visualization
        
           
#------------------------------------------------------------------------------------------
        
environment1 = Environment(r"images\mask\A172_Phase_C7_1_00d00h00m_1_mask.tif")

# move right 50 pixels
for i in range(100):
    environment1.move(1,0)
    print("moved")
    environment1.update_display()
    
# move down 50 pixels
for i in range(100):
    environment1.move(0,1)
    print("moved")
    environment1.update_display()
    