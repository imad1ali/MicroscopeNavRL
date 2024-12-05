import cv2 
import matplotlib.pyplot as plt
import os
 

class environment:
    """
    Class for creating the task space (environment), where the agent can move around the image
    
    """
    def __init__(self, image_path):
        
        # read in sample space image
        self.sample_space = cv2.imread(os.path.expanduser(image_path))
        
        # define the region of interest
        self.roi_x = (0, 150)
        self.roi_y = (0, 150)
        
        # obtain the region of interets in a new image (variable)
        self.roi = self.sample_space[self.roi_y[0]:self.roi_y[1], self.roi_x[0]:self.roi_x[1]]
        
        # define top left and top right of region of interest
        self.top_left = (self.roi_x[0], self.roi_y[0])
        self.top_right = (self.roi_x[1], self.roi_y[1])

        # draw a rectangle around region of interest 
        self.display_sample_space = self.sample_space.copy()
        cv2.rectangle(self.display_sample_space, self.top_left, self.top_right,
                                             color=(255, 255, 0), thickness=1)
        
        # display region of interest 
        fig, ax = plt.subplots(figsize = (8,8), ncols= 2)
        ax[0].imshow(self.roi)
        ax[0].axis('off')
        
        # display main samplace with region of interest
        ax[1].imshow(self.display_sample_space)
        ax[1].axis('off')
        plt.show()
        
    def move(self):
        # decesions go here, they will move the region of interest
        pass
        

environment1 = environment('images/livecell_train_val_images/A172_Phase_A7_1_00d00h00m_1.tif')
