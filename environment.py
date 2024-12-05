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

        # draw a rectangle around region of interest 
        
        

        self.rectangle = plt.Rectangle([self.roi_x[0], self.roi_y[0]],
                                       self.roi_x[1]-self.roi_x[0],
                                       self.roi_y[1]-self.roi_y[0],
                                       linewidth=1, edgecolor='red', facecolor='none')
    
        # display region of interest 
        self.fig, self.ax = plt.subplots(figsize = (8,8), ncols= 2)
        self.ax[0].imshow(self.roi)
        self.ax[0].axis('off')
        
        # display main samplace with region of interest
        self.ax[1].imshow(self.sample_space)
        self.ax[1].add_patch(self.rectangle)
        self.ax[1].axis('off')
        plt.draw()
        plt.pause(0.1)
        
        
    def move(self, move_x, move_y):
        # decesions go here, they will move the region of interest
        new_roi_x_start = self.roi_x[0] + move_x
        new_roi_x_end = self.roi_x[1] + move_x
        
        new_roi_y_start = self.roi_y[0] + move_y
        new_roi_y_end = self.roi_y[1] + move_y
        
        self.roi_x = (new_roi_x_start, new_roi_x_end)
        self.roi_y = (new_roi_y_start, new_roi_y_end)

      
        self.roi = self.sample_space[self.roi_y[0]:self.roi_y[1], self.roi_x[0]:self.roi_x[1]]
        
    def update_display(self):
        self.rectangle.remove()
        self.rectangle = plt.Rectangle([self.roi_x[0], self.roi_y[0]],
                                       self.roi_x[1]-self.roi_x[0],
                                       self.roi_y[1]-self.roi_y[0],
                                       linewidth=1, edgecolor='red', facecolor='none')
    
        # display region of interest 
        self.ax[0].imshow(self.roi)
        
        # display main samplace with region of interest
        self.ax[1].imshow(self.sample_space)
        self.ax[1].add_patch(self.rectangle)
        plt.draw()
        plt.pause(0.0001)
        
        
    
        
#------------------------------------------------------------------------------------------
        
environment1 = environment('images/livecell_train_val_images/A172_Phase_A7_1_00d00h00m_1.tif')

# move right 50 pixels
for i in range(50):
    environment1.move(1,0)
    environment1.update_display()
    
# move down 50 pixels
for i in range(50):
    environment1.move(0,1)
    environment1.update_display()
    