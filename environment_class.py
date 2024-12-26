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
        if self.sample_space is None:
            raise ValueError(f"Image not found at the path: {image_path}")
        self.sample_space_8bit = (self.sample_space / np.max(self.sample_space) * 255).astype('uint8')


        # define the region of interest
        
        self.roi_x = (0, 150)
        self.roi_y = (0, 150)
        
        # obtain the region of interets in a new image (variable)
        self.roi = self.sample_space[self.roi_y[0]:self.roi_y[1], self.roi_x[0]:self.roi_x[1]]

        self.roi_8bit = (self.roi / np.max(self.roi) * 255).astype('uint8')

        self.global_label_mask = np.zeros_like(self.sample_space, dtype=np.uint8) 
        self.total_count = 0 

        self.bounding_box_coordinates = []  # List to store (x, y) coordinates of bounding boxes
        _, self.binary_image = cv2.threshold(self.roi_8bit, 127, 255, cv2.THRESH_BINARY)
        
        self.new_roi_x_start = self.roi_x[0]
        self.new_roi_y_start = self.roi_y[0]
      

    def move(self, move_x, move_y):
        # decesions go here, they will move the region of interest
        
        if (self.roi_x[1] + move_x > self.sample_space.shape[1] - 1  or (self.roi_x[0] + move_x) < 0):
            print("x block")
            return
        
        if ((self.roi_y[0] + move_y) < 0  or (self.roi_y[1] + move_y) > self.sample_space.shape[0] - 1):
            print("y block")
            return
        
        
        self.new_roi_x_start = self.roi_x[0] + move_x
        new_roi_x_end = self.roi_x[1] + move_x
        
        self.new_roi_y_start = self.roi_y[0] + move_y
        new_roi_y_end = self.roi_y[1] + move_y
        
        self.roi_x = (self.new_roi_x_start, new_roi_x_end)
        self.roi_y = (self.new_roi_y_start, new_roi_y_end)
        
        
        self.roi = self.sample_space[self.roi_y[0]:self.roi_y[1], self.roi_x[0]:self.roi_x[1]]
        self.roi_8bit = (self.roi / np.max(self.roi) * 255).astype('uint8')
        _, self.binary_image = cv2.threshold(self.roi_8bit, 127, 255, cv2.THRESH_BINARY)
        
    def update_display(self):
        display_image = cv2.cvtColor(self.sample_space_8bit.copy(), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(display_image,
                    (self.roi_x[0], self.roi_y[0]),
                    (self.roi_x[1], self.roi_y[1]),
                    (0, 0, 255), 1)  # Red rectangle
        cv2.imshow("Sample Space", display_image)
        cv2.imshow("ROI", self.roi_8bit)
        # cv2.imshow("Region of Interest", self.roi_8bit)
        # cv2.imshow('sample_space_8bit', self.sample_space_8bit)
        cv2.waitKey(100)
        
    def get_roi_cell_count(self):

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.binary_image, connectivity=8)
        filtered_mask = np.zeros_like(self.binary_image, dtype=np.uint8)
        height, width = self.binary_image.shape

        for self.label in range(1, num_labels): 
            self.x, self.y, self.w, self.h, area = stats[self.label]
            if self.x == 0 or self.y == 0 or (self.x + self.w) >= width or (self.y + self.h) >= height:
                continue
            filtered_mask[labels == self.label] = 255
            self.get_bounding_boxes()

            #cv2.imshow('Filtered Mask', filtered_mask)
            cv2.waitKey(100)


        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filtered_mask)
        #print(f"Number of blobs in roi: {num_labels - 1}")
        #print(f"number of blobs detected so far: {len(self.bounding_box_coordinates)}")
        return(num_labels-1) 
        


    def get_bounding_boxes(self):
        global_x = self.new_roi_x_start + self.x
        global_y = self.new_roi_y_start + self.y

        if (global_x, global_y) not in self.bounding_box_coordinates:
            self.bounding_box_coordinates.append((global_x, global_y))  
            
    def get_roi_center(self):
        
        row = int((self.roi_x[1]-self.roi_x[0])/2)
        col = int((self.roi_y[1]-self.roi_y[0])/2)
        
        return row, col

        
           
#------------------------------------------------------------------------------------------
        
# environment1 = Environment(r"images\mask\A172_Phase_C7_1_00d00h00m_1_mask.tif")

# # move right 50 pixels
# for i in range(2000):
#     environment1.move(0, 1)
#     print("moved")
#     environment1.get_roi_cell_count()
#     environment1.update_display()

# environment1.move(-1, 1)
# print("moved")
# environment1.get_roi_cell_count()
# environment1.update_display()
    
# # move down 50 pixels
# for i in range(200):
#     environment1.move(1, 0)
#     environment1.get_roi_cell_count()
#     print("moved")
#     environment1.update_display()
    
