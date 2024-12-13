from environment_class import Environment
import cv2 
import matplotlib.pyplot as plt
import numpy as np


"""
    Ali's code
"""
class roi(Environment):
    
    def get_roi_cell_count(self):

        mask = cv2.imread(r"D:\OneDrive - MMU\Desktop\Knowledge Representation and Reasoning\MicroscopeNavRL\images\mask\A172_Phase_C7_1_00d00h00m_1_mask.tif", cv2.IMREAD_UNCHANGED)
        print("Mask dtype:", mask.dtype)


        #print(f"Trying to load image from: {image_path}")

        #_, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        print("Unique pixel values in mask:", np.unique(mask))
        mask_8bit = (mask / np.max(mask) * 255).astype('uint8')
        cv2.imshow('8-bit Mask', mask_8bit)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        
        
mask = roi(r"D:\OneDrive - MMU\Desktop\Knowledge Representation and Reasoning\MicroscopeNavRL\images\mask\A172_Phase_C7_1_00d00h00m_1_mask.tif")
mask.get_roi_cell_count()


