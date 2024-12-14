from environment_class import Environment
import cv2 
import matplotlib.pyplot as plt
import numpy as np


"""
Ali's code
"""
class roi(Environment):

    def get_roi_cell_count(self):
            

        self.move(0,1)     
        self.move(1,0) 


        # return print(f"Unique cells detected so far: {len(self.detected_cells)}")




        
roi_1 = roi(r"images\mask\A172_Phase_C7_1_00d00h00m_1_mask.tif") 
roi_1.get_roi_cell_count()  

