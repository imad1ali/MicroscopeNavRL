import cv2 
import matplotlib.pyplot as plt
import os
 

class microscope:
    def __init__(self,path, startX, startY, endX,endY, color, thickness):

        self.startX = startX
        self.startY = startY
        self.endX = endX
        self.endY = endY
        self.color = color
        self.thickness = thickness
        self.path = path
        return

    def environment(self):
        
        # path = r'images/livecell_train_val_images/A172_Phase_A7_1_00d00h00m_1.tif'
        # path2 = r'/Users/jishaansayyed/Library/Mobile Documents/com~apple~CloudDocs/Downloads/PRI00631.JPG'

        # Reading an image in default mode
        image = cv2.imread(os.path.expanduser(self.path))
        # Window name in which image is displayed
        window_name = 'Cells'

        # Start coordinate, here
        # represents the top left corner of rectangle
        start_point = (self.startY, self.startX)

        # Ending coordinate, here
        # represents the bottom right corner of rectangle
        end_point = (self.endY, self.endX)

        color = self.color

        # Line thickness
        thickness = self.thickness

        # Using cv2.rectangle() method
        # thickness
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Displaying the image 
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.show()



properties = microscope(path='images/livecell_train_val_images/A172_Phase_A7_1_00d00h00m_1.tif',
                        startX=0, startY=0, endX=120, endY=120, color=(255, 255, 0), thickness=1)

microscope.environment(properties)