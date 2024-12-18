from environment_class import Environment

class Model(self):
    
    def __init__(self, env_path):
        
        environment = Environment(env_path)
        
        self.state = {
            "roi_center" : environment.roi_center,
            "cell_count" : None,
            "roi_cell_count" : None
        }
        
        
    pass


env_path = r'C:\Users\imad\Desktop\KRR\MicroscopeNavRL\images\A172_Phase_A7_1_00d00h00m_1.tif'
model1 = Model(env_path)