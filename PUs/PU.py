# Primary user Class 

import torch 
print(torch.__version__ )
class PU: 
    def __init__(self): 
        self.id = 0 
        self.horizon = 20 
        self.TxPattern = torch.zeros(self.horizon, 1)
        self.timer = 0 
    
    def createTxPattern(self):
        self.TxPattern = torch.round(torch.rand((self.horizon,1)))

    def issueWarning(self):
        return 999   # this will return a warining to the SU in order to avoid collision 

    def detectCollision(self, NACK):         
        if NACK == 1 : 
            self.issueWarning() 

if  __name__ == "__main__":
    PU_test = PU()
    PU_test.createTxPattern()
    print(PU_test.TxPattern)
