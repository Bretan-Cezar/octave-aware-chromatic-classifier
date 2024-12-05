import torch
import torch.nn as nn

class NoteClassifier(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.__relu = nn.ReLU(False)

        self.__input = nn.Linear(8000, 4000)
        self.__hidden1 = nn.Linear(4000, 2304)
        self.__hidden2 = nn.Linear(2304, 576)
        self.__output = nn.Linear(576, 72)

        self.__softmax = nn.Softmax(dim=0)

        self.__seq = nn.Sequential(
            self.__input,
            self.__relu,
            self.__hidden1,
            self.__relu,
            self.__hidden2,
            self.__relu,
            self.__output,
            self.__softmax
        )

    def forward(self, x):
        return self.__seq(x)