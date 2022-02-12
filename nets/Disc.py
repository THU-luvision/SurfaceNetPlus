import torch
import torch.nn as nn
import torch.nn.functional as F
'''
class Discriminator(nn.Module):
    def __init__(self, mode = 'relu', image_size = 64):#sigmoid
        super(Discriminator, self).__init__()
        
        self.mode = mode
        
        self.layer_e = nn.Sequential(
            nn.Linear(50,20),
            nn.ReLU(),
            nn.Linear(20,1),
            nn.Sigmoid()
        )
        self.layer_cubic_1 = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer_cubic_2 = nn.Sequential(
            nn.Conv2d(20, 32, 3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer_cubic_3 = nn.Sequential(
            nn.Conv2d(40, 40, 3, padding = 1),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.Conv2d(40, 20, 3, padding = 1),
            nn.BatchNorm2d(20),
            nn.ReLU()
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(image_size*image_size*20/16, 100)
        self.fc2 = nn.Linear(100, 1)
        
        self.layer_transform = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
    
    def forward(self, cubic_total, embedding_total, surface_total):
        
        _,cubic_num,_,_,_ = cubic_total.size()
        
        out_total = 0
        w_total = 0
        self.z_average = 0
        for i_cubic in range(cubic_num):
            
            embedding = embedding_total[:,i_cubic,:]
            cubic = cubic_total[:,i_cubic,:,:,:]
            surface = surface_total
            
            w = self.layer_e(embedding)
            c = self.layer_cubic_1(cubic)
            s = self.layer1(surface)
            c = torch.cat((c,s), dim = 1)

            c = self.layer_cubic_2(c)
            s = self.layer2(s)
            c = torch.cat((c,s), dim = 1)

            c = self.layer_cubic_3(c)

            out = c.reshape(c.size(0), -1)
            self.z_average += out
            if(self.mode == 'sigmoid'):
                out = F.relu(self.fc1(out))
                out = torch.sigmoid(self.fc2(out))
            elif(self.mode == 'relu'):
                out = F.relu(self.fc1(out))
                out = F.relu(self.fc2(out))
            out_total += out * w
            w_total += w
        
        out_total /= (w_total + 1e-9)
        
        return out_total
'''

class Discriminator(nn.Module):
    def __init__(self, mode = 'relu', image_size = 64):#sigmoid
        super(Discriminator, self).__init__()
        
        self.mode = mode
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(8, 20, kernel_size=3, padding=1),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(image_size*image_size*20/64, 100)
        self.fc2 = nn.Linear(100, 1)
        
        self.layer_transform = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
    
    def forward(self, cubic_total, embedding_total, surface_total):
        
        s = self.layer1(surface_total)
        s = self.layer2(s)
        s = self.layer3(s)
        
        out = s.reshape(s.size(0), -1)
        if(self.mode == 'sigmoid'):
            out = F.relu(self.fc1(out))
            out = torch.sigmoid(self.fc2(out))
        elif(self.mode == 'relu'):
            out = F.relu(self.fc1(out))
            out = F.relu(self.fc2(out))
    
        return out
