import torch
import torch.nn as nn
import torch.nn.functional as F




class FineGenerator_3d_res(nn.Module):
    def __init__(self, mode = 'none', residual_type = 'none', activate_type = 'none'):
        super(FineGenerator_3d_res, self).__init__()
        
        self.residual_type = residual_type
        self.mode = mode
        self.activate_type = activate_type
        rate = 0.0
        print('changed rate', rate)

        self.alpha = 3.0
        #self.tau = 0.1 * (0.5 - torch.rand(1).type(torch.cuda.FloatTensor))
        #self.tau = 0
        self.tau = nn.Parameter(torch.zeros(1).type(torch.cuda.FloatTensor))
        #self.tau.requires_grad_(True)
        self.sigmoid_tau = nn.Sigmoid()
        
        self.layer1_0 = nn.Sequential(
            nn.Conv3d(1, 6, 3, padding = 1)
            )
        self.layer1_1 = nn.Sequential(
            nn.BatchNorm3d(6),
            nn.ReLU(),
            nn.Conv3d(6, 12, 3, padding = 1),
            nn.BatchNorm3d(12),
            nn.ReLU(),
            nn.Conv3d(12, 12, 3, padding = 1),
            )
        self.layer1_2 = nn.Sequential(
            nn.BatchNorm3d(12),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(12, 48, 3, padding = 1),
            nn.BatchNorm3d(48),
            nn.ReLU(),
            nn.Conv3d(48, 48, 3, padding = 1)
            )
        self.layer2_0 = nn.Sequential(
            nn.BatchNorm3d(48),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(48, 192, 3, padding = 1),
            nn.BatchNorm3d(192),
            nn.ReLU(),
            nn.Conv3d(192, 192, 3, padding = 1),
            )
        self.layer2_1 = nn.Sequential(
            nn.BatchNorm3d(192),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
            )
        self.up_layer2_1 = nn.Sequential(
            nn.ConvTranspose3d(192, 192, 3, stride = 2, padding = 1, output_padding = 1),
            )
        self.up_layer2_0 = nn.Sequential(
            nn.BatchNorm3d(192),
            nn.ReLU(),
            nn.ConvTranspose3d(192, 48, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(48),
            nn.ReLU(),
            nn.ConvTranspose3d(48, 48, 3, stride = 2, padding = 1, output_padding = 1),
            )
        self.up_layer1_2 = nn.Sequential(
            nn.BatchNorm3d(48),
            nn.ReLU(),
            nn.ConvTranspose3d(48, 12, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(12),
            nn.ReLU(),
            nn.ConvTranspose3d(12, 12, 3, stride = 2, padding = 1, output_padding = 1),
            )   
        self.up_layer1_1 = nn.Sequential(
            nn.BatchNorm3d(12),
            nn.ReLU(),
            nn.ConvTranspose3d(12, 6, 3, stride = 1, padding = 1),
        )
        self.up_layer1_0 = nn.Sequential(
            nn.BatchNorm3d(6),
            nn.ReLU(),
            nn.ConvTranspose3d(6, 1, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
    
    def activate(self,x):
        tau = self.sigmoid_tau(self.tau)
        print('tauFG', tau)
        beta = 1.0 / (1.0/tau-1.0 + 1e-9).pow(self.alpha)
        x = 1.0 / (1.0 + beta * (1.0/(x + 1e-6)-1.0+1e-5).pow(self.alpha)).pow(0.5)
        return x

    def forward(self, input):
        x10 = self.layer1_0(input[:,None,...])
        x11 = self.layer1_1(x10)
        x12 = self.layer1_2(x11)
        x20 = self.layer2_0(x12)

        x = self.layer2_1(x20)
        x = self.up_layer2_1(x)

        if(self.residual_type == 'res'):
            x = x + x20    
        elif(self.residual_type == 'res_detach'):
            x = x + x20.detach()
        x = self.up_layer2_0(x)

        if(self.residual_type == 'res'):
            x = x + x12   
        elif(self.residual_type == 'res_detach'):
            x = x + x12.detach()
        x = self.up_layer1_2(x)

        if(self.residual_type == 'res'):
            x = x + x11  
        elif(self.residual_type == 'res_detach'):
            x = x + x11.detach()
        x = self.up_layer1_1(x)

        if(self.residual_type == 'res'):
            x = x + x10   
        elif(self.residual_type == 'res_detach'):
            x = x + x10.detach()
        elif(self.residual_type == 'res_detach_new'):
            x = x + x10.detach()
        x = self.up_layer1_0(x)

        if(self.activate_type == 'new'):
            x = self.activate(x)
        return x

