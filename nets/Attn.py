import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionSurface(nn.Module):
    def __init__(self, tau = 0.0):
        super(AttentionSurface, self).__init__()
        
        self.tau = tau
        
        
        self.multi_head_num_1 = 1
        self.query_length_1 = 3
        self.in_dim_1 = 3
        self.zout_dim_1 = self.query_length_1 * 1
        
        self.multi_head_num_2 = 1
        self.query_length_2 = 32
        self.in_dim_2 = 32
        self.zout_dim_2 = self.query_length_2 * 1
        
        self.attn1 = Self_Attn(self.multi_head_num_1, self.query_length_1, self.in_dim_1, mode = 'none')
        self.Z_mix1 = nn.Conv2d(self.multi_head_num_1*self.query_length_1,self.zout_dim_1 , 1,padding = 0)
        
        layer1 = []
        layer1.append(nn.Conv2d(self.zout_dim_1 + self.in_dim_1, 16, 3, padding = 1))
        layer1.append(nn.BatchNorm2d(16))
        layer1.append(nn.ReLU())
    
        layer1.append(nn.Conv2d(16, 32, 3, padding = 1))
        layer1.append(nn.BatchNorm2d(32))
        layer1.append(nn.ReLU())
        
        layer1.append(nn.MaxPool2d(2,stride = 2))
        
        self.attn2 = Self_Attn(self.multi_head_num_2, self.query_length_2, 32, mode = 'none')
        self.Z_mix2 = nn.Conv2d(self.multi_head_num_2*self.query_length_2, self.zout_dim_2 , 1,padding = 0)
        
        layer2 = []
        layer2.append(nn.Conv2d(self.zout_dim_2 +  self.in_dim_2, 64, 3, padding = 1))
        layer2.append(nn.BatchNorm2d(64))
        layer2.append(nn.ReLU())
        
        layer2.append(nn.ConvTranspose2d(64, 64, 3, stride = 2, padding = 1, output_padding = 1))
        layer2.append(nn.BatchNorm2d(64))
        layer2.append(nn.ReLU())
        
        layer2.append(nn.Conv2d(64, 64, 3, padding = 1))
        layer2.append(nn.BatchNorm2d(64))
        layer2.append(nn.ReLU())
        layer2.append(nn.Conv2d(64, 1, 3, padding = 1))
        layer2.append(nn.BatchNorm2d(1))
        layer2.append(nn.Sigmoid())
        
        #self.l1 = nn.Sequential(*layer1)
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        
        
        
        self.softmax = nn.Softmax(dim = -1)
        
    def change_input(self, x):
        self.batch_size, self.cubic_num,_,_,self.image_size = x.shape
        image_size = self.image_size
        x_out = torch.zeros(self.batch_size, self.cubic_num * 2, 3, image_size, image_size).to(device)
        
        for i_cubic in range(self.cubic_num):
            x_out[:,i_cubic * 2,:,:,:] = x[:,i_cubic,:3]
            x_out[:,i_cubic * 2 + 1,:,:,:] = x[:,i_cubic,3:6]
        
        x_out = x_out.reshape((self.batch_size, self.cubic_num * 2, 3, image_size, image_size))
        
        return x_out
    
    def forward1(self, z_total, x_total):
        
        x_total_output = torch.zeros(self.batch_size, self.cubic_num * 2, 32, self.image_size/2, self.image_size/2).to(device)
        for i_view in range(self.cubic_num * 2):
            z = self.Z_mix1(z_total[:,i_view,:,:,:])
            x = torch.cat((z, x_total[:,i_view,:,:,:]), dim = 1)
            x = self.l1(x)
            x_total_output[:,i_view,:,:,:] = x
            
        return x_total_output
    
    def forward2(self, z_total, x_total):
        
        x_total_output = torch.zeros(self.batch_size, self.cubic_num * 2, 1, self.image_size, self.image_size).to(device)####here####
        for i_view in range(self.cubic_num * 2):
            z = self.Z_mix2(z_total[:,i_view,:,:,:])###here####
            x = torch.cat((z, x_total[:,i_view,:,:,:]), dim = 1)
            x = self.l2(x)###here####
            x_total_output[:,i_view,:,:,:] = x
            
        return x_total_output
    
    def forward_final(self,x_total):
        s_total = x_total.transpose(1,2).transpose(2,3).transpose(3,4)
        s_total_softmax = self.softmax(self.tau * s_total)[:,:,:,:,None,:]
        s = torch.matmul(s_total_softmax, s_total[:,:,:,:,:,None])[:,:,:,:,0,0]
        return s
    
    
    
    def forward(self,x):
        
        x_total = self.change_input(x) #x_total shape: (batch_size, cubic_num * 2, 3, image_size, image_size)
        
        
        z_total, x_total = self.attn1(x_total)#z_total shape: (batch_size, cubic_num * 2, channel, image_size, image_size)
        self.z_total1, self.x_total1 = z_total, x_total
        
        x_total = self.forward1(z_total, x_total)
        
        
        z_total, x_total = self.attn2(x_total)#z_total shape: (batch_size, cubic_num * 2, channel, image_size, image_size)
        self.z_total2, self.x_total2 = z_total, x_total
        
        x_total = self.forward2(z_total, x_total)
        
        s = self.forward_final(x_total)
        
        return s
        


class Self_Attn(nn.Module):
    def __init__(self, multihead_num, query_length, in_dim, mode = 'none'):
        super(Self_Attn, self).__init__()
        
        self.mode = mode
        self.query_length = query_length
        self.multihead_num = multihead_num
        self.in_dim = in_dim
        self.output_dim = self.query_length + self.in_dim
        
        self.Q = nn.Conv2d(in_dim, self.multihead_num * self.query_length, 1, padding = 0)
        self.batch_norm_Q = nn.BatchNorm2d(self.multihead_num * self.query_length)
        self.K = nn.Conv2d(in_dim, self.multihead_num * self.query_length, 1, padding = 0)
        self.batch_norm_K = nn.BatchNorm2d(self.multihead_num * self.query_length)
        self.V = nn.Conv2d(in_dim, self.multihead_num * self.query_length, 1, padding = 0)
        self.batch_norm_V = nn.BatchNorm2d(self.multihead_num * self.query_length)
        
        self.Z_mix = nn.Conv2d(self.multihead_num * self.query_length , self.query_length , 1, padding = 0)
        
        self.softmax  = nn.Softmax(dim=-1)
        
        
    def forward(self, x):
        
        batch_size, view_num, channel,_,image_size = x.shape
        
        if(self.in_dim != channel):
            raise Exception('input dimension of x not equal to self-attention layer')
            
        v_total = torch.zeros(batch_size, view_num, self.query_length, self.multihead_num, image_size, image_size).to(device)
        q_total = torch.zeros(batch_size, view_num, self.query_length, self.multihead_num, image_size, image_size).to(device)
        k_total = torch.zeros(batch_size, view_num, self.query_length, self.multihead_num, image_size, image_size).to(device)
        #x_total = torch.zeros(batch_size, view_num, in_dim, image_size, image_size).to(device)
        output_total = torch.zeros(batch_size, view_num, self.output_dim, image_size, image_size).to(device)
        
        for i_view_num in range(view_num):
            
            data_input = x[:,i_view_num]
            
            v = self.V(data_input)
            v = data_input
            q = self.batch_norm_Q(self.Q(data_input))
            k = self.batch_norm_K(self.K(data_input))
            
            v_total[:,i_view_num,:,:,:,:] = v.reshape((batch_size,self.query_length,self.multihead_num,image_size,image_size))
            q_total[:,i_view_num,:,:,:,:] = q.reshape((batch_size,self.query_length,self.multihead_num,image_size,image_size))
            k_total[:,i_view_num,:,:,:,:] = k.reshape((batch_size,self.query_length,self.multihead_num,image_size,image_size))
            #x_total[:,i_view_num,:,:,:] = data_input
            
        v_total_matmul = v_total.transpose(2,3).transpose(3,4).transpose(4,5).transpose(1,2).transpose(2,3).transpose(3,4)
        q_total_matmul = q_total.transpose(2,3).transpose(3,4).transpose(4,5).transpose(1,2).transpose(2,3).transpose(3,4)
        k_total_matmul = k_total.transpose(2,3).transpose(3,4).transpose(4,5).transpose(1,2).transpose(2,3).transpose(3,4).transpose(4,5)
        qk_matmul = torch.matmul(q_total_matmul, k_total_matmul) / torch.sqrt(torch.tensor(self.query_length + 0.0)).to(device)
        
        qk_softmax = self.softmax(0*qk_matmul)
        #qk0_softmax = softmax_1(0 * qk0_matmul)
        #print('qk0_softmax',qk0_softmax.size())
        
        z_total = torch.matmul(qk_softmax, v_total_matmul)
        
        z_total = z_total.transpose(5,4).transpose(4,3).transpose(3,2)
        z_total = z_total.reshape((batch_size, -1, image_size, image_size, view_num))
        
        if(self.mode == 'output_total'):
            for i_view_num in range(view_num):

                z_out = self.Z_mix(z_total[:,:,:,:,i_view_num])
                output = torch.cat((z_out, x[:,i_view_num,:,:,:]))
                output_total[:,i_view_num, :,:,:] = output

            return output_total
        else:
            z_total = z_total.transpose(4,3).transpose(3,2).transpose(2,1)
            return z_total, x


