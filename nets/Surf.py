import torch
import torch.nn as nn
import torch.nn.functional as F

class SurfaceNet_3d_old(nn.Module):
    def __init__(self, use_ad = False):
        super(SurfaceNet_3d_old, self).__init__()
        self.use_ad = use_ad

        self.conv1_1 = nn.Conv3d(6, 32, 3, padding = 1)
        self.batch_norm1_1 = nn.BatchNorm3d(32)
        self.conv1_2 = nn.Conv3d(32, 32, 3, padding = 1)
        self.batch_norm1_2 = nn.BatchNorm3d(32)
        self.conv1_3 = nn.Conv3d(32, 32, 3, padding = 1)
        self.batch_norm1_3 = nn.BatchNorm3d(32)
        
        
        self.pool1 = nn.MaxPool3d(2,stride = 2)
        
        self.side_op1 = nn.ConvTranspose3d(32, 16, 3, stride = 1, padding = 1)
        self.batch_norm1_s = nn.BatchNorm3d(16)
        
        
        self.conv2_1 = nn.Conv3d(32, 80, 3, padding = 1)
        self.batch_norm2_1 = nn.BatchNorm3d(80)
        self.conv2_2 = nn.Conv3d(80, 80, 3, padding = 1)
        self.batch_norm2_2 = nn.BatchNorm3d(80)
        self.conv2_3 = nn.Conv3d(80, 80, 3, padding = 1)
        self.batch_norm2_3 = nn.BatchNorm3d(80)
        
        self.pool2 = nn.MaxPool3d(2,stride = 2)
        
        self.side_op2 = nn.ConvTranspose3d(80, 16, 3, stride = 2, padding = 1, output_padding = 1)
        self.batch_norm2_s = nn.BatchNorm3d(16)
        
        self.conv3_1 = nn.Conv3d(80, 160, 3, padding = 1)
        self.batch_norm3_1 = nn.BatchNorm3d(160)
        self.conv3_2 = nn.Conv3d(160, 160, 3, padding = 1)
        self.batch_norm3_2 = nn.BatchNorm3d(160)
        self.conv3_3 = nn.Conv3d(160, 160, 3, padding = 1)
        self.batch_norm3_3 = nn.BatchNorm3d(160)
        
        self.side_op3 = nn.ConvTranspose3d(160, 16, 3, stride = 4, padding = 0, output_padding = 1)
        self.batch_norm3_s = nn.BatchNorm3d(16)
        

        self.pool3 = nn.MaxPool3d(2,stride = 2)

        #self.conv4_1 = nn.Conv3d(160, 300, 3, padding = 2,dilation = 2)
        self.conv4_1 = nn.Conv3d(160, 300, 3, padding = 1)
        self.batch_norm4_1 = nn.BatchNorm3d(300)
        self.conv4_2 = nn.Conv3d(300, 300, 3, padding = 1)
        self.batch_norm4_2 = nn.BatchNorm3d(300)
        #self.conv4_3 = nn.Conv3d(300, 300, 3, padding = 2,dilation = 2)
        self.conv4_3 = nn.Conv3d(300, 300, 3, padding = 1)
        self.batch_norm4_3 = nn.BatchNorm3d(300)
        
        self.side_op4 = nn.ConvTranspose3d(300, 16, 4, stride = 8, padding = 0, output_padding = 4)
        self.batch_norm4_s = nn.BatchNorm3d(16)
        
        
        self.conv5_1 = nn.Conv3d(64, 100, 3, padding = 1)
        self.batch_norm5_1 = nn.BatchNorm3d(100)
        self.conv5_2 = nn.Conv3d(100, 100, 3, padding = 1)
        self.batch_norm5_2 = nn.BatchNorm3d(100)
        
        self.conv5_3 = nn.Conv3d(100, 1, 3, padding = 1)
        self.batch_norm5_3 = nn.BatchNorm3d(1)
        
        if(self.use_ad):
            self.conv6_1 = nn.Conv3d(65, 50, 3, padding = 1)
            self.batch_norm6_1 = nn.BatchNorm3d(50)
            self.conv6_2 = nn.Conv3d(50, 50, 3, padding = 1)
            self.batch_norm6_2 = nn.BatchNorm3d(50)
        
            self.conv6_3 = nn.Conv3d(50, 1, 3, padding = 1)
            self.batch_norm6_3 = nn.BatchNorm3d(1)    
        
    
    def surface(self, x):
        x = F.relu(self.batch_norm1_1(self.conv1_1(x)))
        x = F.relu(self.batch_norm1_2(self.conv1_2(x)))
        x = F.relu(self.batch_norm1_3(self.conv1_3(x)))
        
        s1 = F.relu(self.batch_norm1_s(self.side_op1(x)))
        x = F.relu(self.pool1(x))
        
        
        x = F.relu(self.batch_norm2_1(self.conv2_1(x)))
        x = F.relu(self.batch_norm2_2(self.conv2_2(x)))
        x = F.relu(self.batch_norm2_3(self.conv2_3(x)))
        
        s2 = F.relu(self.batch_norm2_s(self.side_op2(x)))
        x = F.relu(self.pool2(x))
        
        
        x = F.relu(self.batch_norm3_1(self.conv3_1(x)))
        x = F.relu(self.batch_norm3_2(self.conv3_2(x)))
        x = F.relu(self.batch_norm3_3(self.conv3_3(x)))
        
        s3 = F.relu(self.batch_norm3_s(self.side_op3(x)))
        x = F.relu(self.pool3(x))
        

        x = F.relu(self.batch_norm4_1(self.conv4_1(x)))
        x = F.relu(self.batch_norm4_2(self.conv4_2(x)))
        x = F.relu(self.batch_norm4_3(self.conv4_3(x)))
        
        s4 = F.relu(self.batch_norm4_s(self.side_op4(x)))
        
        
        #s = torch.cat((s1,s2,s3),1)
        s = torch.cat((s1,s2,s3,s4),1)
        s_64 = s
        
        s = F.relu(self.batch_norm5_1(self.conv5_1(s)))
        s = F.relu(self.batch_norm5_2(self.conv5_2(s)))
        s = self.conv5_3(s)
        s_ad_res = s
        s_ad = torch.cat((s_64, s_ad_res),1)

        s = torch.sigmoid(self.batch_norm5_3(s))

        if(self.use_ad):
            s_ad = F.relu(self.batch_norm6_1(self.conv6_1(s_ad)))
            s_ad = F.relu(self.batch_norm6_2(self.conv6_2(s_ad)))
            s_ad = self.conv6_3(s_ad)
            s_ad = s_ad_res.detach() + s_ad
            s_ad = torch.sigmoid(self.batch_norm6_3(s_ad))

            return (s, s_ad)
        else:
            return s
    
    def forward(self,x):
        
        if(self.use_ad):
            s,s_ad = self.surface(x)
            return (s,s_ad)
        else:
            s = self.surface(x)
            return s


def count_parameters(surfaceNet):
    return sum(p.numel() for p in surfaceNet.parameters() if p.requires_grad)

#count_parameters(surfaceNet)


class EmbeddingNet_3d_new_big(nn.Module):
    def __init__(self):
        super(EmbeddingNet_3d_new_big, self).__init__()

        #self.fc1 = nn.Linear(50,50)
        self.fc2 = nn.Linear(410,200)
        self.fc3 = nn.Linear(200,50)
        self.fc4 = nn.Linear(64,1)
        #self.fc5 = nn.Linear(10,10)
        #self.fc6 = nn.Linear(10,1)
        
    def arti_embedding(self, e):
        
        p1 = e[:,98:206]
        p2 = e[:,302:410]

        e_a1_1 = (torch.mean(p1, dim =1)).cpu()[:,None].type(torch.cuda.FloatTensor) * 0.1
        e_a1_2 = (torch.mean(p2, dim =1)).cpu()[:,None].type(torch.cuda.FloatTensor) * 0.1

        e_a2_1 = (torch.var(p1, dim =1)).cpu()[:,None].type(torch.cuda.FloatTensor) * 5
        e_a2_2 = (torch.var(p2, dim =1)).cpu()[:,None].type(torch.cuda.FloatTensor) * 5

        e_a5_1 = torch.mean((p1 - p2)**2, dim = 1).cpu()[:,None].type(torch.cuda.FloatTensor)


        h1_r = e[:,2:34]
        h2_r = e[:,206:238]
        h1_g = e[:,34:66]
        h2_g = e[:,238:270]
        h1_b = e[:,66:98]
        h2_b = e[:,270:302]
    
        e_a6_1 = (torch.abs(h1_r-h2_r).mean(dim = 1)).cpu()[:,None].type(torch.cuda.FloatTensor) * 20
        e_a6_2 = (torch.abs(h1_g-h2_g).mean(dim = 1)).cpu()[:,None].type(torch.cuda.FloatTensor) * 20
        e_a6_3 = (torch.abs(h1_b-h2_b).mean(dim = 1)).cpu()[:,None].type(torch.cuda.FloatTensor) * 20

        e_a6_4 = ((torch.max(h1_r, dim = 1)[0])).cpu()[:,None].type(torch.cuda.FloatTensor) * 5
        e_a6_5 = ((torch.max(h2_r, dim = 1)[0])).cpu()[:,None].type(torch.cuda.FloatTensor) * 5
        e_a6_6 = ((torch.max(h1_g, dim = 1)[0])).cpu()[:,None].type(torch.cuda.FloatTensor) * 5
        e_a6_7 = ((torch.max(h2_g, dim = 1)[0])).cpu()[:,None].type(torch.cuda.FloatTensor) * 5
        e_a6_8 = ((torch.max(h1_b, dim = 1)[0])).cpu()[:,None].type(torch.cuda.FloatTensor) * 5
        e_a6_9 = ((torch.max(h2_b, dim = 1)[0])).cpu()[:,None].type(torch.cuda.FloatTensor) * 5
        
        e_out = torch.cat((e_a1_1,e_a1_2,e_a2_1,e_a2_2,e_a5_1,e_a6_1,e_a6_2,e_a6_3,e_a6_4,e_a6_5,e_a6_6,e_a6_7,e_a6_8,e_a6_9), 1)
        return e_out
        #return 0
    def artifact(self, e):
        #print('changed eNet 113')

        p1 = e[:,98:206]
        p2 = e[:,302:410]

        e_a1 = 3000 * ((torch.mean(p1, dim =1) < 3.0/256.0) + (torch.mean(p2, dim = 1) < 3.0/256.0) + 
                (torch.mean(p1, dim =1) > 253.0/256.0) + (torch.mean(p2, dim =1) > 253.0/256.0)).cpu()[:,None].type(torch.cuda.FloatTensor)

        #e_a2 = ((torch.var(e[:,2:194], dim =1) < 0.01) + (torch.var(e[:,194:], dim = 1) < 0.01)).cpu()[:,None].type(torch.cuda.FloatTensor)
        
        #e_a3 = (torch.abs(torch.mean(e[:,2:194], dim =1) - torch.mean(e[:,194:], dim = 1)) > 0.8).cpu()[:,None].type(torch.cuda.FloatTensor)

        e_a4 = e[:,0].cpu()[:,None].type(torch.cuda.FloatTensor)/10.0
        e_a4_1 = 0.3 * (e[:,0] < 10 * 3.14159/180).cpu()[:,None].type(torch.cuda.FloatTensor)
        e_a4_2 = 0.3 * ((e[:,0] < 20 * 3.14159/180) * (torch.mean((p1 - p2)**2, dim = 1) < 0.002)).cpu()[:,None].type(torch.cuda.FloatTensor)

        e_a5 = 0.3 * ((torch.mean((p1 - p2)**2, dim = 1) > 0.2) + (torch.mean((p1 - p2)**2, dim = 1) < 0.002)).cpu()[:,None].type(torch.cuda.FloatTensor)
        #e_a5_1 = torch.mean((p1 - p2)**2, dim = 1).cpu()[:,None].type(torch.cuda.FloatTensor)


        h1_r = e[:,2:34]
        h2_r = e[:,206:238]
        h1_g = e[:,34:66]
        h2_g = e[:,238:270]
        h1_b = e[:,66:98]
        h2_b = e[:,270:302]
    
        e_a6 = 50 * (torch.abs(h1_r-h2_r).mean(dim = 1) + torch.abs(h1_g-h2_g).mean(dim = 1) + torch.abs(h1_b-h2_b).mean(dim = 1)).cpu()[:,None].type(torch.cuda.FloatTensor)
        e_a6_1 = 0.5 * ((torch.abs(h1_r-h2_r).mean(dim = 1)>0.004) + (torch.abs(h1_g-h2_g).mean(dim = 1)>0.004) + (torch.abs(h1_b-h2_b).mean(dim = 1)>0.004)).cpu()[:,None].type(torch.cuda.FloatTensor)
        e_a6_2 = 0.5 * ((torch.max(h1_r, dim = 1)[0]>0.04)+(torch.max(h2_r, dim = 1)[0]>0.04)+(torch.max(h1_g, dim = 1)[0]>0.04)+(torch.max(h2_g, dim = 1)[0]>0.04)+(torch.max(h1_b, dim = 1)[0]>0.04)+(torch.max(h2_b, dim = 1)[0]>0.04)).cpu()[:,None].type(torch.cuda.FloatTensor)
        e_a6_3 = 20 * ((torch.max(h1_r, dim = 1)[0]>0.08)+(torch.max(h2_r, dim = 1)[0]>0.08)+(torch.max(h1_g, dim = 1)[0]>0.08)+(torch.max(h2_g, dim = 1)[0]>0.08)+(torch.max(h1_b, dim = 1)[0]>0.08)+(torch.max(h2_b, dim = 1)[0]>0.08)).cpu()[:,None].type(torch.cuda.FloatTensor)
        
        return e_a1  + e_a4 + e_a4_1 + e_a4_2 + e_a5 + e_a6 + e_a6_1 + e_a6_2 + e_a6_3
        #return 0

    def artifact_inference(self, e):
        #print('changed eNet 101')

        p1 = e[:,98:206]
        p2 = e[:,302:410]

        e_a1 =  30 * ((torch.mean(p1, dim =1) < 3.0/256.0) + (torch.mean(p2, dim = 1) < 3.0/256.0) + 
                 (torch.mean(p1, dim =1) > 253.0/256.0) + (torch.mean(p2, dim =1) > 253.0/256.0)).cpu()[:,None].type(torch.cuda.FloatTensor)
        #e_a1 =  3000 * ((torch.mean(p1, dim =1) < 3.0/256.0) + (torch.mean(p2, dim = 1) < 3.0/256.0)).cpu()[:,None].type(torch.cuda.FloatTensor)

        e_a2 = ((torch.var(e[:,2:194], dim =1) < 0.01) + (torch.var(e[:,194:], dim = 1) < 0.01)).cpu()[:,None].type(torch.cuda.FloatTensor)
        
        #e_a3 = (torch.abs(torch.mean(e[:,2:194], dim =1) - torch.mean(e[:,194:], dim = 1)) > 0.8).cpu()[:,None].type(torch.cuda.FloatTensor)

        e_a4 =  10 * e[:,0].cpu()[:,None].type(torch.cuda.FloatTensor)/10.0
        e_a4_1 = 0.1 * (e[:,0] < 20 * 3.14159/180).cpu()[:,None].type(torch.cuda.FloatTensor)
        e_a4_2 = 0.1 * ((e[:,0] < 30 * 3.14159/180) * (torch.mean((p1 - p2)**2, dim = 1) < 0.001)).cpu()[:,None].type(torch.cuda.FloatTensor)
        e_a4_3 = 10 * ((e[:,0] < 90 * 3.14159/180) * (torch.mean((p1 - p2)**2, dim = 1) < 0.001)).cpu()[:,None].type(torch.cuda.FloatTensor)

        e_a5 = 0.3 * ((torch.mean((p1 - p2)**2, dim = 1) > 0.15) + (torch.mean((p1 - p2)**2, dim = 1) < 0.0003)).cpu()[:,None].type(torch.cuda.FloatTensor)
        #e_a5_1 = -3 * torch.mean((p1 - p2)**2, dim = 1).cpu()[:,None].type(torch.cuda.FloatTensor)


        h1_r = e[:,2:34]
        h2_r = e[:,206:238]
        h1_g = e[:,34:66]
        h2_g = e[:,238:270]
        h1_b = e[:,66:98]
        h2_b = e[:,270:302]
    
        e_a6 = 50 * (torch.abs(h1_r-h2_r).mean(dim = 1) + torch.abs(h1_g-h2_g).mean(dim = 1) + torch.abs(h1_b-h2_b).mean(dim = 1)).cpu()[:,None].type(torch.cuda.FloatTensor)
        e_a6_1 = 0.1 * ((torch.abs(h1_r-h2_r).mean(dim = 1)>0.003) + (torch.abs(h1_g-h2_g).mean(dim = 1)>0.003) + (torch.abs(h1_b-h2_b).mean(dim = 1)>0.003)).cpu()[:,None].type(torch.cuda.FloatTensor)
        e_a6_2 = 0.1 * ((torch.max(h1_r, dim = 1)[0]>0.03)+(torch.max(h2_r, dim = 1)[0]>0.03)+(torch.max(h1_g, dim = 1)[0]>0.03)+(torch.max(h2_g, dim = 1)[0]>0.03)+(torch.max(h1_b, dim = 1)[0]>0.03)+(torch.max(h2_b, dim = 1)[0]>0.03)).cpu()[:,None].type(torch.cuda.FloatTensor)
        e_a6_3 = 3.0 * ((torch.max(h1_r, dim = 1)[0]>0.06)+(torch.max(h2_r, dim = 1)[0]>0.06)+(torch.max(h1_g, dim = 1)[0]>0.06)+(torch.max(h2_g, dim = 1)[0]>0.06)+(torch.max(h1_b, dim = 1)[0]>0.06)+(torch.max(h2_b, dim = 1)[0]>0.06)).cpu()[:,None].type(torch.cuda.FloatTensor)
        e_a6_4 = 30 * (torch.var(h1_r, dim = 1)+torch.var(h1_g, dim = 1)+torch.var(h1_b, dim = 1)+torch.var(h2_r, dim = 1)+torch.var(h2_g, dim = 1)+torch.var(h2_b, dim = 1)).cpu()[:,None].type(torch.cuda.FloatTensor)
        
        #return e_a1 + e_a2 + e_a4 + e_a4_1+ e_a4_2 + e_a4_3 + e_a5 + e_a6 + e_a6_1 + e_a6_2 + e_a6_3 + e_a6_4
        #return 0
        return e_a1 + e_a2 + e_a5 + e_a6 + e_a6_1 + e_a6_2 + e_a6_3 + e_a6_4
        

    def embedding(self,e):
        
        #print('use arti')
        e_arti = self.arti_embedding(e)

        e = F.relu(self.fc2(e))
        e = F.relu(self.fc3(e))
        
        e = torch.cat((e,e_arti),1)
        w = self.fc4(e)
        #print('w',w)
        return w
    
    def forward(self,e):
        
        #w = torch.exp(self.embedding(e))
        #print('self.artifact(e)',self.artifact(e))
        #print('self.artifact_inference(e)', self.artifact_inference(e))
        #print('self.embedding(e)', self.embedding(e))
        w = torch.exp( -0.5 * self.artifact_inference(e)  + 0.5 * self.embedding(e))
        #w = torch.exp( - 0.5 * self.artifact(e) + self.embedding(e))
        #w = torch.exp( - 0.5 * self.artifact_inference(e))

        return w