import numpy as np
import itertools
import torch
import os
import random
import pdb
import scipy.io
import math
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

import time
import torch.optim as optim

import sys
sys.path.append("./nets")
sys.path.append("./tools")

import imp

import Prepair
import Parameter
import Surf
import Attn
import F_G
import Disc

from Surf import SurfaceNet, SurfaceNet_3d, EmbeddingNet, EmbeddingNet_3d, SurfaceNet_3d_old,EmbeddingNet_3d_pixel
from Attn import AttentionSurface, Self_Attn
from F_G import FineGenerator, FineGenerator_3d_res
from Disc import Discriminator


from Parameter import Params
from Prepair import Dataset, Recorder

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import os
import cPickle as pickle
import rayPooling
import sys
#import camera
from plyfile import PlyData, PlyElement

import denoising


def threholdSurf(data, threshold = 0.7):
    
    threshold_surf = (data >= threshold)
    threshold_surf = threshold_surf * 1.0
    return threshold_surf

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

def threholdSurf(data, threshold = 0.7):
    
    threshold_surf = (data >= threshold)
    threshold_surf = threshold_surf * 1.0
    return threshold_surf

class TrainIter(Params):
    def __init__(self, recorder = None, d2s = None):
        super(TrainIter, self).__init__()
        self.recorder = recorder
        self.init_records()
        self.init_data()
        plt.clf()
        
        self.writer = SummaryWriter('./experiment/log')
        self.d2s = d2s

    
    def init_data(self):
        self.cubic_num = 1
        
        self.model_path = 'experiment/network/'
        
        self.criterion = nn.BCELoss()
        
        
    
    def init_records(self):
        self.validation_loss_list = []
        
        self.reward_list = []
        self.accuracy_list = []
        self.completeness_list = []
        self.f_distance_list = []
        
        self.accuracy_list_test= []
        self.completeness_list_test= []
        self.f_distance_list_test= []
        
        self.accuracy_list_test2= []
        self.completeness_list_test2= []
        self.f_distance_list_test2= []
   
    def matrix_reward(self,
                       groundturth, 
                       output,
                       threshold = 0.7,
                       alpha = 0.5):
        
        output = output.detach()
        ground_truth = groundturth.detach()

        thre_output = (output > threshold).to(device)
        
        #batch_size,_,dimention,_ = output.size()
        #rand_tensor = torch.rand(batch_size,1,dimention,dimention).to(device)
        #rand_choice = (output > rand_tensor).type(torch.cuda.FloatTensor)
        #thre_output = rand_choice
        
        and_num = torch.sum(thre_output * ground_truth, dim = [1,2,3])
        thre_num = torch.sum(thre_output, dim = [1,2,3])
        ground_truth_num = ground_truth.sum(dim = [1,2,3]).to(device)
        
        white_num = ground_truth_num - and_num
        black_num = thre_num - and_num
        
        accuracy = (and_num + 0.0) / ( + 1)
        completeness = (and_num + 0.0) / (thre_output.sum(dim = [1,2,3]) + 1)
        
        white_score = 1 / (ground_truth_num + 1)
        black_score = 1 / (thre_num + 1)
        
        white_mask = ground_truth - thre_output * ground_truth
        black_mask = thre_output - thre_output * ground_truth
        
        reward_matrix = - alpha * white_score[:,None,None,None] * white_mask - (1 - alpha) *  black_score[:,None,None,None] * black_mask
        reward_matrix = - alpha *  white_mask - (1 - alpha) *  black_mask
        return reward_matrix
        
    
    def surface_reward(self, 
                       groundturth, 
                       output,
                       threshold = 0.7,
                       alpha = 0.5):
        
        self.output_test = output
        self.data_test = groundturth
        output = output.detach()
        ground_truth = groundturth.detach()

        thre_output = (output > threshold).to(device)
        and_num = torch.sum(thre_output * ground_truth, dim = [1,2,3])
        
        
        #print('ground_truth',ground_truth)
        #print('output',output)
        #print(and_num, thre_output.sum(dim = [1,2,3]) , ground_truth.sum(dim = [1,2,3]))
        #print()
        
        accuracy = (and_num + 0.0) / (ground_truth.sum(dim = [1,2,3]).to(device) + 1)
        completeness = (and_num + 0.0) / (thre_output.sum(dim = [1,2,3]) + 1)

        reward = alpha * accuracy + (1-alpha) * completeness
        
        return reward
        #return 1.0
        
    def batch_chamfer_loss(self, output, ground_truth):
        
        #output = output
        ground_truth_binary = ground_truth
        thre_output_binary =  output
        
        if(self._datasetName == 'MVS2d'): 
            batch_size,_,_,_ = ground_truth_binary.size()
        else:
            batch_size,_,_,_,_ = ground_truth_binary.size()
            
        mean_accuracy = 0
        mean_completeness = 0
        mean_f_distance = 0
        for i_batch in range(batch_size):
            index_truth = (ground_truth_binary[i_batch,0] == 1).nonzero().type(torch.cuda.FloatTensor)
            index_output = (thre_output_binary[i_batch,0] == 1).nonzero().type(torch.cuda.FloatTensor)
            #print(index_truth)
            #print(index_output)
            if(index_truth.size() == torch.Size([0]))and(index_output.size() == torch.Size([0])):
                mean_accuracy += torch.tensor(0.05).type(torch.cuda.FloatTensor)
                mean_completeness += torch.tensor(0.05).type(torch.cuda.FloatTensor)
                mean_f_distance += torch.tensor(0.1).type(torch.cuda.FloatTensor)
            elif(index_truth.size() == torch.Size([0]))or(index_output.size() == torch.Size([0])):
                mean_accuracy += torch.tensor(2.5).type(torch.cuda.FloatTensor)
                mean_completeness += torch.tensor(2.5).type(torch.cuda.FloatTensor)
                mean_f_distance += torch.tensor(5.0).type(torch.cuda.FloatTensor)
            else:
                accuracy_reward, completeness_reward = self.chamfer_distance(index_truth[None,...], index_output[None,...], return_list = False)
                mean_accuracy += accuracy_reward
                mean_completeness += completeness_reward
                mean_f_distance += (accuracy_reward + completeness_reward)
                #print('>>>>',accuracy_reward, completeness_reward)
        mean_accuracy /= batch_size
        mean_completeness /= batch_size
        mean_f_distance /= batch_size
        
      
            
        return mean_accuracy.cpu().item(), mean_completeness.cpu().item(), mean_f_distance.cpu().item()
                
    def chamfer_distance(self, p1, p2, return_list = True):

        '''
        Calculate Chamfer Distance between two point sets
        :param p1: size[B, N, D]
        :param p2: size[B, M, D]
        :param debug: whether need to output debug info
        :return: sum of all batches of Chamfer Distance of two point sets
        '''

        assert p1.size(0) == p2.size(0) and p1.size(2) == p2.size(2)


        p1 = p1.unsqueeze(1)
        p2 = p2.unsqueeze(1)

        p1 = p1.repeat(1, p2.size(2), 1, 1)

        p1 = p1.transpose(1, 2)

        p2 = p2.repeat(1, p1.size(1), 1, 1)

        dist = torch.add(p1, torch.neg(p2))


        dist = torch.norm(dist, 2, dim=3)


        dist2 = torch.min(dist, dim=2)[0]
        dist1 = torch.min(dist, dim = 1)[0]

        accuracy_score = torch.mean(dist1)
        completeness_score = torch.mean(dist2)

        if(return_list):
            return (dist1,dist2)
        else:
            return accuracy_score, completeness_score
       
    def blur_truth(self, data_surface, output_detach, interpolation = 0.1, add_mass = 0.1, minus_mass = 0.1):
        
        #masked_output_new = data_surface * add_mass - (1 - data_surface) * minus_mass + output_detach
        #masked_output_new = torch.clamp(masked_output_new, max = 1.0, min = 0.0)
        
        masked_output_new = data_surface * interpolation + output_detach * (1 - interpolation)
        
        return masked_output_new.detach() 

    
    def train_gan(self,
                num_epochs = 100,
                  model_list = [0],
                  batch_size = 16,
                  lr_gen = 1e-4,
                  lr_dis = 1e-5,
                  weight_cliping_limit = 0.05,
                  add_mass = 0.1, 
                  minus_mass = 0.1, 
                  interpolation = 0.1,
                  use_dynamic_alpha = False,
                  decay_rate = 0.95,
                  alpha_decay = 1.0,
                  reward_epoch = 10,
                  use_augment = True, 
                  use_augment_fg = True,
                  use_teacher = True,
                  use_fg = True,
                  use_fg_teacher = True,
                  fg_detach = False,
                  use_fg_transfer = True,
                  fg_tau = 1.0,
                  alpha_decay_fg = 0.1,
                  god_w = 1e-3,
                  teacher_type = 'none',
                  teacher_type_fg = 'none',
                  d_step = 1,
                  g_step = 3,
                  validation_iter = 5,
                  save_iter = 5,
                  shuffle = False,
                  train_model = 'none',
                  root_path = 'experiment/network/',
                  loss_type = 'gan',
                  training_key = '030401',
                  show = False
                  ):
        model_path = self.model_path
        
        loss_show = 0
        loss_mse = nn.MSELoss()
        
        optimizer_s = optim.RMSprop(surfaceNet.parameters(), lr = lr_gen)
        optimizer_e = optim.RMSprop(eNet.parameters(), lr = lr_gen)
        optimizer_d = optim.RMSprop(disNet.parameters(), lr = lr_dis)
        optimizer_fg = optim.RMSprop(fg.parameters(), lr = lr_gen)
        optimizer_fg_transfer = optim.RMSprop(fg_transfer.parameters(), lr = lr_dis)
        
        i_append = 0
        accuracy_ave = 0
        completeness_ave = 0
        f_distance_ave = 0

        for epoch in range(num_epochs):
            
            alpha_decay /= decay_rate
            alpha_decay_fg /= decay_rate
            
            if(epoch%save_iter == 0):
                root_path = root + '%s/'%(self._datasetName)
                directory = root_path + '%s/%s'%(loss_type, training_key)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                model_name = '%s/%s/surfaceNet_epoch%r'%(loss_type, training_key, epoch)
                torch.save(surfaceNet.cpu().state_dict(), root_path + model_name)
                surfaceNet.to(device)
                model_name = '%s/%s/eNet_epoch%r'%(loss_type, training_key, epoch)
                torch.save(eNet.cpu().state_dict(), root_path + model_name)
                eNet.to(device)
                model_name = '%s/%s/disNet_epoch%r'%(loss_type, training_key, epoch)
                torch.save(disNet.cpu().state_dict(), root_path + model_name)
                disNet.to(device)
                if(use_fg):
                    model_name = '%s/%s/fgNet_epoch%r'%(loss_type, training_key, epoch)
                    torch.save(fg.cpu().state_dict(), root_path + model_name)
                    fg.to(device)
            
            if(epoch % validation_iter == 0):
                self.validationTest(surfaceNet, eNet, model_num = self._test_modelList[0], use_fg = use_fg,fg_tau = fg_tau,  draw = True, train_model = train_model)
                self.validationTest(surfaceNet, eNet, model_num = self._test_modelList[1], use_fg = use_fg,fg_tau = fg_tau,  draw = True, train_model = train_model)
                self.validation_loss_list.append(self.loss_validation)
                #print('USE fg nono')
                #self.validationTest(surfaceNet, eNet, model_num = 344, use_fg = False, draw = True, train_model = train_model)
                #self.validationTest(surfaceNet, eNet, model_num = 325, use_fg = False, draw = True, train_model = train_model)
                
               
            for model_num in model_list:
                
                dataset = Dataset(model_num)
                batch_size = 16

                dataloader = DataLoader(dataset, batch_size= batch_size,
                                        shuffle=shuffle, num_workers=12)
                
                surfaceNet_load.load_state_dict(surfaceNet.state_dict())
                eNet_load.load_state_dict(eNet.state_dict())
                
                for i, data in enumerate(dataloader, 0):

                    t_start = time.time()

                    i_append +=1                   
                    _,self.cubic_num,_,_,_ = data['cvc'].size()
                    
                    output = 0
                    w_total = 0
                    
                    for i_c in range(self.cubic_num):
                        s = surfaceNet(data['cvc'][:,i_c,...].to(device))
                        w = eNet(data['embedding'][:,i_c,...].to(device))
                        #w = w.detach()
                        #s = s.detach()
                        if(self._datasetName == 'MVS2d'):
                            w_total += (w[...,None,None])
                            output += s * w[...,None,None]
                        else:
                            w_total += (w[...,None,None,None])
                            output += s * w[...,None,None,None]
                    
                    output = output/(w_total + 1e-15)
                    
                    output_load = 0
                    w_total_load = 0
                    
                    for i_c in range(self.cubic_num):
                        s_load = surfaceNet_load(data['cvc'][:,i_c,...].to(device)).detach()
                        w_load = eNet_load(data['embedding'][:,i_c,...].to(device)).detach()
                        #w = w.detach()
                        #s = s.detach()
                        w_total_load += w_load[...,None,None].detach()
                        output_load += s_load * w_load[...,None,None]
                        
                    
                    output_load = output_load/(w_total_load + 1e-15)
                    
                    
                    data_surface_old = data['surface'].to(device)
                    
                    data_surface = self.blur_truth(data_surface = data_surface_old, output_detach = output_load, 
                                                   interpolation = interpolation, add_mass = add_mass, minus_mass = minus_mass)
                    
                    if(show):
                        fig, axes = plt.subplots(ncols = 3)
                        axes[0].imshow(data_surface[0,0], cmap = plt.cm.gray)
                        axes[1].imshow(data_surface_old[0,0], cmap = plt.cm.gray)
                        axes[2].imshow(output.detach()[0,0], cmap = plt.cm.gray)
                        plt.show()
                      
                    if(teacher_type == 'MSE'):
                        loss_teacher = loss_mse(output, data_surface)
                    else:
                        if(use_augment):
                            if(self._datasetName == 'MVS2d'):
                                alpha = 1- data['surface'].to(device).sum((1,2,3))/params._cube_D**2
                                alpha = alpha[:,None,None,None]
                            else:
                                alpha = 1- data['surface'].to(device).sum((1,2,3,4))/params._cube_D**3
                                alpha = alpha[:,None,None,None,None]
                            #alpha_num = data['surface'].to(device).sum()
                            #alpha = 1- alpha_num / (params.batch_size * params._cube_D**2)

                            #alpha = 0.999
                        else:
                            alpha = 0.5


                        loss_teacher = - (alpha * torch.log(output + 1e-15) * data_surface + (1-alpha) * torch.log(1-output + 1e-15) * (1-data_surface)).mean()


                    if(loss_type == 'gan'):
                        
                        
                        for _ in range(d_step):
                            
                            disNet.zero_grad()
                            
                            d_output_real = disNet(data['cvc'].to(device), data['embedding'].to(device),data_surface)
                            real_label = (torch.ones(d_output_real.size())).type(torch.cuda.FloatTensor)
                            #print(d_output_real)
                            d_real_error = -torch.log(d_output_real + 1e-6).mean()
                            d_real_error.backward()

                            
                            output_detach = output.detach()
                            if(use_fg):
                                output_detach = fg(output_detach)
                            d_output_fake = disNet(data['cvc'].to(device), data['embedding'].to(device),output_detach)
                            fake_label = (torch.zeros(d_output_real.size())).type(torch.cuda.FloatTensor)
                            #print(d_output_fake)
                            d_fake_error = -torch.log(1 - d_output_fake + 1e-6).mean()
                            d_fake_error.backward()
                            
                            d_loss = d_fake_error + d_real_error
                            
                            optimizer_d.step()
                            
                        for i_step in range(g_step):
                            
                            surfaceNet.zero_grad()
                            eNet.zero_grad()
                            if(use_fg):
                                fg.zero_grad()
                                output = fg(output)
                                if(use_fg_teacher):
                                    loss_teacher_fg = - (alpha * torch.log(output + 1e-15) * data_surface + (1-alpha) * torch.log(1-output + 1e-15) * (1-data_surface)).mean()
                                
                            fake_decision = disNet(data['cvc'].to(device), data['embedding'].to(device),output)
                            real_label = (torch.ones(fake_decision.size())).type(torch.cuda.FloatTensor)
                
                            g_error = -torch.log(fake_decision + 1e-15).mean()
                            if(use_fg_teacher):
                                g_error = alpha_decay_fg * g_error + loss_teacher_fg
                            alpha_error = g_error.detach()/(loss_teacher.detach() + 1e-10)
                            #alpha_error = 100.0
                            #print('alpha_error',alpha_error)
                            if(use_teacher):
                                g_error = g_error * alpha_error * alpha_decay + loss_teacher
                                #g_error = g_error * alpha_error * alpha_decay + loss_teacher
                            if(i_step <(g_step - 1)):
                                g_error.backward(retain_graph=True)
                            
                            if(use_fg):
                                optimizer_fg.step()
                            optimizer_s.step()
                            optimizer_e.step()
                            
                       
                            
                    elif(loss_type == 'Wgan'):
                        
                        one = torch.FloatTensor([1]).to(device)
                        mone = one * -1
                        for i_d in range(d_step):
                            
                            if(use_fg_transfer):
                                fg_transfer.zero_grad()
                            disNet.zero_grad()
                            # Clamp parameters to a range [-c, c], c=self.weight_cliping_limit
                            for p in disNet.parameters():
                                p.data.clamp_(-weight_cliping_limit, weight_cliping_limit)
                            
                            self.data_surface = data_surface
                            
                            if(use_fg_transfer):
                                data_surface_transfer = fg_transfer(data_surface)
                                d_output_real = disNet(data['cvc'].to(device), data['embedding'].to(device),data_surface_transfer)
                                variance_loss = data_surface.std(dim = 0, unbiased=False).mean()
                                self.data_surface_transfer = data_surface_transfer.detach()
                            else:
                                d_output_real = disNet(data['cvc'].to(device), data['embedding'].to(device),data_surface)
                                variance_loss = 0
                            
                            
                            #real_label = (torch.ones(d_output_real.size())).type(torch.cuda.FloatTensor)
                            
                            d_real_error = d_output_real.sum(0).view(1)
                            #d_real_error.backward(one)

                            
                            output_detach = output.detach()
                            self.output_detach = output_detach
                        
                            if(use_fg):
                                fg_output = fg(output_detach).detach()
                                
                                if(use_fg_transfer):
                                    fg_output = fg_output.pow(fg_tau)/(fg_output.pow(fg_tau) + (1-fg_output).pow(fg_tau))
                                    fg_output_transfer = fg_transfer(fg_output)
                                    d_output_fake = disNet(data['cvc'].to(device), data['embedding'].to(device),fg_output_transfer)
                                else:
                                    d_output_fake = disNet(data['cvc'].to(device), data['embedding'].to(device),fg_output)
                                
                            else:
                                if(use_fg_transfer):
                                    output_detach_transfer = fg_transfer(output_detach)
                                    d_output_fake = disNet(data['cvc'].to(device), data['embedding'].to(device),output_detach_transfer)
                                else:
                                    d_output_fake = disNet(data['cvc'].to(device), data['embedding'].to(device),output_detach)
                                
                            #fake_label = (torch.zeros(d_output_real.size())).type(torch.cuda.FloatTensor)
                            
                            d_fake_error = d_output_fake.sum(0).view(1)
                            #d_fake_error.backward(mone)
                            
                            
                            
                            d_loss = -d_fake_error + d_real_error - variance_loss * god_w
                            #print('variance_loss',variance_loss)
                            
                            if(i_d < d_step - 1):
                                d_loss.backward(retain_graph=True)
                            else:
                                d_loss.backward()
        
                            optimizer_d.step()
                            if(use_fg_transfer):
                                optimizer_fg_transfer.step()
                            
                        for i_g in range(g_step):
                            
                            
                            surfaceNet.zero_grad()
                            eNet.zero_grad()
                            if(use_fg):
                                fg.zero_grad()
                                if(fg_detach):
                                    output = output.detach()
                                output = fg(output)#####change here#########
                                output = output.pow(fg_tau) / (output.pow(fg_tau) + (1-output).pow(fg_tau))
                                
                                if(use_fg_teacher): 
                                    #alpha = 0.5
                                    if(teacher_type_fg == 'MSE'):
                                        loss_teacher_fg = loss_mse(output, data_surface)
                                    elif(teacher_type_fg == 'MIXED'):
                                        if(use_augment_fg):
                                            alpha = 1- data['surface'].to(device).sum((1,2,3))/params._cube_D**2
                                            alpha = alpha[:,None,None,None]

                                            #alpha_num = data['surface'].to(device).sum()
                                            #alpha = 1- alpha_num / (params.batch_size * params._cube_D**2)

                                            #alpha = 0.999
                                        else:
                                            alpha = 0.5
                                        loss_teacher_fg = - (alpha * torch.log(output + 1e-15) * data_surface + (1-alpha) * torch.log(1-output + 1e-15) * (1-data_surface)).mean()
                                        alpha_losses = loss_teacher_fg.detach() / (loss_mse(output, data_surface).detach()+1e-9)
                                        loss_teacher_fg += loss_mse(output, data_surface) * alpha_losses * 5
                                    else:
                                        if(use_augment_fg):
                                            if(self._datasetName == 'MVS2d'):
                                                alpha = 1- data['surface'].to(device).sum((1,2,3))/params._cube_D**2
                                                alpha = alpha[:,None,None,None]
                                            else:
                                                alpha = 1- data['surface'].to(device).sum((1,2,3,4))/params._cube_D**3
                                                alpha = alpha[:,None,None,None,None]
                                            #alpha_num = data['surface'].to(device).sum()
                                            #alpha = 1- alpha_num / (params.batch_size * params._cube_D**2)

                                            #alpha = 0.999
                                        else:
                                            alpha = 0.5
                                        loss_teacher_fg = - (alpha * torch.log(output + 1e-15) * data_surface + (1-alpha) * torch.log(1-output + 1e-15) * (1-data_surface)).mean()

                            output_transfer = fg_transfer(output)
                            fake_decision = disNet(data['cvc'].to(device), data['embedding'].to(device),output_transfer)
                            #real_label = (torch.ones(fake_decision.size())).type(torch.cuda.FloatTensor)
                            g_error = fake_decision.mean()
                            if(use_fg_teacher):
                                if(use_dynamic_alpha):
                                    alpha_error = loss_teacher_fg.detach()/(g_error.detach()+1e-10)
                                else:
                                    alpha_error = 1.0
                                g_error = g_error * alpha_error + loss_teacher_fg / alpha_decay_fg
                                
                            if(use_dynamic_alpha):
                                alpha_error = loss_teacher.detach()/(g_error.detach()+1e-10)
                            else:
                                alpha_error = 1.0
                            
                            if(use_teacher):
                                g_error = g_error * alpha_error  + loss_teacher / alpha_decay
                                
                            if(i_g < g_step - 1):
                                g_error.backward(retain_graph=True)
                            else:
                                g_error.backward()
                            
                            if(use_fg):
                                optimizer_fg.step()
                            optimizer_s.step()
                            optimizer_e.step()
                            
                       
                              
                    threshold = 0.5
                    
                    ground_truth_binary = data_surface_old.type(torch.cuda.ByteTensor)
                    thre_output_binary1 = (output > threshold)
                    accuracy, completeness, f_distance = self.batch_chamfer_loss(thre_output_binary1, ground_truth_binary )
                    
                    accuracy_ave += accuracy
                    completeness_ave += completeness
                    f_distance_ave += f_distance
                    
                    if(i_append % reward_epoch == 0):
                        
                        self.accuracy_list.append(accuracy_ave/reward_epoch)
                        self.completeness_list.append(completeness_ave/reward_epoch)
                        self.f_distance_list.append(f_distance_ave/reward_epoch)
                        accuracy_ave = 0
                        completeness_ave = 0
                        f_distance_ave = 0
                        print('acc, comp, f_dis',accuracy, completeness, f_distance)
                        
                        plt.subplot(311)
                        plt.plot(self.accuracy_list)
                        plt.subplot(312)
                        plt.plot(self.completeness_list)
                        plt.subplot(313)
                        plt.plot(self.f_distance_list)
                    
                    t_end = time.time()
                    if i % 5 == 0:
                        print('[%d/%d][%d/%d]\tLossD: %.5f|||LossG:%.5f\tTime for prepare and train:%.4fs'
                              % (epoch, num_epochs, i, len(dataloader),
                                 d_loss, g_error, t_end - t_start))
                        

                       

    def train(self,
              num_epochs = 100,
              model_list = [9],
              batch_size = 8,
              lr = 0.001,
              loss_alpha = 1.0,
              clip_min = 0.2,
              clip_max = 0.2,
              reward_epoch = 10,
              training_draw_iter = 50,
              validation_iter = 5,
              draw_iter = 5,
              save_iter = 5,
              use_augment = True,
              shuffle = False,
              train_model = 'attSurface',
              root = 'experiment/network/',
              root_board = './experiment/log/',
              root_ply = './experiment/cubes/',
              root_slice = './experiment/slice/',
              loss_type = 'rl',
              training_key = '030402'
              ):
        
        model_path = self.model_path
        
        loss_mse = nn.MSELoss()
        
        loss_show = 0
        if(train_model == 'attSurface'):
            optimizer_att = optim.SGD(attSurface.parameters(), lr = lr)
        else:
            optimizer_s = optim.SGD(surfaceNet.parameters(), lr = lr)
            optimizer_e = optim.SGD(eNet.parameters(), lr = lr)
            
        root_path = root + '%s/'%(self._datasetName)
        directory = root_path + '%s/%s'%(loss_type, training_key)
        key_path = root_board + '%s/'%(self._datasetName) + '%s/%s/'%(loss_type, training_key)
        ply_path = root_ply + '%s/'%(self._datasetName) + '%s/%s/'%(loss_type, training_key)
        slice_path = root_slice + '%s/'%(self._datasetName) + '%s/%s/'%(loss_type, training_key)
        self.ply_path = ply_path
        self.slice_path = slice_path
        
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        loss_show = 0
        
        iter_record = 0
        iter_model_record = 0
        
        for epoch in range(num_epochs):
            loss_epoch = 0
            torch.cuda.empty_cache()
            if(epoch%save_iter == 0):
                
                
                if(train_model == 'attSurface'):
                    model_name = '%s/%s/att_epoch%r'%(loss_type, training_key, epoch)
                    torch.save(attSurface.cpu().state_dict(), root_path + model_name)
                    attSurface.to(device)
                else:
                    model_name = '%s/%s/SurfaceNet_epoch%r'%(loss_type, training_key, epoch)
                    torch.save(surfaceNet.cpu().state_dict(), root_path + model_name)
                    surfaceNet.to(device)
                    model_name = '%s/%s/eNet_epoch%r'%(loss_type, training_key, epoch)
                    torch.save(eNet.cpu().state_dict(), root_path + model_name)
                    eNet.to(device)
            
            if(epoch % validation_iter == 0):
                if(epoch % draw_iter == 0):
                    #batch_size = 1
                    self.validationTest(surfaceNet, eNet, epoch_num = epoch, model_num = self._test_modelList[0], train_model = train_model, batch_size = batch_size)
                    self.validationTest(surfaceNet, eNet, epoch_num = epoch, model_num = self._test_modelList[1], train_model = train_model, batch_size = batch_size)
                    self.validation_loss_list.append(self.loss_validation)
                    
                else:
                    batch_size = 1
                    self.validationTest(surfaceNet, eNet, epoch_num = epoch, model_num = self._test_modelList[0], draw = False, train_model = train_model, batch_size = batch_size)
                    self.validationTest(surfaceNet, eNet, epoch_num = epoch, model_num = self._test_modelList[1], draw = False, train_model = train_model, batch_size = batch_size)
                    self.validation_loss_list.append(self.loss_validation)
                    
            i_append = 0
            accuracy_ave = 0
            completeness_ave = 0
            f_distance_ave = 0
            
            for model_num in model_list:
                
                dataset = Dataset(model_num, self.recorder)
                self.recorder = dataset.recorder
                #batch_size = 64

                dataloader = DataLoader(dataset, batch_size= batch_size,
                                        shuffle=shuffle, num_workers=12)
                
                loss_whole_model = 0
                for i, data in enumerate(dataloader, 0):
                    
                    #torch.cuda.empty_cache()
                    #print('memory:',torch.cuda.memory_allocated())
                    self.data_out = data
                    #iter_record += 1
                    i_append += 1
                    t_start = time.time()
                    self.data_test = data
                    
                    if(train_model == 'attSurface'):
                        attSurface.zero_grad()
                    else:
                        surfaceNet.zero_grad()
                        eNet.zero_grad()
                    
                    if(self._datasetName == 'MVS2d'):
                        batch_size_cvc,self.cubic_num,_,_,pixel_size = data['cvc'].size()
                        w_list = torch.zeros([batch_size_cvc, self.cubic_num])
                        s_list = torch.zeros([batch_size_cvc, self.cubic_num, 1,pixel_size,pixel_size ])
                    else:
                        batch_size_cvc,self.cubic_num,_,_,_,pixel_size = data['cvc'].size()
                        w_list = torch.zeros([batch_size_cvc, self.cubic_num])
                        s_list = torch.zeros([batch_size_cvc, self.cubic_num, 1,pixel_size,pixel_size,pixel_size])
                    
                    
                    if(train_model == 'attSurface'):
                        output = attSurface(data['cvc'].to(device))
                        
                    else:
                        output = 0
                        w_total = 0
                        
                        for i_c in range(self.cubic_num):
                        #for i_c in range(1):
                            s = surfaceNet(data['cvc'][:,i_c,...].to(device))

                            w = eNet(data['embedding'][:,i_c,...].to(device))
                            #w = w.detach()
                            #s = s.detach()

                            w_list[:,i_c] = w[:,0].detach().cpu()
                            s_list[:,i_c] = s.detach().cpu()
                            
                            if(self._datasetName == 'MVS2d'):
                                w_total += (w[...,None,None])
                                output += s * w[...,None,None]
                            else:
                                w_total += (w[...,None,None,None])
                                output += s * w[...,None,None,None]
                        #print(w_list)
                        output = output/(w_total + 1e-15)
                        #output = s
                        
                    if(loss_type == 'ppo_chamfer'):
                        
                        output_load = 0
                        w_total_load = 0           
                        for i_c in range(self.cubic_num):
                        #for i_c in range(1):
                            s = surfaceNet_load(data['cvc'][:,i_c,...].to(device))
                            w = eNet_load(data['embedding'][:,i_c,...].to(device))
                            w = w.detach()
                            s = s.detach()
                            if(self._datasetName == 'MVS2d'):
                                w_total_load += w[...,None,None].detach()
                                output_load += s * w[...,None,None]
                            else:
                                w_total_load += w[...,None,None,None].detach()
                                output_load += s * w[...,None,None,None]
                                
                        output_load = output_load/(w_total_load + 1e-15)
                        
                    
                    self.w_list = w_list
                    self.s_list = s_list
                    self.data = data
                    self.output= output.detach().cpu()
                    
                    
                    data_surface = data['surface'].to(device)
                    threshold = 0.7
                    ground_truth_binary = data_surface.type(torch.cuda.ByteTensor)
                    thre_output_binary1 = (output > threshold)
                    if not(loss_type == 'augment'):
                        accuracy, completeness, f_distance = self.batch_chamfer_loss(thre_output_binary1, ground_truth_binary )
                        accuracy_ave += accuracy
                        completeness_ave += completeness
                        f_distance_ave += f_distance

                    if(i_append % reward_epoch == 0):
                        
                        self.accuracy_list.append(accuracy_ave/reward_epoch)
                        self.completeness_list.append(completeness_ave/reward_epoch)
                        self.f_distance_list.append(f_distance_ave/reward_epoch)
                        accuracy_ave = 0
                        completeness_ave = 0
                        f_distance_ave = 0
                    
                    if(loss_type == 'MSE'):
                        loss = loss_mse(output, data_surface)
                        
                    elif(loss_type == 'augment'):
                        
                        if(use_augment):
                            if(self._datasetName == 'MVS2d'):
                                alpha = 1- data['surface'].to(device).sum((1,2,3))/params._cube_D**2
                                alpha = alpha[:,None,None,None]
                            else:
                                alpha = 1- data['surface'].to(device).sum((1,2,3,4))/params._cube_D**3
                                alpha = alpha[:,None,None,None,None]
                            #alpha_num = data['surface'].to(device).sum()
                            #alpha = 1- alpha_num / (params.batch_size * params._cube_D**2)

                            #alpha = 0.999
                        else:
                            alpha = 0.5
                        
                       
                        loss = - (alpha * torch.log(output + 1e-15) * data_surface + (1-alpha) * torch.log(1-output + 1e-15) * (1-data_surface)).mean()
            
                    elif(loss_type == 'ppo_chamfer'):
                        reward = self.surface_reward(groundturth = data_surface, 
                                                     output = output_load,
                                                     alpha = 0.5)
                        reward_ave = reward.mean()
                        
                        threshold = 0.5
                        ground_truth_binary_load = data_surface.type(torch.cuda.ByteTensor)
                        thre_output_binary_load = (output_load > threshold) #use the old policy to sample
                        batch_size,_,dimention,_ = output.size()
                        rand_tensor = torch.rand(batch_size,1,dimention,dimention).to(device)
                        rand_choice = (output > rand_tensor)
                        thre_output_binary2 = rand_choice
                        thre_output_binary = thre_output_binary_load  + thre_output_binary2 - thre_output_binary_load  * thre_output_binary2
                        thre_output_binary_load = thre_output_binary.type(torch.cuda.ByteTensor)
                        #thre_output_binary_load = thre_output_binary2
                        
                        batch_size,_,_,_ = ground_truth_binary.size()
                        loss = 0
                        flag = False
                        mean_accuracy = 0
                        mean_completeness = 0
                        mean_f_distance = 0
                        
                        for i_batch in range(batch_size):
                            index_truth_load = (ground_truth_binary_load[i_batch,0] == 1).nonzero().type(torch.cuda.FloatTensor)
                            index_output_load = (thre_output_binary_load[i_batch,0] == 1).nonzero().type(torch.cuda.FloatTensor)
                            #print(index_truth)
                            #print(index_output)
                            if(index_truth_load.size() == torch.Size([0]))or(index_output_load.size() == torch.Size([0])):
                                loss += torch.tensor(0).type(torch.cuda.FloatTensor)
                            else:
                                accuracy_reward_load, completeness_reward_load = self.chamfer_distance(index_truth_load[None,...], index_output_load[None,...])
                                _, accuracy_length_load = accuracy_reward_load.size()
                                _, completeness_length_load = completeness_reward_load.size()
                                
                                
                                prob_completeness = torch.masked_select(output[i_batch,0], ground_truth_binary_load[i_batch,0])
                                prob_accuracy = torch.masked_select(output[i_batch,0], thre_output_binary_load[i_batch,0])
                                
                                prob_completeness_load = torch.masked_select(output_load[i_batch,0], ground_truth_binary_load[i_batch,0])
                                prob_accuracy_load = torch.masked_select(output_load[i_batch,0], thre_output_binary_load[i_batch,0])
                                
                                
                                accuracy_load = accuracy_reward_load.sum().detach() / accuracy_length_load
                                completeness_load = completeness_reward_load.sum().detach() / completeness_length_load
                                mean_accuracy += accuracy_load
                                mean_completeness += completeness_load
                                mean_f_distance += (accuracy_load + completeness_load)
                                
                                alpha = (completeness_load + 0.1) / (accuracy_load + 0.1)
                                #alpha = 1
                                
                                r_clip_completeness = torch.clamp((1-prob_completeness)/(1-prob_completeness_load + 1e-15), 
                                                                  min = 1-clip_min, max = 1+clip_max)
                                r_clip_accuracy = torch.clamp((prob_accuracy)/(prob_accuracy_load + 1e-15), 
                                                                  min = 1-clip_min, max = 1+clip_max)
                                
                                loss += 0.1 * ((r_clip_completeness * completeness_reward_load / completeness_length_load).sum() \
                                        + alpha * (r_clip_accuracy * accuracy_reward_load / accuracy_length_load).sum())
                                #print('>>>>>>>>',completeness_reward,accuracy_reward)
                               
                        #self.accuracy_list.append(mean_accuracy.cpu()/batch_size)
                        #self.completeness_list.append(mean_completeness.cpu()/batch_size)
                        #self.f_distance_list.append(mean_f_distance.cpu()/batch_size)

                        and_matrix_load = (thre_output_binary_load * ground_truth_binary_load).type(torch.cuda.FloatTensor)
                        oo_matrix_load = ((1 - thre_output_binary_load) * (1 - ground_truth_binary_load)).type(torch.cuda.FloatTensor)
                        
                        #alpha = and_matrix.sum((1,2,3)) / oo_matrix.sum((1,2,3))
                        #alpha = alpha[:,None,None,None]
                        
                        alpha = and_matrix_load.sum((1,2,3)) / thre_output_binary_load.type(torch.cuda.FloatTensor).sum((1,2,3))
                        alpha = alpha[:,None,None,None]
                        
                        if(loss == torch.tensor(0).type(torch.cuda.FloatTensor)):
                            #alpha = 0.5
                            if(self._datasetName == 'MVS2d'):
                                alpha = 1- data['surface'].to(device).sum((1,2,3))/params._cube_D**2
                                alpha = alpha[:,None,None,None]
                            else:
                                alpha = 1- data['surface'].to(device).sum((1,2,3,4))/params._cube_D**3
                                alpha = alpha[:,None,None,None,None]

                            loss = - (alpha * torch.log(output + 1e-15) * data_surface + (1-alpha) * torch.log(1-output + 1e-15) * (1-data_surface)).mean()
            
                        else:
                            #loss_2 = (and_matrix * torch.log(output + 1e-15) - alpha * oo_matrix * torch.log(1 - output + 1e-15)).sum()
                            #loss -= loss_alpha * loss_2
                            
                            alpha = 1 / (and_matrix_load.sum((1,2,3)) + 1)
                            alpha = alpha[:,None,None,None]
                            r_clip_output = torch.clamp(output/(output_load + 1e-15), min = 1-clip_min, max = 1+clip_max)
                            loss_2 = (alpha * and_matrix_load * r_clip_output).sum()
                            
                            alpha1 = 1- data['surface'].to(device).sum((1,2,3))/params._cube_D**2
                            alpha1 = alpha1[:,None,None,None]
                            loss2 = - (alpha1 * torch.log(output + 1e-15) * data_surface + (1-alpha1) * torch.log(1-output + 1e-15) * (1-data_surface)).mean()
            
                            loss -= loss_alpha * loss_2
                            #loss-=0
                        
                        
                    
                    elif(loss_type == 'rl_chamfer'):
                    
                        reward = self.surface_reward(groundturth = data_surface, 
                                                     output = output,
                                                     alpha = 0.5)
                        reward_ave = reward.mean()
                        
                        
                    
                        threshold = 0.7
                        ground_truth_binary = data_surface.type(torch.cuda.ByteTensor)
                        thre_output_binary1 = (output > threshold)
                        
                        batch_size,_,dimention,_ = output.size()
                        rand_tensor = torch.rand(batch_size,1,dimention,dimention).to(device)
                        rand_choice = (output > rand_tensor)
                        thre_output_binary2 = rand_choice
                        
                        thre_output_binary = thre_output_binary1 + thre_output_binary2 - thre_output_binary1 * thre_output_binary2
                        thre_output_binary = thre_output_binary.type(torch.cuda.ByteTensor)

                        thre_output_binary = thre_output_binary1
                        
                        batch_size,_,_,_ = ground_truth_binary.size()
                        loss = 0
                        flag = False
                        mean_accuracy = 0
                        mean_completeness = 0
                        mean_f_distance = 0
                        for i_batch in range(batch_size):
                            index_truth = (ground_truth_binary[i_batch,0] == 1).nonzero().type(torch.cuda.FloatTensor)
                            index_output = (thre_output_binary[i_batch,0] == 1).nonzero().type(torch.cuda.FloatTensor)
                            #print(index_truth)
                            #print(index_output)
                            if(index_truth.size() == torch.Size([0]))or(index_output.size() == torch.Size([0])):
                                loss += torch.tensor(0).type(torch.cuda.FloatTensor)
                            else:
                                accuracy_reward, completeness_reward = self.chamfer_distance(index_truth[None,...], index_output[None,...])
                                _, accuracy_length = accuracy_reward.size()
                                _, completeness_length = completeness_reward.size()
                                #print('completeness_length',accuracy_length, completeness_length)
                                #print('accuracy_reward',accuracy_reward)
                                #print('completeness_reward',completeness_reward)
                                prob_completeness = torch.masked_select(output[i_batch,0], ground_truth_binary[i_batch,0])
                                prob_accuracy = torch.masked_select(output[i_batch,0], thre_output_binary[i_batch,0])
                                
                                accuracy = accuracy_reward.sum().detach() / accuracy_length
                                completeness = completeness_reward.sum().detach() / completeness_length
                                mean_accuracy += accuracy
                                mean_completeness += completeness
                                mean_f_distance += (accuracy + completeness)
                                
                                alpha = (completeness + 0.1) / (accuracy + 0.1)
                                alpha = 1
                                loss += 1 * ((torch.log(1 - prob_completeness + 1e-15) * completeness_reward / completeness_length).sum() \
                                        + alpha * (torch.log(prob_accuracy + 1e-15) * accuracy_reward / accuracy_length).sum())
                                #print('>>>>>>>>',completeness_reward,accuracy_reward)
                                
                        #self.accuracy_list.append(mean_accuracy.cpu()/batch_size)
                        #self.completeness_list.append(mean_completeness.cpu()/batch_size)
                        #self.f_distance_list.append(mean_f_distance.cpu()/batch_size)

                        and_matrix = (thre_output_binary * ground_truth_binary).type(torch.cuda.FloatTensor)
                        oo_matrix = ((1 - thre_output_binary) * (1 - ground_truth_binary)).type(torch.cuda.FloatTensor)
                        
                        #alpha = and_matrix.sum((1,2,3)) / oo_matrix.sum((1,2,3))
                        #alpha = alpha[:,None,None,None]
                        
                        alpha = and_matrix.sum((1,2,3)) / thre_output_binary.type(torch.cuda.FloatTensor).sum((1,2,3))
                        alpha = alpha[:,None,None,None]
                        
                        if(loss == torch.tensor(0).type(torch.cuda.FloatTensor)):
                            #alpha = 0.5
                            alpha = 1- data['surface'].to(device).sum((1,2,3))/params._cube_D**2
                            alpha = alpha[:,None,None,None]

                            loss = - (alpha * torch.log(output + 1e-15) * data_surface + (1-alpha) * torch.log(1-output + 1e-15) * (1-data_surface)).mean()
            
                        else:
                            #loss_2 = (and_matrix * torch.log(output + 1e-15) - alpha * oo_matrix * torch.log(1 - output + 1e-15)).sum()
                            #loss -= loss_alpha * loss_2
                            
                            alpha = 1 / (and_matrix.sum((1,2,3)) + 1)
                            alpha = alpha[:,None,None,None]
                            loss_2 = (alpha * and_matrix * torch.log(output + 1e-15)).sum()
                            loss -= loss_alpha * loss_2
                            #loss-=0
                        
    
                    elif(loss_type == 'rl'):
                    
                        reward = self.surface_reward(groundturth = data_surface, 
                                                     output = output,
                                                     alpha = 0.5)
                        reward_ave = reward.mean()
                        reward_matrix = self.matrix_reward(groundturth = data_surface, 
                                                             output = output,
                                                             alpha = 0.5)
                        #print('reward',reward)
                        alpha = 1- data['surface'].to(device).sum((1,2,3))/params._cube_D**2
                        alpha = alpha[:,None,None,None]

                        #alpha_num = data['surface'].to(device).sum()
                        #alpha = 1- alpha_num / (params.batch_size * params._cube_D**2)

                        #alpha = 0.999
                        #alpha = 0.5
                        #loss =  - (reward * (alpha * torch.log(output + 1e-15) * data_surface + (1 - alpha) * torch.log(1-output + 1e-15) * (1-data_surface)).sum(dim = [1,2,3])).sum()
                        
                        #batch_size,_,dimention,_ = output.size()
                        #rand_tensor = torch.rand(batch_size,1,dimention,dimention).to(device)
                        #rand_choice = (output > rand_tensor).type(torch.cuda.FloatTensor)
                        #loss =  -((reward - reward_ave) * (rand_choice * torch.log(output + 1e-15) * data_surface).sum(dim = [1,2,3])).sum()

                        
                        output_d = output.detach()
                        
                        thre_output = (output_d > 0.7).type(torch.cuda.FloatTensor)
                        
                        #batch_size,_,dimention,_ = output.size()
                        #rand_tensor = torch.rand(batch_size,1,dimention,dimention).to(device)
                        #rand_choice = (output > rand_tensor).type(torch.cuda.FloatTensor)
                        #thre_output = rand_choice
                        
                        alpha = 1- thre_output.sum((1,2,3))/params._cube_D**2
                        alpha = alpha[:,None,None,None]
                        #alpha = 0.5
                        
                        loss = -(reward_matrix * (alpha * torch.log(output + 1e-15) * thre_output + (1 - alpha) * torch.log(1-output + 1e-15) * (1-thre_output))).sum()

                        self.output_test = output
                        
                    #loss /= (params._cube_D**2)
                    loss_show += loss.detach().cpu()
                    loss_whole_model += loss.detach().cpu()
                    t_middle = time.time()
                    
                    if(loss_type == 'ppo_chamfer'):
                        surfaceNet_load.load_state_dict(surfaceNet.state_dict())
                        eNet_load.load_state_dict(eNet.state_dict())
                    
                    
                    loss.backward()
                    if(train_model == 'attSurface'):
                        optimizer_att.step()
                    else:
                        optimizer_s.step()
                        optimizer_e.step()

                    t_end = time.time()
                    
                    
                    if i_append % training_draw_iter == 0:
                        
                        self.draw_sample(self.data, 
                                         self.output, 
                                         show_num = 3, 
                                         train_model = train_model,
                                         epoch_num = epoch, 
                                         mode = 'train',
                                         file_root = self.slice_path,
                                         model_num = model_num)
                        
                        if(self._datasetName != 'MVS2d'):
                            self.save_ply(data = self.data,
                                          output = self.output,
                                          epoch_num = epoch, 
                                          mode = 'train',
                                          file_root = self.ply_path,
                                          model_num = model_num)
                        
                        print('[%d/%d][%d/%d]\tLoss: %.4f\tTime for prepare and train:%.4fs,%.4fs'
                              % (epoch, num_epochs, i, len(dataloader),
                                 loss_show/(i+1), t_middle - t_start, t_end - t_middle))
                        if not(loss_type == 'augment'):
                            print('acc,comp,fd:', accuracy, completeness, f_distance)

                            plt.subplot(311)
                            plt.plot(self.accuracy_list)
                            plt.subplot(312)
                            plt.plot(self.completeness_list)
                            plt.subplot(313)
                            plt.plot(self.f_distance_list)

                        if(loss_type == 'rl'):
                            print('reward:', reward_ave.cpu())
                            self.reward_list.append(reward_ave.cpu().numpy())
                            plt.plot(self.reward_list)
                            #plt.show()
                        if(loss_type == 'rl_chamfer'):
                            #print('reward:', reward_ave.cpu())
                            #self.reward_list.append(reward_ave.cpu().numpy())
                            plt.subplot(311)
                            plt.plot(self.accuracy_list)
                            plt.subplot(312)
                            plt.plot(self.completeness_list)
                            plt.subplot(313)
                            plt.plot(self.f_distance_list)
                            
                        loss_show = 0
                        #self.draw_sample(data, output, detach = True)
                        
                    memory_used = torch.cuda.memory_allocated()
                    max_memory = torch.cuda.max_memory_allocated()
                    
                    self.writer.add_scalars(key_path + 'memory', {'memory':100 * memory_used/(max_memory+0.0)},iter_record)
                    self.writer.add_scalars(key_path + 'training', {'loss_augment':loss.detach().cpu()},iter_record)
                    iter_record += 1
                    #writer.add_scalars(key_path + 'training_each_model/' + str(model_num).zfill(3), {'loss_augment':loss.detach().cpu()},iter_record)
                    
                self.writer.add_scalars(key_path + 'training_whole_model', {'loss_augment':loss_whole_model/(i+1)},iter_model_record)
                iter_model_record += 1
                loss_epoch += loss_whole_model/(i+1)
            self.writer.add_scalars(key_path + 'training_epoch', {'loss_augment':loss_epoch/len(model_list)},epoch)
                
   

    def load_model(self,
                   model_path = 'experiment/network/',
                   model_type = 'load',
                   model_name1 = 'no_w_lr_0.0010_augument_epoch30', 
                   model_name2 = 'a',
                   model_name3 = 'None',
                   train_model = 'none'):
        
        
        #model_name = 'no_w_lr_0.0010/epoch' + str(80)
        PATH1 = model_path + model_name1
        PATH2 = model_path + model_name2
        PATH3 = model_path + model_name3
        if(train_model == 'attSurface'):
            attSurface.load_state_dict(torch.load(PATH1))
            attSurface.to('cpu')
            attSurface.load_state_dict(torch.load(PATH1))
            attSurface.eval()
            attSurface.to(device)
        else:
            if(model_name3 != 'None'):
                fg.to('cpu')
                fg.load_state_dict(torch.load(PATH3))
                fg.eval()
                fg.to(device)
            if(model_type == 'load'):
                surfaceNet_load.load_state_dict(torch.load(PATH1))
                surfaceNet_load.eval()
                eNet_load.load_state_dict(torch.load(PATH2))
                eNet_load.eval()
                surfaceNet_load.to(device)
                eNet_load.to(device)

                #eNet_load.load_state_dict(torch.load(PATH))
                #eNet_load.eval()

                #self.validationTest(surfaceNet_load, eNet_load, model_num = 101)
            elif(model_type == 'train'):
                surfaceNet.to('cpu')
                surfaceNet.load_state_dict(torch.load(PATH1))
                surfaceNet.eval()
                eNet.to('cpu')
                eNet.load_state_dict(torch.load(PATH2))
                eNet.eval()
                surfaceNet.to(device)
                eNet.to(device)

                #eNet_load.load_state_dict(torch.load(PATH))
                #eNet_load.eval()

                #self.validationTest(surfaceNet, eNet, model_num = 1)

    
    def validationTest(self,
                        surfaceNet,
                        eNet,
                       epoch_num = 10000,
                        model_num = 1,
                       threshold = 0.7,
                        use_fg = False,
                       fg_tau = 1.0,
                       count_chamfer = False,
                        train_model = 'none',
                        type_d = 'validation',
                        batch_size = 8,
                        draw = True):
        '''
        get the validation score and result
        '''

        #torch.cuda.empty_cache()
        
        print('start validation test')
        print('##############################################################')
        self.test_dataset = Dataset(model_num, recorder = self.recorder)
        self.recorder = self.test_dataset.recorder
        self.data_test = DataLoader(self.test_dataset, batch_size= batch_size,
                            shuffle=False, num_workers=12)
        
        self.loss_validation = 0
        acc_t = 0
        comp_t = 0
        f_t = 0
        
        for i, data in enumerate(self.data_test, 0):

            if(train_model == 'attSurface'):
                output = attSurface(data['cvc'].to(device)).detach().cpu()
                
                        
            else:
                w_list = []
                if(self._datasetName == 'MVS2d'):
                    _,self.cubic_num,_,_,_ = data['cvc'].size()
                else:
                    _,self.cubic_num,_,_,_,_= data['cvc'].size()
                output = 0
                w_total = 0
                for i_c in range(self.cubic_num):
                #for i_c in range(1):
                    s = surfaceNet(data['cvc'][:,i_c,...].to(device)).detach()
                    w = eNet(data['embedding'][:,i_c,...].to(device)).detach()
                    #self.data_cvc = data['cvc'][:,i_c,...].to(device)
                    #self.data_e = data['embedding'][:,i_c,...].to(device)
                    #w = w.detach()
                    #s = s.detach()
                    #w_list.append(w)
                    if(self._datasetName == 'MVS2d'):
                        w_total += (w[...,None,None])
                        output += s * w[...,None,None]
                    else:
                        w_total += (w[...,None,None,None])
                        output += s * w[...,None,None,None]

                #w_total = w_total.detach()
                output = output/(w_total + 1e-15)
                output = output.cpu()
                #output = s
            if(use_fg):
                output_old = output
                output = fg(output.to(device)).detach().cpu()
                output = output.pow(fg_tau) / (output.pow(fg_tau) + (1-output).pow(fg_tau))
            
            memory_used = torch.cuda.memory_allocated()
            max_memory = torch.cuda.max_memory_allocated()
            #print('memory used',memory_used/(max_memory+0.0))
            
            data_surface = data['surface']
            
            threshold = threshold
            ground_truth_binary = data_surface.type(torch.ByteTensor)
            thre_output_binary1 = (output > threshold)
            if(count_chamfer):
                accuracy, completeness, f_distance = self.batch_chamfer_loss(thre_output_binary1, ground_truth_binary )
            else:
                accuracy, completeness, f_distance = (10,10,10)
            acc_t += accuracy
            comp_t += completeness
            f_t += f_distance
            
            #output = output.detach().cpu()
            
            if(draw and (i<10)):
                if(use_fg):
                    self.draw_sample(data = data,
                                      output = output,
                                      epoch_num = epoch_num,
                                      mode = 'test',
                                      file_root = self.slice_path, 
                                      model_num = model_num,
                                     output_old = output_old,use_fg = True, 
                                     show_num = 1, train_model = train_model)
                    
                else:
                    self.draw_sample(data = data,
                                      output = output,
                                      epoch_num = epoch_num,
                                      mode = 'test',
                                      file_root = self.slice_path, 
                                      model_num = model_num,
                                     show_num = 1, train_model = train_model)
                    
                    if(self._datasetName != 'MVS2d'):
                        self.save_ply(data = data,
                                      output = output,
                                      epoch_num = epoch_num,
                                      mode = 'test',
                                      file_root = self.ply_path, 
                                      model_num = model_num)

                
            
            #alpha = 1- data['surface'].to(device).sum((1,2,3))/params._cube_D**2
            #alpha = alpha[:,None,None,None]

            #alpha_num = data['surface'].sum()
            #alpha = 1- alpha_num / (params.batch_size * params._cube_D**2)

            #alpha = 0.999
            alpha = 0.5
            
            loss = - (alpha * torch.log(output + 1e-15) * data['surface'] + (1-alpha) * torch.log(1-output + 1e-15) * (1-data['surface'])).mean()
            #loss /= (params._cube_D**2)

            self.loss_validation +=loss
            
        acc_t /= i
        comp_t /= i
        f_t /= i
        
        if(model_num < 100):
            self.accuracy_list_test.append(acc_t)
            self.completeness_list_test.append(comp_t)
            self.f_distance_list_test.append(f_t)
        else:
            self.accuracy_list_test2.append(acc_t)
            self.completeness_list_test2.append(comp_t)
            self.f_distance_list_test2.append(f_t)
            
        print('acc:%r||comp:%r||f_distance:%r'%(accuracy, completeness, f_distance))
        
        if(draw):
            if(model_num < 100):
                plt.subplot(311)
                plt.plot(self.accuracy_list_test)
                plt.title('accuravy_test')
                plt.subplot(312)
                plt.plot(self.completeness_list_test)
                plt.title('completeness_test')
                plt.subplot(313)
                plt.plot(self.f_distance_list_test)
                plt.title('f_distance_test')
                plt.show()
            else:
                plt.subplot(311)
                plt.plot(self.accuracy_list_test2)
                plt.title('accuravy_test')
                plt.subplot(312)
                plt.plot(self.completeness_list_test2)
                plt.title('completeness_test')
                plt.subplot(313)
                plt.plot(self.f_distance_list_test2)
                plt.title('f_distance_test')
                plt.show()
        
        self.loss_validation = self.loss_validation.tolist()
        self.loss_validation /= i
        
        print (self.loss_validation)
        print('end validation test')
        print('##############################################################')
        return self.loss_validation
    
    def save_ply(self,
             data,
             output, 
             cvc_num = 3, 
             show_num = 3, 
             mode = 'train',
             file_root = 'rub',  
             epoch_num = 10000, 
             model_num = 10000, 
             threshold = 0.5):
    
        batch_size, view_num, _,_,_,image_size = data['cvc'].shape
        xx, yy,zz = np.meshgrid(np.linspace(0,1,image_size), np.linspace(0,1,image_size),np.linspace(0,1,image_size))
        X = xx.flatten()[:,None]
        Y = yy.flatten()[:,None]
        Z = zz.flatten()[:,None]
        xyz = np.concatenate((X,Y,Z), axis = 1)
        if(cvc_num > view_num):
            cvc_num = view_num
        if(show_num > batch_size):
            show_num = batch_size



        interval_between =  2  
        interval_cvc =  2 
        interval_o_s =  3

        for show_i in range(show_num):

            xyz_total = np.zeros((image_size ** 3 * cvc_num * 2, 3))
            c_total = np.zeros((image_size ** 3 * cvc_num * 2, 3))
            
            idx = data['idx_validCubes'][show_i].item()
            
            for cvc_i in range(cvc_num):
                c1 = data['cvc'][show_i,cvc_i,0:3,:,:,:]
                c1_out = c1.flatten(start_dim = 1, end_dim = 3).transpose(0,-1).numpy()
                c1_out = (c1_out + 0.5) * 256
                c2 = data['cvc'][show_i,cvc_i,3:,:,:,:]
                c2_out = c2.flatten(start_dim = 1, end_dim = 3).transpose(0,-1).numpy()
                c2_out = (c2_out + 0.5) * 256

                c_total[image_size ** 3 * cvc_i*2 : image_size ** 3 * (cvc_i*2+1), :] = c1_out
                c_total[image_size ** 3 * (cvc_i*2+1) : image_size ** 3 * (cvc_i*2+2), :] = c2_out

                xyz_total[image_size ** 3 * cvc_i*2 : image_size ** 3 * (cvc_i*2+1), :] = xyz + np.array([interval_cvc * cvc_i,0,0])
                xyz_total[image_size ** 3 * (cvc_i*2+1) : image_size ** 3 * (cvc_i*2+2), :] = xyz + np.array([interval_cvc * cvc_i,interval_between,0])

            s = data['surface'][show_i,0,:,:,:].numpy()
            s_out = (s == 1).flatten()
            xyz_s = xyz[s_out,:]

            o = output[show_i,0,:,:,:].numpy()
            o_out = (o > threshold).flatten()
            xyz_o = xyz[o_out,:] + np.array([interval_o_s,0,0])

            o = output[show_i,0,:,:,:][None,:,:,:]
            o_dense = o * torch.ones((3,1,1,1))
            o_dense = o_dense.flatten(start_dim = 1, end_dim = 3).transpose(0,-1).numpy()
            o_dense = (1 - o_dense) * 128

            xyz_s = np.concatenate((xyz_s, xyz_o), axis = 0)

            if(mode == 'train'):
                path_cvc = file_root + 'train/cvc/epoch_%s/model_%r__cvc_id_%r__show_id_%r.ply'%(str(epoch_num).zfill(3), model_num, idx, show_i)
                path_s = file_root + 'train/surface/epoch_%s/model_%r__cvc_id_%r__show_id_%r.ply'%(str(epoch_num).zfill(3), model_num, idx, show_i)
                path_o = file_root + 'train/output/epoch_%s/model_%r__cvc_id_%r__show_id_%r.ply'%(str(epoch_num).zfill(3), model_num, idx, show_i)

                #path_out = file_root + '/surface'
                self.d2s.save2ply(path_cvc, xyz_total, c_total)
                self.d2s.save2ply(path_s, xyz_s)
                self.d2s.save2ply(path_o, xyz, o_dense)

            if(mode == 'test'):
                path_cvc = file_root + 'test/cvc/model_%r/epoch_%s__cvc_id_%r__show_id_%r.ply'%(model_num, str(epoch_num).zfill(3), idx, show_i)
                path_s = file_root + 'test/surface/model_%r/epoch_%s__cvc_id_%r__show_id_%r.ply'%(model_num, str(epoch_num).zfill(3), idx, show_i)
                path_o = file_root + 'test/output/model_%r/epoch_%s__cvc_id_%r__show_id_%r.ply'%(model_num, str(epoch_num).zfill(3), idx, show_i)

                #path_out = file_root + '/surface'
                self.d2s.save2ply(path_cvc, xyz_total, c_total)
                self.d2s.save2ply(path_s, xyz_s)
                self.d2s.save2ply(path_o, xyz, o_dense)
        print('save ply result successful')
        
        
    def draw_sample(self, 
                    data, 
                    output, 
                    output_old = None,
                    cvc_num = 3,
                    show_num = 3,
                    save_image = False, 
                    use_fg = False,
                    train_model = 'none',
                    file_root = 'labData/',
                    mode = 'train',
                    epoch_num = 10000,
                    model_num = 10000,
                    detach = False):
        
        if(cvc_num > self.cubic_num):
            cvc_num = self.cubic_num
        
        
        if(detach):
            cvc = data['cvc'].detach().numpy()
            embedding = data['embedding'].detach().numpy()
            surface = data['surface'].detach().numpy()
        else:
            cvc = data['cvc'].numpy()
            embedding = data['embedding'].numpy()
            surface = data['surface'].numpy()
        
        if(use_fg):
            output_old = output_old.detach().numpy()
            
        output = output.cpu().detach().numpy()
        output_surf = threholdSurf(output,threshold = 0.65)
        
        if(self._datasetName != 'MVS2d'):
            cvc = cvc[...,0]
            surface = surface[...,0]
            output = output[...,0]
            output_surf = output_surf[...,0]
            if(use_fg):
                output_old = output_old[...,0]
        
        if(train_model == 'attSurface'):
            #key = attSurface.z_1.cpu().detach().numpy()
            #x1 = attSurface.x_1.cpu().detach().numpy()
            
            for i in range(show_num):
                surface_one = surface[i,0]
                show_length = 2
                fig, axes = plt.subplots(ncols = (cvc_num * show_length + 3))

                for cvc_i in range(cvc_num):
                    cvc_temp = cvc[i,cvc_i].transpose((1, 2, 0))
                    axes[cvc_i * show_length].imshow(cvc_temp[...,0:3])
                    axes[1 + cvc_i * show_length].imshow(cvc_temp[...,3:])
                    #axes[2 + cvc_i * show_length].imshow(cvc_temp[...,0:3]-cvc_temp[...,3:])
                   # axes[0 + cvc_i * show_length].imshow(key[i].transpose((1, 2, 0)))
                    #axes[1 + cvc_i * show_length].imshow(x1[i].transpose((1, 2, 0)))
                    #axes[2 + cvc_i * show_length].imshow(key[i].transpose((1, 2, 0))-x1[i].transpose((1, 2, 0)))
                    #bbq = 1.0 * (((key[i].transpose((1, 2, 0))-x1[i].transpose((1, 2, 0))).sum(-1)>-0.1) * ((key[i].transpose((1, 2, 0))-x1[i].transpose((1, 2, 0))).sum(-1)<0.1))
                    #axes[3 + cvc_i * show_length].imshow(bbq,cmap = plt.cm.gray)

                axes[cvc_num * show_length].imshow(surface_one, cmap = plt.cm.gray)
                axes[cvc_num * show_length + 1].imshow(output[i,0], cmap = plt.cm.gray)
                axes[cvc_num * show_length + 2].imshow(output_surf[i,0], cmap = plt.cm.gray)

                #scipy.misc.imsave(file_root + 'model_%r_1.jpg'%i, cvc_0[...,0:3])
                #scipy.misc.imsave(file_root + 'model_%r_2.jpg'%i, cvc_0[...,3:])
                #scipy.misc.imsave(file_root + 'model_%r_3.jpg'%i, surface_one)
                #scipy.misc.imsave(file_root + 'model_%r_4.jpg'%i, output[i,0])
                plt.show()
        else:
            for i in range(show_num):
                idx = data['idx_validCubes'][i].item()
                
                surface_one = surface[i,0]
                show_length = 2
                fig, axes = plt.subplots(ncols = (cvc_num * show_length + 3))
                
                
                for cvc_i in range(cvc_num):
                    cvc_temp = cvc[i,cvc_i].transpose((1, 2, 0))
                    axes[cvc_i * show_length].imshow(cvc_temp[...,0:3])
                    axes[1 + cvc_i * show_length].imshow(cvc_temp[...,3:])
                    #axes[2 + cvc_i * show_length].imshow(cvc_temp[...,3:]-cvc_temp[...,0:3])
                    #bbq = 1.0 * (((cvc_temp[...,3:]-cvc_temp[...,0:3]).sum(-1)>-0.1) * ((cvc_temp[...,3:]-cvc_temp[...,0:3]).sum(-1)<0.1))
                    #axes[3 + cvc_i * show_length].imshow(bbq,cmap = plt.cm.gray)
                if(use_fg):
                    axes[cvc_num * show_length].imshow(surface_one, cmap = plt.cm.gray)
                    axes[cvc_num * show_length + 1].imshow(output_old[i,0], cmap = plt.cm.gray)
                    axes[cvc_num * show_length + 2].imshow(output[i,0], cmap = plt.cm.gray)
                else:
                    axes[cvc_num * show_length].imshow(surface_one, cmap = plt.cm.gray)
                    axes[cvc_num * show_length + 1].imshow(output[i,0], cmap = plt.cm.gray)
                    axes[cvc_num * show_length + 2].imshow(output_surf[i,0], cmap = plt.cm.gray)

                #scipy.misc.imsave(file_root + 'model_%r_1.jpg'%i, cvc_0[...,0:3])
                #scipy.misc.imsave(file_root + 'model_%r_2.jpg'%i, cvc_0[...,3:])
                #scipy.misc.imsave(file_root + 'model_%r_3.jpg'%i, surface_one)
                #scipy.misc.imsave(file_root + 'model_%r_4.jpg'%i, output[i,0])
                if(mode == 'train'):
                    path_dirc = file_root + 'train/epoch_%s'%(str(epoch_num).zfill(3))
                    if not os.path.exists(path_dirc):
                        os.makedirs(path_dirc)
                    path = file_root + 'train/epoch_%s/model_%r__cvc_id_%r__show_id_%r.png'%(str(epoch_num).zfill(3), model_num, idx, i)
                    plt.savefig(path)
                    plt.show()
                else:
                    path_dirc = file_root + 'test/model_%r'%(model_num)
                    if not os.path.exists(path_dirc):
                        os.makedirs(path_dirc)
                    path = file_root + 'test/model_%r/epoch_%s__cvc_id_%r__show_id_%r.png'%(model_num, str(epoch_num).zfill(3), idx, i)
                    plt.savefig(path)
                    plt.show()


params = Params()
d2s = Dense2Sparse()


class Dense2Sparse(Params):
    def __init__(self):
        super(Dense2Sparse, self).__init__()
        pass
    
    def generate_voxelLevelWeighted_coloredCubes(self, viewPair_coloredCubes, viewPair_surf_predictions, weight4viewPair):
        """
        fuse the color based on the viewPair's colored cubes, surface predictions, and weight4viewPair

        inputs
        -----
        weight4viewPair (N_cubes, N_viewPairs): relative importance of each viewPair
        viewPair_surf_predictions (N_cubes, N_viewPairs, D,D,D): relative importance of each voxel in the same cube
        viewPair_coloredCubes (N_cubes * N_viewPairs, 6, D,D,D): rgb values from the views in the same viewPair 
            randomly select one viewPair_coloredCubes (N_cubes, N_viewPairs, 3, D,D,D), otherwise the finnal colorized cube could have up/down view bias
            or simply take average

        outputs
        ------
        new_coloredCubes: (N_cubes, 3, D,D,D)

        notes
        ------
        The fusion idea is like this: 
            weight4viewPair * viewPair_surf_predictions = voxel_weight (N_cubes, N_viewPairs, D,D,D) generate relative importance of voxels in all the viewPairs
            weighted_sum(randSelect_coloredCubes, normalized_voxel_weight) = new_colored_cubes (N_cubes, 3, D,D,D)
        """
        N_cubes, N_viewPairs, _D = viewPair_surf_predictions.shape[:3]
        # (N_cubes, N_viewPairs,1,1,1) * (N_cubes, N_viewPairs, D,D,D) ==> (N_cubes, N_viewPairs, D,D,D)
        voxel_weight = weight4viewPair[...,None,None,None] * viewPair_surf_predictions
        voxel_weight /= np.sum(voxel_weight, axis=1, keepdims=True) # normalization along different view pairs

        # take average of the view0/1
        # (N_cubes, N_viewPairs, 2, 3, D,D,D) ==> (N_cubes, N_viewPairs, 3, D,D,D) 
        mean_viewPair_coloredCubes = np.mean(viewPair_coloredCubes.astype(np.float32).reshape((N_cubes, N_viewPairs, 2, 3, _D,-1,_D)), axis=2)

        # sum[(N_cubes, N_viewPairs, 1, D,D,D) * (N_cubes, N_viewPairs, 3, D,D,D), axis=1] ==>(N_cubes, 3, D,D,D)
        new_coloredCubes = np.sum(voxel_weight[:,:,None,...] * mean_viewPair_coloredCubes, axis=1)

        return new_coloredCubes.astype(np.uint8)
    
    
    def dense2sparse(self, prediction, rgb, param, viewPair, min_prob = 0.5, rayPool_thresh = 0, \
            enable_centerCrop = False, cube_Dcenter = None, \
            enable_rayPooling = False, cameraPOs = None, cameraTs = None):
        """
        convert dense prediction / rgb to sparse representation
        using rayPooling & prob_thresholding & center crop

        Note:
            rayPooling: threshold of max_votes = rayPool_thresh 
            after center crop: the min_xyz should be shifted to the new position

        --------------
        inputs:
            prediction: np.float16(N_cubes,D,D,D)
            rgb: np.uint8(N_cubes,D,D,D,3)
            param: np.float32(N_cubes, N_params): 'ijk'/'xyz'/'resol'
            viewPair: np.uint16(N_cubes, N_viewPairs, 2)
            min_prob = 0.5

            enable_centerCrop = False # used for center crop
            cube_Dcenter = None

            enable_rayPooling = False # used for rayPooling
            cameraPOs = None
            cameraTs = None
        ---------------
        outputs:
            nonempty_cube_indx: np.uint32 (N_nonempty_cubes,)
            vxl_ijk_list[i]: np.uint8 (iN_voxels, 3)
            prediction_list[i]: np.float16 (iN_voxels,) # nth element corresponds to cube_n
            rgb_list[i]: np.uint8 (iN_voxels, 3)
            rayPooling_votes_list[i]: np.uint8 (iN_voxels,)
            param_new: np.float32(N_nonempty_cubes, 4): after center crop
        """

        N_cubes, D_orig, _, _ = prediction.shape # [:2]
        nonempty_cube_indx, vxl_ijk_list, prediction_list, rgb_list, rayPooling_votes_list =\
                [], [], [], [], []
        param_new = np.copy(param)

        slc = np.s_[:,:,:] # select all in first 3 dims
        if enable_centerCrop:
            _Cmin, _Cmax = (D_orig-cube_Dcenter)/2, (D_orig-cube_Dcenter)/2 + cube_Dcenter 
            # np.s_[_Cmin:_Cmax,_Cmin:_Cmax,_Cmin:_Cmax]
            slc = (slice(_Cmin, _Cmax, 1),)*3 # np.s_[1:6] = slice(1,6)
            # shift the min_xyz of the center_cropped cubes
            param_new['xyz'] += param_new['resol'][:, None] * _Cmin # (N_cubes, 3) + (N_cubes,1) = (N_cubes, 3)

        for _n in range(N_cubes):
            if enable_rayPooling:
                # rayPooling function has already done the prob_thresholding
                #print(prediction[_n].shape)
                rayPool_votes = rayPooling.rayPooling_1cube_numpy(cameraPOs, cameraTs, \
                        viewPair_viewIndx = viewPair[_n], xyz = param[_n]['xyz'], resol = param[_n]['resol'],\
                        cube_prediction = prediction[_n], prediction_thresh = min_prob).astype(np.uint8)
                # 2n view pairs, only reserve the voxel with raypooling votes >= n
                vxl_ijk_tuple = np.where(rayPool_votes[slc] >= rayPool_thresh) 
            if (not enable_rayPooling) or rayPool_thresh == 0: # only filter out voxels with low prob
                vxl_ijk_tuple = np.where(prediction[_n][slc] > min_prob)
            if vxl_ijk_tuple[0].size == 0:
                continue # empty cube

            nonempty_cube_indx.append(_n)
            vxl_ijk_list.append(np.c_[vxl_ijk_tuple].astype(np.uint8)) # (iN_vxl,3)
            prediction_list.append(prediction[_n][slc][vxl_ijk_tuple].astype(np.float16)) # (D,D,D)-->(iN_vxl,)
            rgb_list.append(rgb[_n][slc][vxl_ijk_tuple].astype(np.uint8)) # (D,D,D,3)-->(iN_vxl,3)
            if enable_rayPooling:
                rayPooling_votes_list.append(rayPool_votes[slc][vxl_ijk_tuple].astype(np.uint8)) # (cube_Dcenter,)*3 --> (iN_voxel,)

        return nonempty_cube_indx, vxl_ijk_list, prediction_list, rgb_list, rayPooling_votes_list, param_new




    def append_dense_2sparseList(self, prediction_sub, rgb_sub, param_sub, viewPair_sub, min_prob = 0.5, rayPool_thresh = 0, 
            enable_centerCrop = False, cube_Dcenter = None, 
            enable_rayPooling = False, cameraPOs = None, cameraTs = None, 
            prediction_list = [], rgb_list = [], vxl_ijk_list = [], rayPooling_votes_list = [], 
            cube_ijk_np = None, param_np = None, viewPair_np = None):
        """
        append the sparse lists/nps results to empty or non-empty lists/nps.

        --------------
        inputs:
            prediction_sub: np.float16(N_cubes,1,D,D,D)/(N_cubes,D,D,D)
            rgb_sub: np.uint8(N_cubes,3,D,D,D)
            param_sub: np.float32(N_cubes, N_params): 'ijk'/'xyz'/'resol'
            viewPair_sub: np.uint16(N_cubes, N_viewPairs, 2)
            min_prob = 0.5

            enable_centerCrop = False # used for center crop
            cube_Dcenter = None

            enable_rayPooling = False # used for rayPooling
            cameraPOs = None
            cameraTs = None

            prediction_list, rgb_list, vxl_ijk_list, rayPooling_votes_list: orignal lists before append
            cube_ijk_np, param_np, viewPair_np: orignal np before append

        --------------
        outputs:
            prediction_list, rgb_list, vxl_ijk_list, rayPooling_votes_list: updated lists after append
            cube_ijk_np, param_np, viewPair_np: updated np after append
        """

        if prediction_sub.ndim == 5:
            prediction_sub = prediction_sub.astype(np.float16)[:,0]  # (N,1,D,D,D)-->(N,D,D,D)
        rgb_sub = np.transpose(rgb_sub.astype(np.uint8), axes=(0,2,3,4,1)) #{N,3,D,D,D} --> {N,D,D,D,3}
        # finnally, only the xyz/resol/modelIndx will be stored. In case the entire param_sub will be saved in memory, we deep copy it.
        cube_ijk_sub = param_sub['ijk']
        viewPair_sub = viewPair_sub.astype(np.uint16) # (N,N_viewPairs,2)
        sparse_output = self.dense2sparse(prediction = prediction_sub, rgb = rgb_sub, param = param_sub,\
                viewPair = viewPair_sub, min_prob = min_prob, rayPool_thresh = rayPool_thresh,\
                enable_centerCrop = enable_centerCrop, cube_Dcenter = cube_Dcenter,\
                enable_rayPooling = enable_rayPooling, cameraPOs = cameraPOs, cameraTs = cameraTs)
        nonempty_cube_indx_sub, vxl_ijk_sub_list, prediction_sub_list, \
                rgb_sub_list, rayPooling_sub_votes_list, param_new_sub = sparse_output
        param_sub = param_new_sub[nonempty_cube_indx_sub]
        viewPair_sub = viewPair_sub[nonempty_cube_indx_sub]
        cube_ijk_sub = cube_ijk_sub[nonempty_cube_indx_sub]
        if not len(prediction_sub_list) == len(rgb_sub_list) == len(vxl_ijk_sub_list) == \
                param_sub.shape[0] == viewPair_sub.shape[0] == cube_ijk_sub.shape[0]:
            raise Warning('load dense data, # of cubes is not consistent.')
        prediction_list.extend(prediction_sub_list)
        rgb_list.extend(rgb_sub_list)
        vxl_ijk_list.extend(vxl_ijk_sub_list)
        rayPooling_votes_list.extend(rayPooling_sub_votes_list)
        param_np = param_sub if param_np is None else np.concatenate([param_np, param_sub], axis=0)  # np append / concatenate
        viewPair_np = viewPair_sub if viewPair_np is None else np.vstack([viewPair_np, viewPair_sub])
        cube_ijk_np = cube_ijk_sub if cube_ijk_np is None else np.vstack([cube_ijk_np, cube_ijk_sub])

        return prediction_list, rgb_list, vxl_ijk_list, rayPooling_votes_list, \
                cube_ijk_np, param_np, viewPair_np




    def load_dense_as_sparse(self,files, cube_Dcenter, cameraPOs, min_prob=0.5, rayPool_thresh = 0):
        """
        load multiple dense cube voxels as sparse voxels data

        only reserve the voxels with prediction prob < min_prob
        --------------
        inputs:
            files: file names
            min_prob: 0.5
        --------------
        outputs:
            prediction_list[i]: np.float16 (iN_voxels,) # nth element corresponds to cube_n
            rgb_list[i]: np.uint8 (iN_voxels, 3)
            vxl_ijk_list[i]: np.uint8 (iN_voxels, 3)
            rayPooling_votes_list

            cube_ijk_np: np.uint16 (N,3)
            param_np: np.float32 (N,N_param)
            viewPair_np: np.uint16 (N,N_viewPairs,2)
        """
        prediction_list, rgb_list, vxl_ijk_list, rayPooling_votes_list = [], [], [], []
        cube_ijk_np, param_np, viewPair_np = None, None, None

        cameraT_folder = '/home/mengqi/dataset/MVS/cameraT/'
        cameraPO_folder = '/home/mengqi/dataset/MVS/pos/'

        # TODO: the new load_selected_POs hide the view index
        # cameraPOs = camera.load_selected_cameraPO_files_f64(dataset_name=param_volum.__datasetName, view_list=param_volum.__view_set)
        # cameraPOs = prepare_data.load_cameraPos_as_np(cameraPO_folder)
        cameraTs = camera.cameraPs2Ts(cameraPOs)


        for file_name in files: 
            print file_name
            try:
                with open(file_name) as f:
                    npz_file = np.load(f)
                    """
                    prediction_sub: {N,1,D,D,D} float16 --> {N,D,D,D}
                    rgb_sub: {N,3,D,D,D} uint8 --> {N,D,D,D,3}
                    param_sub: {N,8} float64 # x,y,z,resol,modelIndx,indx_d0,indx_d1,indx_d2
                    selected_viewPair_viewIndx_sub: {N, No_viewPairs, 2}
                    """
                    prediction_sub, rgb_sub, param_sub, viewPair_sub = \
                            npz_file["prediction"], npz_file["rgb"], npz_file["param"], npz_file["selected_pairIndx"] 
                prediction_list, rgb_list, vxl_ijk_list, rayPooling_votes_list, \
                        cube_ijk_np, param_np, viewPair_np = \
                        append_dense_2sparseList(prediction_sub = prediction_sub, rgb_sub = rgb_sub, param_sub = param_sub,\
                                viewPair_sub = viewPair_sub, min_prob = min_prob, rayPool_thresh = rayPool_thresh,\
                                enable_centerCrop = True, cube_Dcenter = cube_Dcenter,\
                                enable_rayPooling = True, cameraPOs = cameraPOs, cameraTs = cameraTs, \
                                prediction_list = prediction_list, rgb_list = rgb_list, vxl_ijk_list = vxl_ijk_list, \
                                rayPooling_votes_list = rayPooling_votes_list, \
                                cube_ijk_np = cube_ijk_np, param_np = param_np, viewPair_np = viewPair_np)
            except:
                print('Warning: this file not exist / valid')
        return prediction_list, rgb_list, vxl_ijk_list, rayPooling_votes_list, \
                cube_ijk_np, param_np, viewPair_np

    def filter_voxels(self, vxl_mask_list=[],prediction_list=None, prob_thresh=None,\
            rayPooling_votes_list=None, rayPool_thresh=None):
        """
        thresholding using the prediction or rayPooling 
        consider the given vxl_mask_list

        ---------
        inputs:
            prediction_list[i]: np.float16 (iN_voxels,) # nth element corresponds to cube_n 
            prob_thresh: np.float16 scalar / list
            rayPooling_votes_list[i]: np.uint8 (iN_voxels,)
            rayPool_thresh: np.uint8, scalar
            vxl_mask_list[i]: np.bool (iN_voxels,)
        ---------
        outputs:
            vxl_mask_list[i]: np.bool (iN_voxels,)
        """
        empty_vxl_mask = True if len(vxl_mask_list) == 0 else False
        if prediction_list is not None:
            if prob_thresh is None:
                raise Warning('prob_thresh should not be None.')
            for _c, _prediction in enumerate(prediction_list):
                _prob_thresh = prob_thresh[_c] if isinstance(prob_thresh, list) else prob_thresh
                _surf = _prediction >= _prob_thresh
                if empty_vxl_mask:
                    vxl_mask_list.append(_surf)
                else:
                    vxl_mask_list[_c] &= _surf
        empty_vxl_mask = True if len(vxl_mask_list) == 0 else False
        if rayPooling_votes_list is not None:
            if rayPool_thresh is None:
                raise Warning('rayPool_thresh should not be None.')
            for _cube, _votes in enumerate(rayPooling_votes_list):
                _surf = _votes >= rayPool_thresh
                if empty_vxl_mask:
                    vxl_mask_list.append(_surf)
                else:
                    vxl_mask_list[_cube] &= _surf
        return vxl_mask_list


    def save2ply(self, ply_filePath, xyz_np, rgb_np = None, normal_np = None):
        """
        save data to ply file, xyz (rgb, normal)

        ---------
        inputs:
            xyz_np: (N_voxels, 3)
            rgb_np: None / (N_voxels, 3)
            normal_np: None / (N_voxels, 3)

            ply_filePath: 'xxx.ply'
        outputs:
            save to .ply file
        """
        N_voxels = xyz_np.shape[0]
        atributes = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
        if normal_np is not None:
            atributes += [('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4')]
        if rgb_np is not None:
            atributes += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        saved_pts = np.zeros(shape=(N_voxels,), dtype=np.dtype(atributes))

        saved_pts['x'], saved_pts['y'], saved_pts['z'] = xyz_np[:,0], xyz_np[:,1], xyz_np[:,2] 
        if rgb_np is not None:
            #print('saveed', saved_pts)
            saved_pts['red'], saved_pts['green'], saved_pts['blue'] = rgb_np[:,0], rgb_np[:,1], rgb_np[:,2]
        if normal_np is not None:
            saved_pts['nx'], saved_pts['ny'], saved_pts['nz'] = normal_np[:,0], normal_np[:,1], normal_np[:,2] 

        el_vertex = PlyElement.describe(saved_pts, 'vertex')
        outputFolder = os.path.dirname(ply_filePath)
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        PlyData([el_vertex]).write(ply_filePath)
        #print('saved ply file: {}'.format(ply_filePath))
        return 1



    def save_sparseCubes_2ply(self, vxl_mask_list, vxl_ijk_list, rgb_list, \
            param, ply_filePath, normal_list=None):
        """
        save sparse cube to ply file

        ---------
        inputs:
            vxl_mask_list[i]: np.bool (iN_voxels,)
            vxl_ijk_list[i]: np.uint8 (iN_voxels, 3)
            rgb_list[i]: np.uint8 (iN_voxels, 3)
            normal_list[i]: np.float16 (iN_voxels, 3)

            param: np.float32(N_nonempty_cubes, 4)
            ply_filePath: 'xxx.ply'
        outputs:
            save to .ply file
        """
        vxl_mask_np = np.concatenate(vxl_mask_list, axis=0) 
        N_voxels = vxl_mask_np.sum()
        vxl_ijk_np = np.vstack(vxl_ijk_list)
        rgb_np = np.vstack(rgb_list)
        if not vxl_mask_np.shape[0] == vxl_ijk_np.shape[0] == rgb_np.shape[0]:
            raise Warning('make sure # of voxels in each cube are consistent.')
        if normal_list is None:
            dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), \
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
            normal_np = None
        else:
            dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), \
                    ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'), \
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
            normal_np = np.vstack(normal_list)[vxl_mask_np]
        saved_pts = np.zeros(shape=(N_voxels,), dtype=dt)

        # calculate voxels' xyz 
        xyz_list = []
        for _cube, _select in enumerate(vxl_mask_list):
            resol = param[_cube]['resol']
            xyz_list.append(vxl_ijk_list[_cube][_select] * resol + param[_cube]['xyz'][None,:]) # (iN, 3) + (1, 3)
        xyz_np = np.vstack(xyz_list)
        rgb_np = rgb_np[vxl_mask_np]
        #print('>>>>>',xyz_np, rgb_np, normal_np)
        if(self._datasetName != 'MVS2d'):
            self.save2ply(ply_filePath, xyz_np, rgb_np, normal_np)
        return xyz_np, rgb_np, normal_np



    def save_sparseCubes(self, filePath, \
            prediction_list, rgb_list, vxl_ijk_list, rayPooling_votes_list, \
            cube_ijk_np, param_np, viewPair_np):
        """
        save sparse cube voxels using numpy!

        --------------
        inputs:
            prediction_list[i]: np.float16 (iN_voxels,) # nth element corresponds to cube_n
            rgb_list[i]: np.uint8 (iN_voxels, 3)
            vxl_ijk_list[i]: np.uint8 (iN_voxels, 3)
            rayPooling_votes_list[i]: np.uint8 (iN_voxels,)

            cube_ijk_np: np.uint16 (N,3)
            param_np: np.float32 (N,N_param)
            viewPair_np: np.uint16 (N,N_viewPairs,2)    
        --------------
        outputs:
        """
        prediction_np = np.concatenate(prediction_list, axis=0)
        rgb_np = np.vstack(rgb_list)
        vxl_ijk_np = np.vstack(vxl_ijk_list)
        rayPooling_votes_np = np.empty((0,), np.uint8) if len(rayPooling_votes_list) == 0 else \
                np.concatenate(rayPooling_votes_list, axis=0) 

        N_cube = cube_ijk_np.shape[0]
        # cube_1st_vxlIndx_np: record the start voxel index of ith cube in the (i+1)th position, in order to recover the lists.
        cube_1st_vxlIndx_np = np.zeros((N_cube+1,)).astype(np.uint32)      
        for _n_cube, _prediction in enumerate(prediction_list):
            cube_1st_vxlIndx_np[_n_cube + 1] = _prediction.size + cube_1st_vxlIndx_np[_n_cube] 
        if not cube_1st_vxlIndx_np[-1] == prediction_np.shape[0] == rgb_np.shape[0] == vxl_ijk_np.shape[0]:
            raise Warning("# of voxels is not consistent while saving sparseCubes.")
        with open(filePath, 'wb') as f:
            np.savez_compressed(f, cube_1st_vxlIndx_np = cube_1st_vxlIndx_np, prediction_np = prediction_np, \
                    rgb_np = rgb_np, vxl_ijk_np = vxl_ijk_np, rayPooling_votes_np = rayPooling_votes_np, \
                    cube_ijk_np = cube_ijk_np, param_np = param_np, viewPair_np = viewPair_np)
            print("saved sparseCubes to file: {}".format(filePath))


    def load_sparseCubes(self,filePath):
        """
        load sparse cube voxels from saved numpy npz!

        --------------
        outputs:
            prediction_list[i]: np.float16 (iN_voxels,) # nth element corresponds to cube_n
            rgb_list[i]: np.uint8 (iN_voxels, 3)
            vxl_ijk_list[i]: np.uint8 (iN_voxels, 3)
            rayPooling_votes_list[i]: np.uint8 (iN_voxels,)

            cube_ijk_np: np.uint16 (N,3)
            param_np: np.float32 (N,N_param)
            viewPair_np: np.uint16 (N,N_viewPairs,2)    
        """
        with open(filePath) as f:
            npz = np.load(f)
            cube_1st_vxlIndx_np, prediction_np, rgb_np, vxl_ijk_np, rayPooling_votes_np, cube_ijk_np, param_np, viewPair_np = \
                    npz['cube_1st_vxlIndx_np'], npz['prediction_np'], npz['rgb_np'], npz['vxl_ijk_np'], npz['rayPooling_votes_np'], \
                    npz['cube_ijk_np'], npz['param_np'], npz['viewPair_np']
            print("loaded sparseCubes to file: {}".format(filePath))

        if not cube_1st_vxlIndx_np[-1] == prediction_np.shape[0] == rgb_np.shape[0] == vxl_ijk_np.shape[0]:
            raise Warning("# of voxels is not consistent while saving sparseCubes.")
        if not rayPooling_votes_np.shape[0] in [0, cube_1st_vxlIndx_np[-1]]:
            raise Warning("rayPooling_votes_np.shape[0] != 0 / # of voxels.")

        prediction_list, rgb_list, vxl_ijk_list, rayPooling_votes_list = [], [], [], []
        N_cube = cube_ijk_np.shape[0]
        for _n_cube in range(N_cube):
            slc = np.s_[cube_1st_vxlIndx_np[_n_cube]: cube_1st_vxlIndx_np[_n_cube + 1]]
            prediction_list.append(prediction_np[slc])
            rgb_list.append(rgb_np[slc])
            vxl_ijk_list.append(vxl_ijk_np[slc])
            rayPooling_votes_list.append(rayPooling_votes_np[slc])
        return prediction_list, rgb_list, vxl_ijk_list, rayPooling_votes_list, \
            cube_ijk_np, param_np, viewPair_np


class Reconstruct(Params):
    def __init__(self):
        super(Reconstruct, self).__init__()
        pass
    #def reconstruct_resol_list(self,)
    
    def reconstruct(self,
                    test_dataset = None,
                    model_num = 340, resol_list = [1.0],
                    batch_size = 4, min_prob = 0.5, 
                    tau = 0.7, gamma = 0.8,
                    fg_tau = 1.0,
                    attention_type = 'none',
                    train_model = 'attSurface',
                    use_fg = False,
                    mode = 'init_volumn',
                    thre_angle = 150,
                    type_volumn = 'test',
                    train_rejection = 'none',
                    selection_method = 'random',
                    enable_rayPooling = False,
                    model_path = './experiment/network/DTU/augment/040103/',
                    model_name = 'epoch10/',
                    draw = False):
        self.xyzs = None
        self.rgbs = None
        
        for resol in resol_list:
            
            if(test_dataset is None):
                test_dataset = Dataset(model_num = model_num, 
                                       mode = mode,
                                       thre_angle = thre_angle,
                                       type_volumn = type_volumn,
                                       train_rejection = train_rejection,
                                       selection_method = selection_method,
                                       resol = resol)
            dense2sparse = Dense2Sparse()


            data_test = DataLoader(test_dataset, batch_size= batch_size,
                                        shuffle=False, num_workers=12)
            print('dataset prepare complete')

            prediction_list, rgb_list, vxl_ijk_list, rayPooling_votes_list = [], [], [], []
            cube_ijk_np, param_np, viewPair_np = None, None, None

            for i, data in enumerate(data_test,0):
                #if(i < 258):
                #    continue
                print('start evaluate %r epoch'%i)
                self.data_o = data
                if(self._datasetName == 'MVS2d'):
                    batch_size_cvc,cubic_num,_,_,pixel_size = data['cvc'].size()
                    w_list = torch.zeros([batch_size_cvc, cubic_num]).type(torch.cuda.FloatTensor)
                    s_list = torch.zeros([batch_size_cvc, cubic_num, 1,pixel_size,pixel_size ]).type(torch.cuda.FloatTensor)
                else:
                    batch_size_cvc,cubic_num,_,_,_,pixel_size = data['cvc'].size()
                    
                    #w_list = torch.zeros([batch_size_cvc, cubic_num])
                    w_list = torch.ones([batch_size_cvc, cubic_num])
                    s_list = torch.zeros([batch_size_cvc, cubic_num, pixel_size,pixel_size,pixel_size ])

                if(train_model == 'attSurface'):
                    output = attSurface(data['cvc'].to(device))
                else:
                    output = 0
                    w_total = 0
                    
                    for i_c in range(cubic_num):
                    #for i_c in range(1):
                        if(self._datasetName == 'MVS2d'):
                            s = surfaceNet(data['cvc'][:,i_c,...].to(device)).detach()
                        else:
                            s = surfaceNet(data['cvc'][:,i_c,...].to(device))[:,0].detach()
                            #if(use_fg):
                            #    s = fg(s)[:,0].detach()
                        
                        if(attention_type == 'pixel'):
                            w = eNet(surfaceNet.s)[:,0].detach()
                            w_total += w
                            output += s * w
                            
                            #w_list[:,i_c] = w.sum((1,2,3)).cpu()
                            s_list[:,i_c] = output.cpu()
                            #print(w)
                            
                        else:
                            w = eNet(data['embedding'][:,i_c,...].to(device)).detach()
                        
                            w_list[:,i_c] = w[:,0].cpu()
                            s_list[:,i_c] = s.cpu()

                            w_total += w[...,None,None]
                            output += s * w[...,None,None]
                            #print('s',(s>0.1).sum())
                            #print(w_total.shape)
                            #output += s
                        
                    output = output/(w_total + 1e-15)
                    #output = output/cubic_num
                    
                if(use_fg):
                    print('before fg',(output>0.4).sum())
                    output = fg(output)
                    print('after fg',(output>0.4).sum())
                    #output = output.pow(fg_tau) / (output.pow(fg_tau) + (1-output).pow(fg_tau))
                    
                if(self._datasetName == 'MVS2d'):
                    output_numpy = output.detach().cpu().numpy()[:,:,:,None,:]
                    #output_numpy = np.swapaxes(output_numpy, 2, -1)

                    w_list_numpy = w_list.detach().cpu().numpy()
                    s_list_numpy = s_list.detach().cpu().numpy()[:,:,0,:,None,:]
                    #s_list_numpy = np.swapaxes(s_list_numpy, 2, -1)

                    cube_numpy = ((data['cvc'])).detach().cpu().numpy()[:,:,:,:,None,:]
                else:
                    output_numpy = output.detach().cpu().numpy()
                    #output_numpy = np.swapaxes(output_numpy, 2, -1)

                    w_list_numpy = w_list.detach().cpu().numpy()
                    s_list_numpy = s_list.detach().cpu().numpy()
                    #s_list_numpy = np.swapaxes(s_list_numpy, 2, -1)
                    #print(w_list_numpy)

                    cube_numpy = ((data['cvc'])).detach().cpu().numpy()
                    
                cube_numpy = 256 * (cube_numpy + 0.5)
                #cube_numpy = np.swapaxes(cube_numpy, 3, -1)
                new_cube = dense2sparse.generate_voxelLevelWeighted_coloredCubes(
                    viewPair_coloredCubes = cube_numpy, 
                    viewPair_surf_predictions = s_list_numpy, 
                    weight4viewPair = w_list_numpy
                )

                if(draw):
                    train_iter.draw_sample(data, output, show_num = 1)

                if(self._datasetName == 'MVS2d'):
                    output_numpy = np.repeat(output_numpy, 32, axis = 3)
                    new_cube = np.repeat(new_cube, 32, axis = 3)
                
                
                if(batch_size_cvc == 1):
                    idx_validCubes = np.array([test_dataset.volumn.cubes_param_np[data['idx_validCubes']]])
                    idx = test_dataset.volumn.viewPairs4Reconstr[data['idx']][None,...]
                else:
                    idx_validCubes = test_dataset.volumn.cubes_param_np[data['idx_validCubes']]
                    idx = test_dataset.volumn.viewPairs4Reconstr[data['idx']]
                
                updated_sparse_list_np = dense2sparse.append_dense_2sparseList(
                            prediction_sub = output_numpy, 
                            rgb_sub = new_cube, 
                            param_sub = idx_validCubes,                                        
                            viewPair_sub = idx, 
                            min_prob = min_prob, rayPool_thresh = 0,
                            enable_centerCrop = True, 
                            cube_Dcenter = params._cube_Dcenter,
                            enable_rayPooling = enable_rayPooling, 
                            cameraPOs = test_dataset.cameraPOs_np, 
                            cameraTs = test_dataset.cameraTs_np, 
                            prediction_list = prediction_list, 
                            rgb_list = rgb_list, vxl_ijk_list = vxl_ijk_list, \
                            rayPooling_votes_list = rayPooling_votes_list, \
                            cube_ijk_np = cube_ijk_np, 
                            param_np = param_np, 
                            viewPair_np = viewPair_np)

                prediction_list, rgb_list, vxl_ijk_list, rayPooling_votes_list, cube_ijk_np, param_np, viewPair_np = updated_sparse_list_np

            print('end surfaceNet predict')
            vxl_mask_list = dense2sparse.filter_voxels(
                vxl_mask_list=[],
                prediction_list=prediction_list, 
                prob_thresh= tau,
                rayPooling_votes_list=rayPooling_votes_list, 
                rayPool_thresh = gamma * self.N_viewPairs4inference[0] * 2) 

            vxl_maskDenoised_list = denoising.denoise_crossCubes(cube_ijk_np, 
                                                                 vxl_ijk_list, 
                                                                 vxl_mask_list = vxl_mask_list, 
                                                                 D_cube = self._cube_D)
            if(enable_rayPooling):
                reconstruct_name = 'model:%s__ray:%s_tau:%s_gamma:%s__reject:%s_select:%s__angle:%s_resol:%s.ply'%(str(model_num),str(enable_rayPooling),str(tau), str(gamma), str(train_rejection), str(selection_method),str(thre_angle), str(resol))
            else:
                reconstruct_name = 'model:%s__ray:%s_min_prob:%s__reject:%s_select:%s__angle:%s_resol:%s.ply'%(str(model_num),str(enable_rayPooling),str(min_prob),str(train_rejection), str(selection_method), str(thre_angle),str(resol))
            
            if not os.path.exists(model_path + model_name):
                os.makedirs(model_path + model_name)
            
            ply_filePath = model_path + model_name + reconstruct_name
            
            #pdb.set_trace()
            self.xyz_np, self.rgb_np, self.normal_np = dense2sparse.save_sparseCubes_2ply(vxl_maskDenoised_list, vxl_ijk_list, rgb_list, \
                        param_np, ply_filePath=ply_filePath, normal_list=None)
            if(self.xyzs is None):
                self.xyzs = self.xyz_np
                self.rgbs = self.rgb_np
            else:
                self.xyzs = np.concatenate((self.xyzs,self.xyz_np), axis = 0)
                self.rgbs = np.concatenate((self.rgbs,self.rgb_np), axis = 0)
        
        
        
        
        
        
        return self.xyzs, self.rgbs
    

    def test(self):
        dense2sparse = Dense2Sparse()
        
        vxl_mask_list = dense2sparse.filter_voxels(
            vxl_mask_list=[],
            prediction_list=prediction_list, 
            prob_thresh= self._tau,
            rayPooling_votes_list=rayPooling_votes_list, 
            rayPool_thresh = self._gamma * self.N_viewPairs4inference[0] * 2) 

        vxl_maskDenoised_list = denoising.denoise_crossCubes(cube_ijk_np, 
                                                             vxl_ijk_list, 
                                                             vxl_mask_list = vxl_mask_list, 
                                                             D_cube = self._cube_D)
        dense2sparse.save_sparseCubes_2ply(vxl_maskDenoised_list, vxl_ijk_list, rgb_list, \
                    param_np, ply_filePath='rub/reconstruct_test.ply', normal_list=None)




if __name__ == '__main__':

	params = Params()
	d2s = Dense2Sparse()
	train_iter = TrainIter(recorder = None,d2s = d2s)

	use_cuda = True
	#torch.cuda.set_device(0)
	device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

	surfaceNet = SurfaceNet_3d_old().to(device)

	attention_type = 'pixel'
	if(attention_type == 'pixel'):
	    eNet = EmbeddingNet_3d_pixel().to(device)
	    print('pixel')
	else:
	    eNet = EmbeddingNet_3d().to(device)
	    print('none')

	fg = FineGenerator_3d_res().to(device)
	torch.cuda.empty_cache()
	print('finished network Prepair')

	epoch_name = 'epoch4'
	model_path = 'DTU/augment/040912_pixel_pretrain/'
	train_iter.load_model(model_path = './experiment/network/' + model_path,
	                   model_type = 'train',
	                   model_name1 = 'SurfaceNet_' + epoch_name, 
	                   model_name2 = 'eNet_' + epoch_name,
	                   #model_name3 = 'fgNet_' + epoch_name,
	                   train_model = 'none')
	

	
	for model in [9,10,11]:
	    for tau in [0.7,0.8]:
	        xyz_np, rgb_np = reconstruct.reconstruct(model_num = model,
	                                                 mode = 'load_volumn',
	                                                 train_rejection = 'total',
	                                                 type_volumn = 'test',
	                                                 thre_angle = 360,
	                                                 selection_method = 'random',
	                                                 resol_list = [0.8],
	                                                 attention_type = attention_type,
	                                                 enable_rayPooling = True,
	                                                 batch_size = 8,
	                                                 fg_tau = 1,
	                                                 min_prob=0.3, 
	                                                 tau = tau, gamma = 0.2,
	                                                 model_path = './experiment/output/' + model_path,
	                                                 model_name = epoch_name + '/',
	                                                 train_model = 'attSurfac',
	                                                 use_fg = False)
