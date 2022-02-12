import sys
sys.path.append("./nets")
sys.path.append("./tools")
from Surf import SurfaceNet_3d_old, EmbeddingNet_3d_new_big
from Attn import AttentionSurface, Self_Attn
from F_G import FineGenerator_3d_res
from Disc import Discriminator
import numpy as np
import itertools
import torch
import os
import random
import pdb
import scipy.io
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim

from Parameter import Params
from Prepair import Dataset, Recorder, Dense2Sparse
from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import pickle
#from train import TrainIter


def threholdSurf(data, threshold = 0.5):
    
    threshold_surf = (data >= threshold)
    threshold_surf = threshold_surf * 1.0
    return threshold_surf

class TrainIter(Params):
    def __init__(self, recorder = None, d2s = None):
        super(TrainIter, self).__init__()
        #self.recorder = recorder
        self.init_records()
        self.init_data()
        #plt.clf()
        
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
        lr_gen_s = 1e-4,
        lr_gen_e = 1e-4,
        lr_gen_fg = 1e-4,
        lr_dis = 1e-5,
        weight_cliping_limit = 0.05,
        add_mass = 0.1, 
        minus_mass = 0.1, 
        interpolation = 0.1,
        use_dynamic_alpha = False,
        decay_rate = 0.95,
        alpha_decay = 1.0,
        reward_epoch = 10,
        attention_type = 'none',
        use_augment = True, 
        use_augment_fg = True,
        use_teacher = True,
        use_fg = True,
        use_fg_new = False,
        use_ad = False,
        use_fg_teacher = True,
        use_g_error = True,
        fg_detach = False,
        use_fg_transfer = True,
        gan_layer = 2,
        fg_tau = 1.0,
        layer1_tau = 1.0,
        alpha_decay_fg = 0.1,
        god_w = 1e-3,
        teacher_type = 'none',
        teacher_type_fg = 'none',
        mixed_alpha = 1.0,
        d_step = 1,
        g_step = 3,
        count_distance = False,
        training_draw_iter = 150,
        validation_iter = 5,
        save_iter = 5,
        shuffle = False,
        train_model = 'none',
        root = 'experiment/network/',
        root_board = './experiment/log/',
        root_ply = './experiment/cubes/',
        root_slice = './experiment/slice/',
        loss_type = 'gan',
        training_key = '030401',
        show = False
                  ):


            
        print('lr_gen_s:%r||lr_gen_e:%r||lr_gen_fg:%r'%(lr_gen_s, lr_gen_e, lr_gen_fg))
        
        model_path = self.model_path
        
        loss_show = 0
        loss_mse = nn.MSELoss()
        
        optimizer_s = optim.RMSprop(surfaceNet.parameters(), lr = lr_gen_s)
        optimizer_e = optim.RMSprop(eNet.parameters(), lr = lr_gen_e)
        #optimizer_d = optim.RMSprop(disNet.parameters(), lr = lr_dis)
        if(use_fg):
            optimizer_fg = optim.RMSprop(fg.parameters(), lr = lr_gen_fg)
        #if(use_ad):
        #    optimizer_ad = optim.RMSprop(ad.parameters(), lr = lr_gen_fg)
        #optimizer_fg_transfer = optim.RMSprop(fg_transfer.parameters(), lr = lr_dis)
        
        iter_record = 0
        iter_model_record = 0
        i_append = 0
        accuracy_ave = 0
        completeness_ave = 0
        f_distance_ave = 0

        root_path = root + '%s/'%(self._datasetName)
        directory = root_path + '%s/%s'%(loss_type, training_key)
        key_path = root_board + '%s/'%(self._datasetName) + '%s/%s/'%(loss_type, training_key)
        ply_path = root_ply + '%s/'%(self._datasetName) + '%s/%s/'%(loss_type, training_key)
        slice_path = root_slice + '%s/'%(self._datasetName) + '%s/%s/'%(loss_type, training_key)
        self.ply_path = ply_path
        self.slice_path = slice_path
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        t_start = time.time()
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
                #surfaceNet.train()
                model_name = '%s/%s/eNet_epoch%r'%(loss_type, training_key, epoch)
                torch.save(eNet.cpu().state_dict(), root_path + model_name)
                eNet.to(device)
                #eNet.train()
                #model_name = '%s/%s/disNet_epoch%r'%(loss_type, training_key, epoch)
                #torch.save(disNet.cpu().state_dict(), root_path + model_name)
                #disNet.to(device)
                if(use_fg):
                    model_name = '%s/%s/fgNet_epoch%r'%(loss_type, training_key, epoch)
                    torch.save(fg.cpu().state_dict(), root_path + model_name)
                    fg.to(device)
                    #fg.train()
                #if(use_ad):
                #    model_name = '%s/%s/adNet_epoch%r'%(loss_type, training_key, epoch)
                #    torch.save(ad.cpu().state_dict(), root_path + model_name)
                #    ad.to(device)
                    #ad.train()
            
            if(epoch % validation_iter == 100):
                self.validationTest(surfaceNet, eNet, model_num = self._test_modelList[0], use_fg = use_fg,fg_tau = fg_tau,  draw = True, train_model = train_model)
                self.validationTest(surfaceNet, eNet, model_num = self._test_modelList[1], use_fg = use_fg,fg_tau = fg_tau,  draw = True, train_model = train_model)
                self.validation_loss_list.append(self.loss_validation)
                #print('USE fg nono')
                #self.validationTest(surfaceNet, eNet, model_num = 344, use_fg = False, draw = True, train_model = train_model)
                #self.validationTest(surfaceNet, eNet, model_num = 325, use_fg = False, draw = True, train_model = train_model)
                
            
            for model_num in model_list:
                
                if(model_num == 51):
                    model_name = '%s/%s/SurfaceNet_epoch%r_half51'%(loss_type, training_key, epoch)
                    torch.save(surfaceNet.cpu().state_dict(), root_path + model_name)
                    surfaceNet.to(device)
                    #surfaceNet.train()
                    model_name = '%s/%s/eNet_epoch%r_half51'%(loss_type, training_key, epoch)
                    torch.save(eNet.cpu().state_dict(), root_path + model_name)
                    eNet.to(device)
                    #eNet.train()
                    if(use_fg):
                        model_name = '%s/%s/fgNet_epoch%r_half51'%(loss_type, training_key, epoch)
                        torch.save(fg.cpu().state_dict(), root_path + model_name)
                        fg.to(device)
                        #fg.train()
                    #if(use_ad):
                    #    model_name = '%s/%s/adNet_epoch%r_half51'%(loss_type, training_key, epoch)
                    #    torch.save(ad.cpu().state_dict(), root_path + model_name)
                    #    ad.to(device)
                        #ad.train()
                if(model_num == 100):
                    model_name = '%s/%s/SurfaceNet_epoch%r_half100'%(loss_type, training_key, epoch)
                    torch.save(surfaceNet.cpu().state_dict(), root_path + model_name)
                    surfaceNet.to(device)
                    #surfaceNet.train()
                    model_name = '%s/%s/eNet_epoch%r_half100'%(loss_type, training_key, epoch)
                    torch.save(eNet.cpu().state_dict(), root_path + model_name)
                    eNet.to(device)
                    #eNet.train()
                    if(use_fg):
                        model_name = '%s/%s/fgNet_epoch%r_half100'%(loss_type, training_key, epoch)
                        torch.save(fg.cpu().state_dict(), root_path + model_name)
                        fg.to(device)
                        #fg.train()
                    #if(use_ad):
                    #    model_name = '%s/%s/adNet_epoch%r_half100'%(loss_type, training_key, epoch)
                    #    torch.save(ad.cpu().state_dict(), root_path + model_name)
                    #    ad.to(device)
                        #ad.train()

                t1 = time.time()
                dataset = Dataset(model_num, eNet = eNet)
                
                dataloader = DataLoader(dataset, batch_size= batch_size,
                                        shuffle=shuffle, num_workers=16)
                
                #surfaceNet_load.load_state_dict(surfaceNet.state_dict())
                #eNet_load.load_state_dict(eNet.state_dict())
                print('load model cost:', time.time() - t1)
                t1 = time.time()

                loss_teacher_model = 0
                loss_teacher_fg_model = 0
                g_error_model = 0
                for i, data in enumerate(dataloader, 0):
                    #print('get next batch cost',time.time()-t1)
                    t1 = time.time()
                    i_append +=1   
                    if(self._datasetName == 'MVS2d'):                
                        batch_size_cvc,self.cubic_num,_,_,_ = data['cvc'].size()
                    else:
                        batch_size_cvc,self.cubic_num,_,_,_,_ = data['cvc'].size()
                    #print('get batch_size cost:',time.time() - t1)
                    output = 0
                    w_total = 0
                    output_ad = 0
                    output_fg_new = 0
                    
                    t1 = time.time()
                    #self.cubic_num = 1
                    for i_c in range(self.cubic_num):
                        if(self._datasetName == 'MVS2d'):
                            if not (use_ad):
                                s = surfaceNet(data['cvc'][:,i_c,...].to(device))
                            else:
                                s,s_ad = surfaceNet(data['cvc'][:,i_c,...].to(device))
                            #s = surfaceNet(data['cvc'][:,i_c,...].to(device))
                        else:
                            if not (use_ad):
                                s = surfaceNet(data['cvc'][:,i_c,...].to(device))[:,0]
                            else:
                                s,s_ad = surfaceNet(data['cvc'][:,i_c,...].to(device))
                                s = s[:,0]
                                s_ad = s_ad[:,0]
                            if(use_fg_new):
                                s_fg_new = fg(s)[:,0]


                        #s = surfaceNet(data['cvc'][:,i_c,...].to(device))

                        if(attention_type == 'pixel'):
                            w = eNet(surfaceNet.s)[:,0]
                            w_total += w
                            output += s * w
                            if(use_ad):
                                output_ad += s_ad * w.detach()
                        else:
                            w = eNet(data['embedding'][:,i_c,...].to(device))
                            w_total += (w[...,None,None])
                            output += s * w[...,None,None]
                            if(use_ad):
                                output_ad += s_ad * w[...,None,None].detach()
                            if(use_fg_new):
                                output_fg_new += s_fg_new * w[...,None,None]


                    
                    output = output/(w_total + 1e-5)
                    if(use_ad):
                        output_ad = output_ad/(w_total + 1e-5)
                    if(use_fg_new):
                        output_fg_new = output_fg_new/(w_total + 1e-5)
                    #print('count output_total cost:',time.time() - t1)
                    output_load = 0
                    w_total_load = 0
                    
                    #for i_c in range(self.cubic_num):
                    #    s_load = surfaceNet_load(data['cvc'][:,i_c,...].to(device)).detach()
                    #    w_load = eNet_load(data['embedding'][:,i_c,...].to(device)).detach()
                        #w = w.detach()
                        #s = s.detach()
                    #    w_total_load += w_load[...,None,None].detach()
                    #    output_load += s_load * w_load[...,None,None]
                    #output_load = output_load/(w_total_load + 1e-15)
                    
                    
                    #data_surface_old = data['surface'].to(device)
                    
                    #data_surface = self.blur_truth(data_surface = data_surface_old, output_detach = output_load, 
                    #                               interpolation = interpolation, add_mass = add_mass, minus_mass = minus_mass)
                    t1 = time.time()
                    if(self._datasetName != 'MVS2d'):
                        data_surface = data['surface'][:,0].to(device)
                    #print('load data_surface cost:',time.time() - t1)

                    #if(show):
                    #    fig, axes = plt.subplots(ncols = 3)
                    #    axes[0].imshow(data_surface[0,0], cmap = plt.cm.gray)
                    #    axes[1].imshow(data_surface_old[0,0], cmap = plt.cm.gray)
                    #    axes[2].imshow(output.detach()[0,0], cmap = plt.cm.gray)
                    #    plt.show()
                    
                    t1 = time.time()
                    if(teacher_type == 'MSE'):
                        loss_teacher = loss_mse(output, data_surface)
                    else:
                        if(use_augment):
                            if(self._datasetName == 'MVS2d'):
                                alpha = 1- data_surface.to(device).sum((1,2,3))/params._cube_D**2
                                alpha = alpha[:,None,None,None]
                            else:
                                alpha = 1- data_surface.to(device).sum((1,2,3))/params._cube_D**3
                                alpha = alpha[:,None,None,None]
                            #alpha_num = data['surface'].to(device).sum()
                            #alpha = 1- alpha_num / (params.batch_size * params._cube_D**2)

                            #alpha = 0.999
                        else:
                            alpha = 0.5


                        loss_teacher = - (alpha * torch.log(output + 1e-15) * data_surface + (1-alpha) * torch.log(1-output + 1e-15) * (1-data_surface)).mean()
                    #print('count loss_teacher cost:',time.time() - t1)


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
                        
                            if(use_fg and (gan_layer == 2)):
                                fg_output = fg(output_detach).detach()
                                
                                if(use_fg_transfer):
                                    fg_output = fg_output.pow(fg_tau)/(fg_output.pow(fg_tau) + (1-fg_output).pow(fg_tau))
                                    fg_output_transfer = fg_transfer(fg_output)
                                    d_output_fake = disNet(data['cvc'].to(device), data['embedding'].to(device),fg_output_transfer)
                                else:
                                    d_output_fake = disNet(data['cvc'].to(device), data['embedding'].to(device),fg_output)
                                
                            else:
                                if(use_fg_transfer):
                                    output_detach = output_detach.pow(layer1_tau)/(output_detach.pow(layer1_tau) + (1-output_detach).pow(layer1_tau))
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
                            
                            t1 = time.time()
                            #if not(fg_detach):
                            surfaceNet.zero_grad()
                            eNet.zero_grad()
                            #if(use_ad):
                            #    ad.zero_grad()
                            if(use_fg or use_ad):
                                if(use_fg):
                                    fg.zero_grad()
                                    if(fg_detach):
                                        output = output.detach()
                                    
                                    if(use_fg_new):
                                        output_layer1 = output.detach().cpu()
                                        output = output_fg_new
                                        output = output.pow(fg_tau) / (output.pow(fg_tau) + (1-output).pow(fg_tau))
                                    else:
                                        #output_layer1 = output
                                        #if i_append % training_draw_iter == 1:
                                        output = output.pow(fg_tau) / (output.pow(fg_tau) + (1-output).pow(fg_tau))
                                        output_layer1 = output.detach().cpu()
                                        #output_layer1 = output_layer1.pow(layer1_tau) / (output_layer1.pow(layer1_tau) + (1-output_layer1).pow(layer1_tau))
                                        output = fg(output)[:,0]
                                        #output = output.pow(fg_tau) / (output.pow(fg_tau) + (1-output).pow(fg_tau))
                                        #print('count fg_output cost:',time.time() - t1)
                                elif(use_ad):
                                    output_layer1 = output.detach().cpu()
                                    output = output_ad
                                t1 = time.time()
                                if(use_fg_teacher): 
                                    if(teacher_type_fg == 'MSE'):
                                        #loss_teacher_fg = loss_mse(output[:,0], data_surface)
                                        loss_teacher_fg = (output - data_surface).pow(2).mean()
                                        #print('output', output.shape)
                                        #print('data_surface', data_surface.shape)
                                    elif(teacher_type_fg == 'MIXED'):
                                        if(use_augment_fg):
                                            alpha = 1- data_surface.to(device).sum((1,2,3))/params._cube_D**3
                                            alpha = alpha[:,None,None,None]

                                            #alpha_num = data['surface'].to(device).sum()
                                            #alpha = 1- alpha_num / (params.batch_size * params._cube_D**2)
                                            #alpha = 0.999
                                        else:
                                            alpha = 0.5
                                        loss_teacher_fg = - (alpha * torch.log(output + 1e-15) * data_surface + (1-alpha) * torch.log(1-output + 1e-15) * (1-data_surface)).mean()
                                        #alpha_losses = loss_teacher_fg.detach() / (loss_mse(output, data_surface).detach()+1e-9)
                                        loss_teacher_fg += (output - data_surface).pow(2).mean() / mixed_alpha 
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
                                #print('count fg_loss cost:',time.time() - t1)
                            if(use_g_error):
                                if(use_fg_transfer):
                                    if((gan_layer == 1) and use_fg):
                                        output_transfer = fg_transfer(output_layer1)
                                    else:
                                        output_transfer = fg_transfer(output)
                                    fake_decision = disNet(data['cvc'].to(device), data['embedding'].to(device),output_transfer)
                                else:
                                    if((gan_layer == 1) and use_fg):
                                        fake_decision = disNet(data['cvc'].to(device), data['embedding'].to(device),output_layer1)
                                    else:
                                        fake_decision = disNet(data['cvc'].to(device), data['embedding'].to(device),output)
                                    
                                g_error = fake_decision.mean()
                            else:
                                g_error = 0

                            if(use_fg_teacher):
                                if(use_dynamic_alpha):
                                    #alpha_error = loss_teacher_fg.detach()/(g_error.detach()+1e-10)
                                    alpha_error = 1.0
                                else:
                                    alpha_error = 1.0
                                g_error = g_error * alpha_error + loss_teacher_fg / alpha_decay_fg
                                
                            if(use_dynamic_alpha):
                                alpha_error = loss_teacher.detach()/(g_error.detach()+1e-10)
                            else:
                                alpha_error = 1.0
                            
                            if(use_teacher):
                                g_error = g_error * alpha_error  + loss_teacher / alpha_decay
                            
                            t1 = time.time()
                            if(i_g < g_step - 1):
                                g_error.backward(retain_graph=True)
                            else:
                                g_error.backward()
                            #print('g_error backward cost:',time.time() - t1)
                            t1 = time.time()
                            if(use_fg):
                                optimizer_fg.step()
                            #if(use_ad):
                            #    optimizer_ad.step()
                                #print('fg step cost:',time.time() - t1)
                            #if not (fg_detach):
                            optimizer_s.step()
                            optimizer_e.step()

                            
                       
                    if(count_distance):      
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

                    if i_append % training_draw_iter == 1:
                        print('start drawing slice and saving ply')
                        t_drawing = time.time()
                        print(i_append)
                        #self.draw_sample(self.data, 
                        #                 self.output, 
                        #                 train_model = train_model,
                        #                 epoch_num = epoch, 
                        #                 mode = 'train',
                        #                 file_root = self.slice_path,
                        #                 model_num = model_num)
                        
                        if(self._datasetName != 'MVS2d'):
                            self.save_ply(data = data,
                                          output = output.detach().cpu(),
                                          output_old = output_layer1,
                                          use_fg = True,
                                          epoch_num = epoch, 
                                          mode = 'train',
                                          file_root = self.ply_path,
                                          model_num = model_num)
                        print('drawing fucking painting cost time: ', time.time()-t_drawing)
                        print('[%d/%d][%d/%d]\tTime for training and drawing:%.4fs,%.4fs'
                              % (epoch, num_epochs, i, len(dataloader),
                                 t_drawing - t_start, time.time() - t_drawing))
                        t_start = time.time()
                        #loss_show = 0
                        
                    memory_used = torch.cuda.memory_allocated()
                    max_memory = torch.cuda.max_memory_allocated()

                    #self.writer.add_scalars(key_path + 'memory', {'memory':100 * memory_used/(max_memory+1e-1)},iter_record)
                    #self.writer.add_scalars(key_path + 'training', \
                    #    {'loss_teacher':loss_teacher.detach().cpu(), \
                    #    'loss_teacher_fg':loss_teacher_fg.detach().cpu()/alpha_decay_fg, \
                    #    'g_error':g_error.detach().cpu() \
                    #    },iter_record) 

                    iter_record += 1
                    loss_teacher_model += loss_teacher.detach().cpu()
                    loss_teacher_fg_model += loss_teacher_fg.detach().cpu()/alpha_decay_fg
                    g_error_model += g_error.detach().cpu()

                    t1 = time.time()
                self.writer.add_scalars(key_path + 'training', \
                        {'loss_teacher':loss_teacher_model/(i+1), \
                        'loss_teacher_fg':loss_teacher_fg_model/(i+1), \
                        'g_error':g_error_model/(i+1) \
                        },iter_model_record) \
                    
                #self.writer.add_scalars(key_path + 'training_whole_model', {'loss_augment':loss_whole_model/(i+1)},iter_model_record)
                iter_model_record += 1
                #loss_epoch += loss_whole_model/(i+1)
                    
    def train(self,
        num_epochs = 100,
        model_list = [9],
        w_detach = False,
        w_detach_num = 10,
        batch_size = 8,
        lr_s = 0.001,
        lr_e = 1e-3,
        lr_decay = 0.97,
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
        attention_type = 'none',
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
            optimizer_s = optim.SGD(surfaceNet.parameters(), lr = lr_s)
            optimizer_e = optim.SGD(eNet.parameters(), lr = lr_e)
            
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

            #lr *= lr_decay
            if not (w_detach  and (epoch < w_detach_num)):
                optimizer_s = optim.SGD(surfaceNet.parameters(), lr = lr_s)
            optimizer_e = optim.SGD(eNet.parameters(), lr = lr_e)
            #print('learning rate', lr)

            loss_epoch = 0
            #torch.cuda.empty_cache()
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
            
            if(epoch % validation_iter == 100):
                if(epoch % draw_iter == 100):
                    #batch_size = 1
                    print('start drawing and validation')
                    t_validation = time.time()
                    self.validationTest(surfaceNet, eNet, epoch_num = epoch, model_num = self._test_modelList[0], train_model = train_model, batch_size = batch_size)
                    self.validationTest(surfaceNet, eNet, epoch_num = epoch, model_num = self._test_modelList[1], train_model = train_model, batch_size = batch_size)
                    self.validation_loss_list.append(self.loss_validation)
                    print('validation cost time : ', time.time()-t_validation)
                else:
                    self.validationTest(surfaceNet, eNet, epoch_num = epoch, model_num = self._test_modelList[0], draw = False, train_model = train_model, batch_size = batch_size)
                    self.validationTest(surfaceNet, eNet, epoch_num = epoch, model_num = self._test_modelList[1], draw = False, train_model = train_model, batch_size = batch_size)
                    self.validation_loss_list.append(self.loss_validation)
                    
            i_append = 0
            accuracy_ave = 0
            completeness_ave = 0
            f_distance_ave = 0
            
            t_start = time.time()
            for model_num in model_list:
                print('start training model',model_num)
                if(model_num == 51):
                    model_name = '%s/%s/SurfaceNet_epoch%r_half51'%(loss_type, training_key, epoch)
                    torch.save(surfaceNet.cpu().state_dict(), root_path + model_name)
                    surfaceNet.to(device)
                    model_name = '%s/%s/eNet_epoch%r_half51'%(loss_type, training_key, epoch)
                    torch.save(eNet.cpu().state_dict(), root_path + model_name)
                    eNet.to(device)
                if(model_num == 100):
                    model_name = '%s/%s/SurfaceNet_epoch%r_half100'%(loss_type, training_key, epoch)
                    torch.save(surfaceNet.cpu().state_dict(), root_path + model_name)
                    surfaceNet.to(device)
                    model_name = '%s/%s/eNet_epoch%r_half100'%(loss_type, training_key, epoch)
                    torch.save(eNet.cpu().state_dict(), root_path + model_name)
                    eNet.to(device)
                t_bug = time.time()

                dataset = Dataset(model_num, eNet = eNet)
                print('>>>>>~~~~~~: ', time.time() - t_bug)
                #self.recorder = dataset.recorder
                #batch_size = 64

                dataloader = DataLoader(dataset, batch_size= batch_size,
                                        shuffle=shuffle, num_workers=12)
                
                loss_whole_model = 0
                for i, data in enumerate(dataloader, 0):
                    
                    #torch.cuda.empty_cache()
                    #print('memory:',torch.cuda.memory_allocated())
                    #iter_record += 1
                    i_append += 1

                    if(self._datasetName == 'MVS2d'):                
                        batch_size_cvc,self.cubic_num,_,_,_ = data['cvc'].size()
                    else:
                        batch_size_cvc,self.cubic_num,_,_,_,_ = data['cvc'].size()
                                        
                    if(train_model == 'attSurface'):
                        attSurface.zero_grad()
                    else:
                        surfaceNet.zero_grad()
                        eNet.zero_grad()
                    
                    
                    if(train_model == 'attSurface'):
                        output = attSurface(data['cvc'].to(device))
                        
                    else:
                        output = 0
                        w_total = 0
                        
                        #t1 = time.time()
                        for i_c in range(self.cubic_num):
                        #for i_c in range(1):

                            if(self._datasetName == 'MVS2d'):
                                s = surfaceNet(data['cvc'][:,i_c,...].to(device))
                            else:
                                s = surfaceNet(data['cvc'][:,i_c,...].to(device))[:,0]
                            #s = surfaceNet(data['cvc'][:,i_c,...].to(device))
                            if not (w_detach  and (epoch < w_detach_num)):
                                s = s.detach()
                            if(attention_type == 'pixel'):
                                w = eNet(surfaceNet.s)[:,0]
                                w_total += w
                                output += s * w
                            else:
                                w = eNet(data['embedding'][:,i_c,...].to(device))
                                w_total += (w[...,None,None])
                                output += s * w[...,None,None]
                            #w = w.detach()
                            #s = s.detach()

                            
                            
                        
                        output = output/(w_total + 1e-15)
                        #output = s
                    #print('count output cost:', time.time() - t1)

                    if(loss_type == 'ppo_chamfer'):
                        
                        output_load = 0
                        w_total_load = 0           
                        for i_c in range(self.cubic_num):
                        #for i_c in range(1):
                            s = surfaceNet_load(data['cvc'][:,i_c,...].to(device))
                            w = eNet_load(data['embedding'][:,i_c,...].to(device))
                            w = w.detach()
                            s = s.detach()
                            
                            w_total_load += w[...,None,None].detach()
                            output_load += s * w[...,None,None]
                            
                        output_load = output_load/(w_total_load + 1e-15)
                        
                    
                    self.data = data
                    self.output= output.detach().cpu()
                    
                    if(self._datasetName == 'MVS2d'):
                        data_surface = data['surface'].to(device)
                    else:
                        data_surface = data['surface'][:,0,:,:,:].to(device)

                    threshold = 0.7
                    if(loss_type != 'augment')and(loss_type!='MSE'):
                        ground_truth_binary = data_surface.type(torch.cuda.ByteTensor)
                        thre_output_binary1 = (output > threshold)
                    if (loss_type == 'rl'):
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
                                alpha = 1- data_surface.sum((1,2,3))/params._cube_D**2
                                alpha = alpha[:,None,None,None]
                            
                                #alpha_num = data['surface'].to(device).sum()
                                #alpha = 1- alpha_num / (data['surface'].shape[0] * params._cube_D**3)

                                #alpha = 0.999
                            else:
                                alpha = 1- data_surface.sum((1,2,3))/params._cube_D**3
                                alpha = alpha[:,None,None,None]
                            
                                #alpha_num = data['surface'].to(device).sum()
                                #alpha = 1- alpha_num / (data['surface'].shape[0] * params._cube_D**3)

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
                    
                    if(loss_type == 'ppo_chamfer'):
                        surfaceNet_load.load_state_dict(surfaceNet.state_dict())
                        eNet_load.load_state_dict(eNet.state_dict())
                    
                    
                    #t1 = time.time()
                    loss.backward()
                    #print('backward time:', time.time() - t1)
                    if(train_model == 'attSurface'):
                        optimizer_att.step()
                    else:
                        if not(w_detach  and (epoch < w_detach_num)):
                            optimizer_s.step()
                        optimizer_e.step()
                    #print('step time:', time.time() - t1)
                    
                    if i_append % training_draw_iter == 1:
                        print('start drawing slice and saving ply')
                        t_drawing = time.time()
                        print(i_append)
                        #self.draw_sample(self.data, 
                        #                 self.output, 
                        #                 train_model = train_model,
                        #                 epoch_num = epoch, 
                        #                 mode = 'train',
                        #                 file_root = self.slice_path,
                        #                 model_num = model_num)
                        
                        if(self._datasetName != 'MVS2d'):
                            self.save_ply(data = self.data,
                                          output = self.output,
                                          epoch_num = epoch, 
                                          mode = 'train',
                                          file_root = self.ply_path,
                                          model_num = model_num)
                        print('drawing fucking painting cost time: ', time.time()-t_drawing)
                        print('[%d/%d][%d/%d]\tLoss: %.4f\tTime for training and drawing:%.4fs,%.4fs'
                              % (epoch, num_epochs, i, len(dataloader),
                                 loss_show/(i+1), t_drawing - t_start, time.time() - t_drawing))
                        t_start = time.time()
                        loss_show = 0
                        if (loss_type == 'rl'):
                            print('acc,comp,fd:', accuracy, completeness, f_distance)

                            #plt.subplot(311)
                            #plt.plot(self.accuracy_list)
                            #plt.subplot(312)
                            #plt.plot(self.completeness_list)
                            #plt.subplot(313)
                            #plt.plot(self.f_distance_list)

                        if(loss_type == 'rl'):
                            print('reward:', reward_ave.cpu())
                            self.reward_list.append(reward_ave.cpu().numpy())
                            #plt.plot(self.reward_list)
                            #plt.show()
                        if(loss_type == 'rl_chamfer'):
                            pass
                            #print('reward:', reward_ave.cpu())
                            #self.reward_list.append(reward_ave.cpu().numpy())
                            #plt.subplot(311)
                            #plt.plot(self.accuracy_list)
                            #plt.subplot(312)
                            #plt.plot(self.completeness_list)
                            #plt.subplot(313)
                            #plt.plot(self.f_distance_list)
                    memory_used = torch.cuda.memory_allocated()
                    max_memory = torch.cuda.max_memory_allocated()

                    self.writer.add_scalars(key_path + 'memory', {'memory':100 * memory_used/(max_memory+1e-1)},iter_record)
                    self.writer.add_scalars(key_path + 'training', {'loss_augment':loss.detach().cpu()},iter_record)
                    iter_record += 1
                    #writer.add_scalars(key_path + 'training_each_model/' + str(model_num).zfill(3), {'loss_augment':loss.detach().cpu()},iter_record)
                    
                self.writer.add_scalars(key_path + 'training_whole_model', {'loss_augment':loss_whole_model/(i+1)},iter_model_record)
                iter_model_record += 1
                loss_epoch += loss_whole_model/(i+1)
            
            self.writer.add_scalars(key_path + 'training_epoch', {'loss_augment':loss_epoch/len(model_list)},epoch)
                
   

    def load_model(self,
        model_path = 'experiment/network/',
        type_net = 'test',
        model_name1 = 'None', 
        model_name2 = 'None',
        model_name3 = 'None',
        ):
        
        
        #model_name = 'no_w_lr_0.0010/epoch' + str(80)

        PATH1 = model_path + model_name1
        PATH2 = model_path + model_name2
        PATH3 = model_path + model_name3
        

        if(model_name3 != 'None'):
            fg.to('cpu')
            fg.load_state_dict(torch.load(PATH3))
            if(type_net == 'train'):
                fg.train()
            elif(type_net == 'test'):
                fg.eval()
                print('use fg eval')
                #pass
            fg.to(device)

        if(model_name2 != 'None'):
            eNet.to('cpu')
            eNet.load_state_dict(torch.load(PATH2))
            if(type_net == 'train'):
                eNet.train()
            elif(type_net == 'test'):
                eNet.eval()
                print('use eNet eval')
                pass
            eNet.to(device)

        if(model_name1 != 'None'):
            surfaceNet.to('cpu')
            surfaceNet.load_state_dict(torch.load(PATH1), strict= False)
            if(type_net == 'train'):
                surfaceNet.train()
            elif(type_net == 'test'):
                surfaceNet.eval()
                print('use surface eval')
                #pass
            surfaceNet.to(device)
            

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
        test_dataset = Dataset(model_num, recorder = self.recorder)
        #self.recorder = self.test_dataset.recorder
        data_test = DataLoader(test_dataset, batch_size= batch_size,
                            shuffle=False, num_workers=12)
        
        self.loss_validation = 0
        acc_t = 0
        comp_t = 0
        f_t = 0
        
        for i, data in enumerate(data_test, 0):

            if(train_model == 'attSurface'):
                output = attSurface(data['cvc'].to(device)).detach()
                
                        
            else:
                if(self._datasetName == 'MVS2d'):
                    batch_size,self.cubic_num,_,_,_ = data['cvc'].size()
                else:
                    batch_size,self.cubic_num,_,_,_,_= data['cvc'].size()
                output = 0
                w_total = 0
                for i_c in range(self.cubic_num):
                #for i_c in range(1):
                    if(self._datasetName == 'MVS2d'):
                        s = surfaceNet(data['cvc'][:,i_c,...].to(device)).detach()
                    else:
                        s = surfaceNet(data['cvc'][:,i_c,...].to(device))[:,0].detach()
                    #s = surfaceNet(data['cvc'][:,i_c,...].to(device)).detach()
                    w = eNet(data['embedding'][:,i_c,...].to(device)).detach()
                    #self.data_cvc = data['cvc'][:,i_c,...].to(device)
                    #self.data_e = data['embedding'][:,i_c,...].to(device)
                    #w = w.detach()
                    #s = s.detach()
                    #w_list.append(w)
                    
                    w_total += (w[...,None,None])
                    output += s * w[...,None,None]
                
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
            
            
            if(count_chamfer):
                threshold = threshold
                ground_truth_binary = data_surface.type(torch.ByteTensor)
                thre_output_binary1 = (output > threshold)
                accuracy, completeness, f_distance = self.batch_chamfer_loss(thre_output_binary1, ground_truth_binary )
            else:
                accuracy, completeness, f_distance = (10,10,10)
            acc_t += accuracy
            comp_t += completeness
            f_t += f_distance
            
            #output = output.detach().cpu()
            
                
            
            #alpha = 1- data['surface'].to(device).sum((1,2,3))/params._cube_D**2
            #alpha = alpha[:,None,None,None]

            alpha_num = data['surface'].sum()
            alpha = 1- alpha_num / (batch_size * params._cube_D**3)

            #alpha = 0.999
            #alpha = 0.5
            
            if(self._datasetName == 'MVS2d'):
                loss = - (alpha * torch.log(output + 1e-15) * data['surface'] + (1-alpha) * torch.log(1-output + 1e-15) * (1-data['surface'])).mean()
            else:
                loss = - (alpha * torch.log(output + 1e-15) * data['surface'][:,0] + (1-alpha) * torch.log(1-output + 1e-15) * (1-data['surface'][:,0])).mean()
            
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
        
        #if(draw):
        #    if(model_num < 100):
        #        plt.subplot(311)
        #        plt.plot(self.accuracy_list_test)
        #        plt.title('accuravy_test')
        #        plt.subplot(312)
        #        plt.plot(self.completeness_list_test)
        #        plt.title('completeness_test')
        #        plt.subplot(313)
        #        plt.plot(self.f_distance_list_test)
        #        plt.title('f_distance_test')
                #plt.show()
        #    else:
        #        plt.subplot(311)
        #        plt.plot(self.accuracy_list_test2)
        #        plt.title('accuravy_test')
        #        plt.subplot(312)
        #        plt.plot(self.completeness_list_test2)
        #       plt.title('completeness_test')
        #        plt.subplot(313)
        #        plt.plot(self.f_distance_list_test2)
        #        plt.title('f_distance_test')
                #plt.show()
        
        self.loss_validation = self.loss_validation.tolist()
        self.loss_validation /= i
        
        print (self.loss_validation)
        print('end validation test')
        print('##############################################################')
        return self.loss_validation
    
    def save_ply(self,
        data,
        output,
        output_old = 'None',
        use_fg = False,
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
        interval_o_s =  2
        interval_o_old_s = 4

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

            o = output[show_i,:,:,:].numpy()
            o_out = (o > threshold).flatten()
            xyz_o = xyz[o_out,:] + np.array([interval_o_s,0,0])

            o = output[show_i,:,:,:][None,:,:,:]
            o_dense = o * torch.ones((3,1,1,1))
            o_dense = o_dense.flatten(start_dim = 1, end_dim = 3).transpose(0,-1).numpy()
            o_dense = (1 - o_dense) * 128

            xyz_s = np.concatenate((xyz_s, xyz_o), axis = 0)

            if(use_fg):
                o_old = output_old[show_i,:,:,:].numpy()
                o_old_out = (o_old > threshold).flatten()
                xyz_o_old = xyz[o_old_out,:] + np.array([interval_o_old_s,0,0])

                o_old = output_old[show_i,:,:,:][None,:,:,:]
                o_old_dense = o_old * torch.ones((3,1,1,1))
                o_old_dense = o_old_dense.flatten(start_dim = 1, end_dim = 3).transpose(0,-1).numpy()
                o_old_dense = (1 - o_old_dense) * 128

                xyz_s = np.concatenate((xyz_s, xyz_o_old), axis = 0)

            

            if(mode == 'train'):
                path_cvc = file_root + 'train/cvc/epoch_%s/model_%r__cvc_id_%r__show_id_%r.ply'%(str(epoch_num).zfill(3), model_num, idx, show_i)
                path_s = file_root + 'train/surface/epoch_%s/model_%r__cvc_id_%r__show_id_%r.ply'%(str(epoch_num).zfill(3), model_num, idx, show_i)
                path_o = file_root + 'train/output/epoch_%s/model_%r__cvc_id_%r__show_id_%r.ply'%(str(epoch_num).zfill(3), model_num, idx, show_i)

                #path_out = file_root + '/surface'
                self.d2s.save2ply(path_cvc, xyz_total, c_total)
                self.d2s.save2ply(path_s, xyz_s)
                #self.d2s.save2ply(path_o, xyz, o_dense)

            if(mode == 'test'):
                path_cvc = file_root + 'test/cvc/model_%r/epoch_%s__cvc_id_%r__show_id_%r.ply'%(model_num, str(epoch_num).zfill(3), idx, show_i)
                path_s = file_root + 'test/surface/model_%r/epoch_%s__cvc_id_%r__show_id_%r.ply'%(model_num, str(epoch_num).zfill(3), idx, show_i)
                path_o = file_root + 'test/output/model_%r/epoch_%s__cvc_id_%r__show_id_%r.ply'%(model_num, str(epoch_num).zfill(3), idx, show_i)

                #path_out = file_root + '/surface'
                self.d2s.save2ply(path_cvc, xyz_total, c_total)
                self.d2s.save2ply(path_s, xyz_s)
                #self.d2s.save2ply(path_o, xyz, o_dense)
        print('save ply result successful')
        
        
    def draw_sample(self, 
        data, 
        output, 
        output_old = None,
        cvc_num = 3,
        show_num = 4,
        save_image = False, 
        use_fg = False,
        train_model = 'none',
        file_root = 'labData/',
        mode = 'train',
        epoch_num = 10000,
        model_num = 10000,
        detach = False):

        if(self._datasetName != 'MVS2d'):
            batch_size, view_num, _,_,_,image_size = data['cvc'].shape

        if(cvc_num > self.cubic_num):
            cvc_num = self.cubic_num
        if(show_num > batch_size):
            show_num = batch_size
        
        
        if(detach):
            cvc = data['cvc'].detach().numpy()
            embedding = data['embedding'].detach().numpy()
            surface = data['surface'].detach().numpy()
            idxs = data['idx_validCubes'].detach()
        else:
            cvc = data['cvc'].numpy()
            embedding = data['embedding'].numpy()
            surface = data['surface'].numpy()
            idxs = data['idx_validCubes']
        
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
                #plt.show()
        else:
            for i in range(show_num):
                idx = idxs[i].item()
                
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
                    #plt.show()
                    plt.clf()
                else:
                    path_dirc = file_root + 'test/model_%r'%(model_num)
                    if not os.path.exists(path_dirc):
                        os.makedirs(path_dirc)
                    path = file_root + 'test/model_%r/epoch_%s__cvc_id_%r__show_id_%r.png'%(model_num, str(epoch_num).zfill(3), idx, i)
                    plt.savefig(path)
                    #plt.show()
                    plt.clf()


if __name__ == '__main__':

    print('start training')
    params = Params()
    d2s = Dense2Sparse()
    #model_list = range(7,9)
    model_list = [7,8,14,16,18,19,20,22,30,31,36,39,41,42,44,45,46,47,50,51,52,53,55,57,58,60,61,63,64,65,68,69,70,71,72,74,76,83,84,85,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,107,108,109,111,112,113,115,116,119,120,121,122,123,124,125,126,2,6]

    #model_list = [7,8,14,16,18,19,20,22,30,31,36,39,41,42,44,45,46,47,50,51,52,53,55,57,58,60,61,63,64,65,68,69,70,71,72,74,76]

    print('finished data preparation')

    train_iter = TrainIter(d2s = d2s)

    use_cuda = True
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    attention_type = 'new_big'
    
    if(params._datasetName != 'MVS2d'):
        print('3d')
        surfaceNet = SurfaceNet_3d_old(use_ad = False).to(device)
        #surfaceNet.train()

        if(attention_type == 'pixel'):
            eNet = EmbeddingNet_3d_pixel().to(device)
        elif(attention_type == 'new'):
            eNet = EmbeddingNet_3d_new().to(device)
        elif(attention_type == 'new_big'):
            eNet = EmbeddingNet_3d_new_big().to(device)
        else:
            eNet = EmbeddingNet_3d().to(device)
        #eNet.train()

        #fg = FineGenerator_3d_small().to(device)
        fg = FineGenerator_3d_res(residual_type = 'res_detach', activate_type = 'new').to(device)
        #fg.eval()
        #fg.train()
        #ad = Addition_3d_res_new(residual_type = 'res_detach').to(device)
    torch.cuda.empty_cache()
    #memory_used = torch.cuda.memory_allocated()
    #max_memory = torch.cuda.max_memory_allocated()
    #print('memory used',memory_used/(max_memory+0.0))
    
    '''
    epoch_name = 'epoch0'
    model_path = 'DTU/augment/041701_trainW/'
    train_iter.load_model(model_path = './experiment/network/' + model_path,
                       model_name1 = 'SurfaceNet_' + epoch_name, 
                      # model_name2 = 'eNet_' + epoch_name,
                       #model_name3 = 'fgNet_' + epoch_name,
                       )
    '''
    
    '''
    epoch_name = 'epoch1'
    model_path = 'DTU/Wgan/041402_ad_pretrain/'
    train_iter.load_model(model_path = './experiment/network/' + model_path,
                   model_name1 = 'surfaceNet_' + epoch_name, 
                   #model_name2 = 'eNet_' + epoch_name,
                   #model_name4 = 'adNet_' + epoch_name,
                    )
    '''
    
    '''
    train_iter.load_model(model_path = './experiment/network/DTU/Wgan/041410/',
                   #model_name1 = 'surfaceNet_epoch1', 
                   #model_name2 = 'eNet_epoch1',
                   model_name3 = 'fgNet_epoch1',
                   )
    '''
    
    epoch_name = 'epoch3'
    model_path = 'DTU/Wgan/050201_newW/'
    train_iter.load_model(model_path = './experiment/network/' + model_path,
                       #model_name1 = 'SurfaceNet_' + epoch_name, 
                       model_name2 = 'eNet_' + epoch_name,
                       #model_name3 = 'fgNet_' + epoch_name,
                       )
    
    epoch_name = 'epoch1_half51'
    model_path = 'DTU/Wgan/050701_fg_begin/'
    train_iter.load_model(model_path = './experiment/network/' + model_path,
                       model_name1 = 'SurfaceNet_' + epoch_name, 
                       model_name2 = 'eNet_' + epoch_name,
                       #model_name3 = 'fgNet_' + epoch_name,
                       )
    
    '''
    train_iter.train(lr_s = 2e-3,
                     lr_e = 1e-3,
                    lr_decay = 0.95, 
                 loss_alpha = 0.1,
                 clip_min = 0.2,
                 clip_max = 0.2,
                 num_epochs = 30,
                 reward_epoch =10005,
                 model_list = model_list,
                 w_detach = False, 
                 w_detach_num = 10,
                 use_augment = True,
                 shuffle = True,
                 validation_iter = 100,
                 training_draw_iter = 200,
                 draw_iter = 100,
                 save_iter = 1,
                 batch_size = 10,
                 attention_type = attention_type,
                 train_model = 'attSurfac',
                 loss_type = 'augment',
                 training_key = '050303_freeSpace')
    
    '''
    
    train_iter.train_gan(model_list = model_list,
                     lr_gen_s = 2e-4, 
                     lr_gen_e = 1e-4,
                     lr_gen_fg = 3e-5,
                     lr_dis = 0,
                     weight_cliping_limit = 0.2,
                     g_step = 1,
                     d_step = 0,
                     
                     teacher_type = 'none',
                     use_augment = True,
                     
                     use_fg = True,
                     use_fg_new = False,
                     use_ad = False,
                     
                     teacher_type_fg = 'MSE',
                     use_augment_fg = True,
                     mixed_alpha = 5.0,
                     
                     use_dynamic_alpha = False,
                     use_teacher = True,
                     
                     use_g_error = False,
                     gan_layer = 2,
                     use_fg_teacher = True,
                     fg_detach = False,
                     
                     alpha_decay_fg = 3.0,
                     alpha_decay = 1.0,
                     decay_rate = 1.0,
                     
                     
                     use_fg_transfer = False,
                     fg_tau = 3.0,
                     layer1_tau = 1.0,
                     god_w = 1e-1,
                     
                     interpolation = 1.0,
                     add_mass = 0.2,
                     minus_mass = 0.05,
                     
                     attention_type = attention_type,
                     num_epochs = 50,
                     shuffle = True,
                     validation_iter = 50,
                     training_draw_iter = 200,
                     batch_size = 10,
                     save_iter = 1,
                     loss_type = 'Wgan',
                     training_key = '050710_newFG',
                     show = False)
    
    