import numpy as np
import itertools
import torch
import os
import random
import pdb
import scipy.io
import math
import matplotlib.pyplot as plt
import time
import torch

class Params(object):
    def __init__(self):
	
        self.whatUWant = "reconstruct_model"

        self._datasetName = 'DTU'  
        self.batch_size = 128
        self._GPUMemoryGB = 12  # how large is your GPU memory (GB)
        self._input_data_rootFld = "./inputs"
        self._output_data_rootFld = "./outputs"

        self._DEBUG_output_data_rootFld = ""

        self._DEBUG_input_data_rootFld = "./inputs"  

        self._DEBUG_input_data_rootFld_exists = os.path.exists(self._DEBUG_input_data_rootFld)
        self._DEBUG_input_data_rootFld_exists  = True
        self._DEBUG_output_data_rootFld_exists = os.path.exists(self._DEBUG_output_data_rootFld)
        self._DEBUG_output_data_rootFld_exists = True
        self._input_data_rootFld = self._DEBUG_input_data_rootFld if self._DEBUG_input_data_rootFld_exists else self._input_data_rootFld

        self._output_data_rootFld = self._DEBUG_output_data_rootFld if self._DEBUG_output_data_rootFld_exists else self._output_data_rootFld

        self.debug_BB = False  ######################!!!!!!!!!!!!!!!!!!!!!!!!!!!1
        self._output_data_rootFld += '_Debug_BB' if self.debug_BB else ''
        
        self._use_old_early_rejection = False

        
        if self.whatUWant is "reconstruct_model": 
            """
            In this mode, reconstruct models using the similarityNet and SurfaceNet.
            """
            #------------ 
            ## params only for reconstruction
            #------------F
            # DTU: numbers: 1 .. 128
            # Middlebury: dinoSparseRing
           
            if self._datasetName is 'DTU':
                self._modelList = [9]     # [3,18,..]
                self._test_modelList = [6,7]
            
            if self._datasetName is 'DTU':
                self._cube_D = 32 #32/64 # size of the CVC = __cube_D ^3, in the paper it is (s,s,s)
           
            self.load_viewPair_mode = 'dynamic'
            self._min_prob = 0.46 # in order to save memory, filter out the voxels with prob < min_prob
            self._tau = 0.3     # fix threshold for thinning
            self._gamma = 0.5   # used in the ray pooling procedure

            self._batchSize_similNet_patch2embedding_perGB = 2#100
            self._batchSize_similNet_embeddingPair2simil_perGB = 5#100000
            self._batchSize_viewPair_w_perGB = 5#100000     
            self._batchSize_viewPair_w = 5000

            use_cuda = True
            self.device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
        

            #self._batchSize_similNet_patch2embedding, self._batchSize_similNet_embeddingPair2simil,self. _batchSize_viewPair_w = np.array([\
            ##        self._batchSize_similNet_patch2embedding_perGB, \
            #        self._batchSize_similNet_embeddingPair2simil_perGB, \
            #        self._batchSize_viewPair_w_perGB, \
            #        ], dtype=np.uint64) * self._GPUMemoryGB

            ############# 
            ## similarNet
            #############

            # each patch pair --> features to learn to decide view pairs
            # 2 * 128D/image patch + 1 * (dis)similarity + 1 * angle<v1,v2>
            if(self._datasetName is 'MVS2d'):
                self._D_imgPatchEmbedding = 24
            else:
                self._D_imgPatchEmbedding = 204#120 #must have factor 12
            self._D_viewPairFeature = self._D_imgPatchEmbedding * 2 + 1 + 1     # embedding / view pair angle / similarity
            self._similNet_hidden_dim = 100
            self._pretrained_similNet_model_file = os.path.join(self._input_data_rootFld, 'SurfaceNet_models/epoch33_acc_tr0.707_val0.791.model') # allDTU
            self._imgPatch_hw_size = 64
            self._MEAN_IMAGE_BGR = np.asarray([103.939,  116.779,  123.68]).astype(np.float32)
            self._triplet_alpha = 100
            self._weight_decay = 0.0001
            self._DEFAULT_LR = 0 # will be updated during param tuning

            ############
            # SurfaceNet
            ############

            # view index of the considered views
            #self._use_pretrained_model = True
            #if self._use_pretrained_model:
            #    self._layerNameList_2_load = ["output_SurfaceNet_reshape","output_softmaxWeights"] ##output_fusionNet/fuse_op_reshape
            #    self._pretrained_SurfaceNet_model_file = os.path.join(self._input_data_rootFld, 'SurfaceNet_models/2D_2_3D-19-0.918_0.951.model') # allDTU
            #self._cube_Dcenter = {32:26, 64:52}[self._cube_D] # only keep the center part of the cube because of boundary effect of the convNet.

            ####################
            # adaptive threshold
            ####################
            self._beta = 6
            self._N_refine_iter = 8
            self._cube_overlapping_ratio = 0.5 ## how large area is covered by the neighboring cubes. 
            self._weighted_fusion = True # True: weighted average in the fusion layer; False: average

            #self._batchSize_nViewPair_SurfaceNet_perGB = {32:1.2, 64:0.1667}[self._cube_D]  # 0.1667 = 1./6
            #self._batchSize_nViewPair_SurfaceNet = int(math.floor(self._batchSize_nViewPair_SurfaceNet_perGB * self._GPUMemoryGB))

            

        elif self.whatUWant is "train_xxx":
            pass
        
        self._MEAN_CVC_RGBRGB = np.asarray([123.68,  116.779,  103.939, 123.68,  116.779,  103.939]).astype(np.float32) # RGBRGB order (VGG mean)
        self._MEAN_PATCHES_BGR = np.asarray([103.939,  116.779,  123.68]).astype(np.float32)
        
        #self.load_modelSpecific_params(self._datasetName,self._modelList[0])#TODO
        
    def load_modelSpecific_params(self, datasetName, model, cube_D = None, sparsity = None):
        
        if(cube_D is not None):
            self._cube_D = cube_D
        if(sparsity is not None):
            self.sparsity = sparsity
        else:
            self.sparsity = 5
        self.initialPtsNamePattern = None  # if defined, the cubes position will be initialized according to these points that will be quantizated by $resolution$

        # view index of the considered views
        self._use_pretrained_model = True
        if self._use_pretrained_model:
            self._layerNameList_2_load = ["output_SurfaceNet_reshape","output_softmaxWeights"] ##output_fusionNet/fuse_op_reshape
            self._pretrained_SurfaceNet_model_file = os.path.join(self._input_data_rootFld, 'SurfaceNet_models/2D_2_3D-19-0.918_0.951.model') # allDTU
        self._cube_Dcenter = {32:26, 64:52, 128:104}[self._cube_D] # only keep the center part of the cube because of boundary effect of the convNet.

        ####################
        # adaptive threshold
        ####################


        self._beta = 6
        self._N_refine_iter = 8
        #self._cube_overlapping_ratio = 0.5 ## how large area is covered by the neighboring cubes. 
        self._weighted_fusion = True # True: weighted average in the fusion layer; False: average

        self._batchSize_nViewPair_SurfaceNet_perGB = {32:1.2, 64:0.1667}[self._cube_D]  # 0.1667 = 1./6
        self._batchSize_nViewPair_SurfaceNet = int(math.floor(self._batchSize_nViewPair_SurfaceNet_perGB * self._GPUMemoryGB))


        #self._batchSize_viewPair_w = 1000000000

        use_cuda = True
        self.device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
        
       
        if datasetName is "DTU":
            self.datasetFolder = os.path.join(self._input_data_rootFld, 'DTU_MVS')
            self.imgNamePattern = "Rectified/scan{}/rect_#_3_r5000.{}".format(model, 'png' if self._DEBUG_input_data_rootFld_exists else 'jpg')    # replace # to {:03} 
            print(self.imgNamePattern)
            print('self._cube_D', self._cube_D)
            print('use newnewnew')

            self.poseNamePattern = "SampleSet/MVS Data/Calibration/cal18/pos_#.txt"  # replace # to {:03}
            self.stl_name = 'Points/stl/stl%s_total.ply'%str(model).zfill(3)
            self.stl_comb_name = 'Points/stl_comb/stl%s_total.ply'%str(model).zfill(3)
            self.file_root_volumn_train = "./inputs/DTU_MVS/dataset_class/train_0406/"
            self.file_root_volumn_train = "/home/jinzhi/hdd10T/dataset_class/test_0508/"
            
            self.file_root_volumn_test = '/home/jinzhi/hdd10T/dataset_class/test_0417/' #'/home/jinzhi/hdd10T/dataset_class/test_0406'
            self.N_viewPairs4inference = [4]
            self.thre_angle = 360
            self.selection_method = 'random' #angle/random
            self.load_viewPair_mode = 'dynamic' #'dynamic'/'fixed'
            self.viewPair_mode = 'inference'  #'inference'
            self.addition_name = 'embedding_204'
            self.free_space = False
            self.resol = np.float32(1.0) #0.4 resolution / the distance between adjacent voxels
            self.BBNamePattern = "SampleSet/MVS Data/ObsMask/ObsMask{}_10.mat".format(model)
            self.BB_filePath = os.path.join(self.datasetFolder, self.BBNamePattern)
            self.BB_matlab_var = scipy.io.loadmat(self.BB_filePath)   # matlab variable
            #self.reconstr_sceneRange = np.asarray([(-40, 40), (80, 160), (630, 680)])
            #self.reconstr_sceneRange = np.asarray([(-80, 135), (-220, 200), (440, 840)])
            self.reconstr_sceneRange = self.BB_matlab_var['BB'].T
            size_reconstr_sceneRange = self.reconstr_sceneRange[:,1] - self.reconstr_sceneRange[:,0]
            reconstr_sceneRange_low = self.reconstr_sceneRange[:,0] - 0.1 * size_reconstr_sceneRange
            reconstr_sceneRange_up = self.reconstr_sceneRange[:,1] + 0.1 * size_reconstr_sceneRange
            self.reconstr_sceneRange = np.concatenate((reconstr_sceneRange_low[:,None], reconstr_sceneRange_up[:,None]), axis = 1)
            #[-73 129], [-197  183], [472 810]
            #self.reconstr_sceneRange = np.asarray([(-20, 20), (100, 140), (640, 670)])
            #self.BB = self.reconstr_sceneRange if self.debug_BB else self.BB_matlab_var['BB'].T   # np(3,2)
            self.BB = self.reconstr_sceneRange
            self.viewList = range(1,50,self.sparsity)  # range(1,50)
            #self.viewList = [1,2,3,11,12,13,21,22,23,31,32,33,41,42,43]
            #self.viewList = [1,2,3,10,11,12,19,20,21,28,29,30,37,38,39,46,47]
            #self.viewList = [1,2,12,13,23,24,34,35,45,46,]
            #self.viewList = [1,2,11,12,21,22,31,32,41,42,]
            #self.viewList = [1,2,10,11,19,20,28,29,37,38,46,47]
            #self.viewList = [30,31,41] 
            print('sparsity:',self.sparsity)
            #self.viewList = [1,5,24,39,49]  #g100
            #self.viewList = [1,5,39,49]  #g200
            #self.viewList = [1,5,49]  #g300
            #self.viewList = [3,48]  #g400
            #self.viewList = [2,7,9,12,15,18,20,22,24,27,30,32,35,38,40,42,44,46,49]
            #self.viewList = [2,4,6,9,12,14,17,20,24,27,29,32,35,38,40,44,46]
            #self.viewList = [1,3,5,7,10,13,16,19,21,25,28,30,34,37,39,42,45,49]
            #self.viewList = [1,5,10,13,17,24,29,38,41,45] #g50

        