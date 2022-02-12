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

class Params(object):
    def __init__(self):
	
        self.whatUWant = "reconstruct_model"

        self._datasetName = 'MVS2d'  # Middlebury / DTU / people/MVS2d
        self.batch_size = 128
        self._GPUMemoryGB = 12  # how large is your GPU memory (GB)
        self._input_data_rootFld = "./inputs"
        self._output_data_rootFld = "./outputs"

        self._DEBUG_input_data_rootFld = "/home/mengqi/fileserver/datasets"     # used for debug: if exists, use this path
        self._DEBUG_output_data_rootFld = "/home/mengqi/fileserver/results/MVS/SurfaceNet"
        self._DEBUG_input_data_rootFld_exists = os.path.exists(self._DEBUG_input_data_rootFld)
        self._DEBUG_input_data_rootFld_exists = True
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
            if self._datasetName is 'MVS2d':
                self._modelList = [1]     # [3,18,..]
                self._test_modelList = [310,344]
            if self._datasetName is 'DTU':
                self._modelList = [9]     # [3,18,..]
                self._test_modelList = [6,7]
            elif self._datasetName is 'Middlebury':
                self._modelList = ["dinoSparseRing"]     # ["dinoSparseRing", "..."]
            elif self._datasetName is 'people':
                # frame 0: ["head", "render_07"]
                # frame 1: ["model_20_anim_4", "model_42_anim_8", "model_42_anim_9", "render_11"]
                # frame 11: ["model_20_anim_4", "model_42_anim_8", "model_42_anim_9"]  # may have different resol
                # frame 30: ["T_samba"]
                # frame 40: ["T_samba"]
                # frame 50: ["flashkick", "I_crane"]
                # frame 70: ["T_samba"]  # resol: 0.005
                # frame 100: ["pop", "I_crane"]
                # frame 120: ["I_crane"]
                self._frame = 100  # [0, 50, 100, 150]
                self._modelList = ["pop", "I_crane"]     # ["D_bouncing", "T_samba", "..."]
                self._viewList = range(1, 5)  # range(1, 5)
                self._output_data_rootFld = os.path.join(self._output_data_rootFld, "people/frame{}_views{}".format(self._frame, self._viewList))
            
            if self._datasetName is 'MVS2d':
                self._cube_D = 64 #32/64 # size of the CVC = __cube_D ^3, in the paper it is (s,s,s)
            if self._datasetName is 'DTU':
                self._cube_D = 32 #32/64 # size of the CVC = __cube_D ^3, in the paper it is (s,s,s)
            
            self._min_prob = 0.46 # in order to save memory, filter out the voxels with prob < min_prob
            self._tau = 0.7     # fix threshold for thinning
            self._gamma = 0.8   # used in the ray pooling procedure

            self._batchSize_similNet_patch2embedding_perGB = 2#100
            self._batchSize_similNet_embeddingPair2simil_perGB = 10#100000
            self._batchSize_viewPair_w_perGB = 10#100000     


            self._batchSize_similNet_patch2embedding, self._batchSize_similNet_embeddingPair2simil,self. _batchSize_viewPair_w = np.array([\
                    self._batchSize_similNet_patch2embedding_perGB, \
                    self._batchSize_similNet_embeddingPair2simil_perGB, \
                    self._batchSize_viewPair_w_perGB, \
                    ], dtype=np.uint64) * self._GPUMemoryGB

            ############# 
            ## similarNet
            #############

            # each patch pair --> features to learn to decide view pairs
            # 2 * 128D/image patch + 1 * (dis)similarity + 1 * angle<v1,v2>
            if(self._datasetName is 'MVS2d'):
                self._D_imgPatchEmbedding = 24
            else:
                self._D_imgPatchEmbedding = 120 #must have factor 12
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
            self._use_pretrained_model = True
            if self._use_pretrained_model:
                self._layerNameList_2_load = ["output_SurfaceNet_reshape","output_softmaxWeights"] ##output_fusionNet/fuse_op_reshape
                self._pretrained_SurfaceNet_model_file = os.path.join(self._input_data_rootFld, 'SurfaceNet_models/2D_2_3D-19-0.918_0.951.model') # allDTU
            self._cube_Dcenter = {32:26, 64:52}[self._cube_D] # only keep the center part of the cube because of boundary effect of the convNet.

            ####################
            # adaptive threshold
            ####################
            self._beta = 6
            self._N_refine_iter = 8
            self._cube_overlapping_ratio = 1/2. ## how large area is covered by the neighboring cubes. 
            self._weighted_fusion = True # True: weighted average in the fusion layer; False: average

            self._batchSize_nViewPair_SurfaceNet_perGB = {32:1.2, 64:0.1667}[self._cube_D]  # 0.1667 = 1./6
            self._batchSize_nViewPair_SurfaceNet = int(math.floor(self._batchSize_nViewPair_SurfaceNet_perGB * self._GPUMemoryGB))

            

        elif self.whatUWant is "train_xxx":
            pass
        
        self._MEAN_CVC_RGBRGB = np.asarray([123.68,  116.779,  103.939, 123.68,  116.779,  103.939]).astype(np.float32) # RGBRGB order (VGG mean)
        self._MEAN_PATCHES_BGR = np.asarray([103.939,  116.779,  123.68]).astype(np.float32)
        
        self.load_modelSpecific_params(self._datasetName,self._modelList[0])#TODO
        
    def load_modelSpecific_params(self, datasetName, model):
        
        self.initialPtsNamePattern = None  # if defined, the cubes position will be initialized according to these points that will be quantizated by $resolution$

        if datasetName is "MVS2d":
            self.datasetFolder = os.path.join('./2dData', 'MVS2d')
            self.imgNamePattern = "Rectified/scan{}/rect_#_0_r5000.{}".format(model, 'png' if self._DEBUG_input_data_rootFld_exists else 'jpg')    # replace # to {:03} 
            self.poseNamePattern = "Calibration/cal18/pos_#.txt"  # replace # to {:03}
            if(model<100):
                self.stl_name = './2dData/MVS2d/Points/arti_'+str(model).zfill(3) + '.npy'
            elif(model<200):
                self.stl_name = './2dData/MVS2d/Points/regu_'+str(model-100).zfill(3) + '.npy'
            elif(model<300):
                self.stl_name = './2dData/MVS2d/Points/regu_'+str(model-200).zfill(3) + '.npy'
            elif(model<400):
                self.stl_name = './2dData/MVS2d/Points/regu_'+str(model-300).zfill(3) + '.npy'
            
            self.N_viewPairs4inference = [6]
            self.thre_angle = 60
            self.resol = np.float32(1.0) #0.4 resolution / the distance between adjacent voxels
            #self.BBNamePattern = "SampleSet/MVS Data/ObsMask/ObsMask{}_10.mat".format(model)
            #self.BB_filePath = os.path.join(self.datasetFolder, self.BBNamePattern)
            #self.BB_matlab_var = scipy.io.loadmat(self.BB_filePath)   # matlab variable
            self.reconstr_sceneRange = np.asarray([(0, 800), (0, self.resol), (0, 800)])
            #self.reconstr_sceneRange = np.asarray([(-20, 20), (100, 140), (640, 670)])
            self.BB = self.reconstr_sceneRange 
            self.viewList = range(1,127)  # range(1,50)
        
        if datasetName is "DTU":
            self.datasetFolder = os.path.join(self._input_data_rootFld, 'DTU_MVS')
            self.imgNamePattern = "Rectified/scan{}/rect_#_3_r5000.{}".format(model, 'png' if self._DEBUG_input_data_rootFld_exists else 'jpg')    # replace # to {:03} 
            self.poseNamePattern = "SampleSet/MVS Data/Calibration/cal18/pos_#.txt"  # replace # to {:03}
            self.stl_name = 'Points/stl/stl%s_total.ply'%str(model).zfill(3)
            self.N_viewPairs4inference = [5]
            self.thre_angle = 360
            self.resol = np.float32(1.0) #0.4 resolution / the distance between adjacent voxels
            self.BBNamePattern = "SampleSet/MVS Data/ObsMask/ObsMask{}_10.mat".format(model)
            self.BB_filePath = os.path.join(self.datasetFolder, self.BBNamePattern)
            self.BB_matlab_var = scipy.io.loadmat(self.BB_filePath)   # matlab variable
            self.reconstr_sceneRange = np.asarray([(-40, 40), (80, 160), (630, 680)])
            #self.reconstr_sceneRange = np.asarray([(-20, 20), (100, 140), (640, 670)])
            self.BB = self.reconstr_sceneRange if self.debug_BB else self.BB_matlab_var['BB'].T   # np(3,2)
            self.viewList = range(1,50)  # range(1,50)

        if datasetName is "Middlebury":
            self.datasetFolder = os.path.join(self._input_data_rootFld, 'Middlebury')
            self.N_viewPairs4inference = [3]
            self.resol = np.float32(0.00025) # 0.00025 resolution / the distance between adjacent voxels
            if model is "dinoSparseRing":
                self.imgNamePattern = "{}/dinoSR0#.png".format(model)   # replace # to {:03}
                self.poseNamePattern = "{}/dinoSR_par.txt".format(model)
                self.BB = np.array([(-0.061897, 0.010897), (-0.018874, 0.068227), (-0.057845, 0.015495)], dtype=np.float32)   # np(3,2)
                self.viewList = range(7,13) #range(1,16)
            else:
                raise Warning('current model is unexpected: '+model+'.') 

        if datasetName is "people":
            # people dataset website: http://people.csail.mit.edu/drdaniel/mesh_animation/
            self.datasetFolder = os.path.join(self._input_data_rootFld, 'people/samples/mit_format_mvs_example_data_4')
            self.imgNamePattern = "{}/images/Image@_{:04}.png".format(model, self._frame)   # replace # to {:03}, relace @ to {}
            self.poseNamePattern = "{}/calibration/Camera@.Pmat.cal".format(model)
            self.BBNamePattern = "{}/meshes/mesh_{:04}.obj".format(model, self._frame)
            self.N_viewPairs4inference = [2]
            self.resol = np.float32(0.005) # resolution / the distance between adjacent voxels
            self.BB = scene.readBB_fromModel(objFile = os.path.join(self.datasetFolder, self.BBNamePattern)) # None / np.array([(0.2091, 0.5904), (0.0327, 1.7774), (-0.3977, 0.3544)], dtype=np.float32)   # np(3,2)
            self.initialPtsNamePattern = None # None / "{}/visualHull/vhull_4_views_1346/{:04}.ply".format(model, __frame)
            self.viewList = self._viewList # range(1,5) # range(4,9) + [1] #range(1,9)

