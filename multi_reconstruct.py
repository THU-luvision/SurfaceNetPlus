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


imp.reload(Parameter)
from Parameter import Params
imp.reload(Prepair)
from Prepair import Dataset, Recorder

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import os
import cPickle as pickle
import rayPooling
import sys
#import camera
from plyfile import PlyData, PlyElement

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


import main
imp.reload(Surf)
imp.reload(Attn)
imp.reload(F_G)
imp.reload(Disc)
imp.reload(main)
from Surf import EmbeddingNet_3d_new_big, SurfaceNet_3d_old
from Attn import AttentionSurface, Self_Attn
from F_G import  FineGenerator_3d_res
from Disc import Discriminator
from main import TrainIter

from Parameter import Params
from Prepair import Dataset, Recorder, ViewPairSelection
params = Params()
d2s = Dense2Sparse()
#train_iter = TrainIter(recorder = None,d2s = d2s)



def load_model(model_path = 'experiment/network/',
        #model_type = 'load',
        model_name1 = 'None', 
        model_name2 = 'None',
        model_name3 = 'None',
        model_name4 = 'None',
        model_name5 = 'None',
        train_model = 'none',):
        
        
        #model_name = 'no_w_lr_0.0010/epoch' + str(80)

        PATH1 = model_path + model_name1
        PATH2 = model_path + model_name2
        PATH3 = model_path + model_name3
        PATH4 = model_path + model_name4
        PATH5 = model_path + model_name5
        
        if(model_name5 != 'None'):
            eNet_p.to('cpu')
            eNet_p.load_state_dict(torch.load(PATH5))
            eNet_p.eval()
            eNet_p.to(device)

        
        if(model_name4 != 'None'):
            ad.to('cpu')
            ad.load_state_dict(torch.load(PATH4))
            
            #ad.train()
            ad.to(device)
            ad.eval()
        
        if(model_name1 != 'None'):
            surfaceNet.to('cpu')
            surfaceNet.load_state_dict(torch.load(PATH1))
            surfaceNet.eval()
            surfaceNet.to(device)
        
        
        if(model_name3 != 'None'):
            fg.to('cpu')
            fg.load_state_dict(torch.load(PATH3), strict = False)
            #fg.train()
            fg.eval()
            fg.to(device)

        
        if(model_name2 != 'None'):
            eNet.to('cpu')
            eNet.load_state_dict(torch.load(PATH2))
            eNet.eval()
            eNet.to(device)

        #surfaceNet.train()
        #eNet.train()
        
        #('N_validCubes', 12960)

import denoising
from Prepair import Dataset, Recorder, ViewPairSelection, Volumn

class Reconstruct(Params):
    def __init__(self):
        super(Reconstruct, self).__init__()
        self.xyz_dict = {}
        self.rgb_dict = {}
        
        self.params = {6.4:{'min_prob':0.5, 
                           'tau_list':[0.7,0.5],
                           'gamma_list':[0.9],
                           'tau_fg':[-1.0,-0.6],
                           'use_fg':True,
                           'use_fg_new':False,
                           'enable_rayPooling':True},
                     }
        self.pts_xyz_dict = {}
        self.updated_sparse_dict = {}
        
    def multi_scale(self,
                    resol_tau_list,
                    model_path = './experiment/network/DTU/augment/040103/',
                    model_name = 'epoch10/',):
        self.xyzs = None
        self.rgbs = None
        for ele in resol_tau_list:
            if(self.xyzs is None):
                self.xyzs = self.xyz_dict[ele]
                self.rgbs = self.rgb_dict[ele]
            else:
                self.xyzs = np.concatenate((self.xyzs,self.xyz_dict[ele]), axis = 0)
                self.rgbs = np.concatenate((self.rgbs,self.rgb_dict[ele]), axis = 0)
        
        
        #if(enable_rayPooling):
        reconstruct_name = 'model:%s_gamma:%s__reject:%s_select:%s__angle:%s_resol_tau:%s.ply'%(str(model_num), str(gamma), str(train_rejection), str(selection_method),str(thre_angle), str(resol_tau_list))
        #else:
        #    reconstruct_name = 'model:%s__ray:%s_min_prob:%s__reject:%s_select:%s__angle:%s_resol:%s.ply'%(str(model_num),str(enable_rayPooling),str(min_prob),str(train_rejection), str(selection_method), str(thre_angle),str(resol))

        if not os.path.exists(model_path + model_name):
            os.makedirs(model_path + model_name)

        ply_filePath = model_path + model_name + reconstruct_name
        dense2sparse.save2ply(ply_filePath, self.xyzs, self.rgbs, None)
        return self.xyzs, self.rgbs
        
    
    def multi_reconstruct(self, 
                           test_dataset_dict,
                            model = 9, 
                            batch_size = 4,
                            N_viewPairs4inference = [5],
                            sparsity = 3,
                            addition_name = '',
                            addition_name2 = '',
                            model_path = './experiment/network/DTU/augment/040103/',
                            model_name = 'epoch10/',
                            debug = False,
                            use_combine = True,
                            ):
                        
        
        self.class_dataset_dict = {}
        #resol_list = self.params.keys()
        #resol_list.sort(reverse = True)
        self.pts_xyz = None
        for i,param_dict in enumerate(self.params):
            print('resol', param_dict['resol'])
            print(param_dict)
            resol = param_dict['resol']
            ttt = time.time()
            #if((i == 0) and (resol > 6.0)):

            # if((i == -1)):
            #     if(addition_name != ''):
            #         self.class_dataset_dict[model, resol, addition_name] = test_dataset_dict[model, resol, addition_name]
            #     else:
            #         self.class_dataset_dict[model, resol] = test_dataset_dict[model, resol]
            if (i == 0):
                blur_ratio = None if (resol < 1.1) else int(resol/1.0)
                self.class_dataset_dict[model, resol] = Dataset(model_num = model, 
                                                            mode = 'init_volumn',
                                                            type_volumn = 'test',
                                                            load_type = 'new',
                                                            #viewList = self.viewList,
                                                            addition_name = '',
                                                            viewPair_mode = 'train',
                                                            blur_ratio = blur_ratio,
                                                            cube_D = param_dict['cube_D'],
                                                            #train_rejection = 'train_rejection',
                                                            train_rejection = 'None',
                                                            #train_rejection = 'multiscale',
                                                            sparsity = sparsity,
                                                            pts_xyz = self.pts_xyz,
                                                            thre_angle = 90,
                                                            selection_method = 'angle',
                                                            resol = resol)
            else:
                blur_ratio = None if (resol < 1.1) else int(resol/1.0)
                self.class_dataset_dict[model, resol] = Dataset(model_num = model, 
                                                            mode = 'init_volumn',
                                                            type_volumn = 'test',
                                                            load_type = 'new',
                                                            #viewList = self.viewList,
                                                            addition_name = '',
                                                            viewPair_mode = 'train',
                                                            blur_ratio = blur_ratio,
                                                            cube_D = param_dict['cube_D'],
                                                            #train_rejection = 'train_rejection',
                                                            #train_rejection = 'None',
                                                            train_rejection = 'multiscale',
                                                            sparsity = sparsity,
                                                            pts_xyz = self.pts_xyz,
                                                            thre_angle = 90,
                                                            selection_method = 'angle',
                                                            resol = resol)
            print('prepare data cost: ', time.time()-ttt)
            for tau_fg in param_dict['tau_fg']:
                ttt = time.time()
                fg.tau = nn.Parameter(torch.tensor(tau_fg).type(torch.cuda.FloatTensor))
                if(resol < 6.0):
                    N_viewPairs = N_viewPairs4inference
                else:
                    N_viewPairs = [5]

                self.pts_xyz = self.reconstruct(self.class_dataset_dict,
                                model = model, 
                                resol_list = [resol],
                                batch_size = param_dict['batch_size'], 
                                min_prob = param_dict['min_prob'], 
                                tau_list = param_dict['tau_list'], 
                                gamma_list = param_dict['gamma_list'],
                                use_fg = param_dict['use_fg'],
                                use_fg_new = param_dict['use_fg_new'],
                                use_superResol = param_dict['use_superResol'],
                                cube_D = param_dict['cube_D'],
                                N_viewPairs4inference = N_viewPairs,
                                enable_rayPooling = param_dict['enable_rayPooling'],
                                next_level = param_dict['next_level'],
                                sparsity = sparsity,
                                tau_fg = tau_fg,
                                addition_name = addition_name,
                                addition_name2 = addition_name2 + 'tauFG:'+str(tau_fg),
                                model_path = model_path,
                                model_name = 'multi_'+model_name,
                                debug = debug)
                
        if(use_combine):       
            self.rayPool(model = model,
                            resol = self.comp_params['resol'],
                            super_fg = self.comp_params['super_fg'],
                            min_fg = self.comp_params['min_fg'],
                            tau_list = self.comp_params['tau_list'], 
                            gamma_list = self.comp_params['gamma_list'], 
                            N_viewPairs4inference = N_viewPairs4inference,
                            model_path = model_path,
                            model_name = 'multi_'+model_name,
                            addition_name = addition_name,
                            addition_name2 = '')
        print('doing reconstruction cost',time.time()-ttt)

    
    
    def reconstruct(self,
                    test_dataset_dict,
                    model = 9, resol_list = [1.0],
                    batch_size = 4, min_prob = 0.5, 
                    tau_list = [0.7], gamma_list = [0.8],
                    tau_fg = 1.0,
                    next_level = None,
                    attention_type = 'none',
                    train_model = '',
                    use_fg = False,
                    use_fg_new = False,
                    use_ad = False,
                    use_mask = False,
                    use_superResol = False,
                    cube_D = None,
                    mode = 'init_volumn',
                    viewPair_mode = 'inference',
                    N_viewPairs4inference = [5],
                    thre_angle = 150,
                    type_volumn = 'test',
                    train_rejection = 'none',
                    selection_method = 'random',
                    enable_rayPooling = False,
                    sparsity = None,
                    addition_name = '',
                    addition_name2 = '',
                    model_path = './experiment/network/DTU/augment/040103/',
                    model_name = 'epoch10/',
                    draw = False,
                    draw_view = False,
                    debug = False):
        
        #prediction_list_total, rgb_list_total, vxl_ijk_list_total, rayPooling_votes_list_total = [], [], [], []
        #for model in model_list:
        self.load_modelSpecific_params(self._datasetName, model = model, cube_D = cube_D, sparsity = sparsity)
        for i,resol in enumerate(resol_list):
            
            print('resol', resol)
            print('start record time:')
            start_time = time.time()
            
            if(use_superResol):
                #addition_name = addition_name + 'super'
                addition_name2 = addition_name2 + '_super_'
            
            if(addition_name != ''):
                test_dataset = test_dataset_dict[model, resol, addition_name]
            else:
                test_dataset = test_dataset_dict[model, resol]
            
                
            if(self.load_viewPair_mode == 'dynamic'):
                torch.cuda.empty_cache()
                test_dataset.volumn._batchSize_viewPair_w = 1000000
                test_dataset.volumn.N_viewPairs4inference = N_viewPairs4inference
                test_dataset.volumn.viewPairSelection = ViewPairSelection(selection_method = selection_method, thre_angle = thre_angle)
                test_dataset.volumn.viewSelection(mode = viewPair_mode, eNet = eNet_p)
                

                torch.cuda.empty_cache()
                
            dense2sparse = Dense2Sparse()


            data_test = DataLoader(test_dataset, batch_size= batch_size,
                                        shuffle=False, num_workers=12)
            print('dataset prepare complete')
            
            if(next_level == 'previous'):
                print('load updated_sparse_list_np')
                prediction_list, rgb_list, vxl_ijk_list, rayPooling_votes_list, cube_ijk_np, param_np, viewPair_np = self.updated_sparse_list_np
            else:
                prediction_list, rgb_list, vxl_ijk_list, rayPooling_votes_list = [], [], [], []
                cube_ijk_np, param_np, viewPair_np = None, None, None

            for i, data in enumerate(data_test,0):
                if(debug):
                    if(i > 25):
                        continue
                print('start evaluate %r epoch'%i)
                self.data_o = data
                if(self._datasetName == 'MVS2d'):
                    batch_size_cvc,self.cubic_num,_,_,pixel_size = data['cvc'].size()
                    
                    w_list = torch.zeros([batch_size_cvc, cubic_num]).type(torch.cuda.FloatTensor)
                    s_list = torch.zeros([batch_size_cvc, cubic_num, 1,pixel_size,pixel_size ]).type(torch.cuda.FloatTensor)
                else:
                    batch_size_cvc,cubic_num,_,_,_,pixel_size = data['cvc'].size()
                    #cubic_num = 1
                    #w_list = torch.zeros([batch_size_cvc, cubic_num])
                    w_list = torch.ones([batch_size_cvc, cubic_num])
                    s_list = torch.zeros([batch_size_cvc, cubic_num, pixel_size,pixel_size,pixel_size ])

                if(train_model == 'attSurface'):
                    output = attSurface(data['cvc'].to(device))
                else:
                    output = 0
                    output_ad = 0
                    w_total = 0
                    output_fg_new = 0
                    
                    for i_c in range(cubic_num):
                    #for i_c in range(1):
                        if(self._datasetName == 'MVS2d'):
                            s = surfaceNet(data['cvc'][:,i_c,...].to(device)).detach()
                        else:
                            if(use_superResol):
                                data_cvc = self.superResol(data['cvc'][:,i_c,...].to(device), way = 'in')
                                #print('fuck ',data_cvc.shape)
                                s = surfaceNet(data_cvc).detach()
                                
                                s = self.superResol(s, way = 'out')[:,0].detach()
                            else:
                                s = surfaceNet(data['cvc'][:,i_c,...].to(device))[:,0].detach()
                            
                            #if(use_fg):
                            #    s = fg(s)[:,0].detach()
                            if(use_ad):
                                s_ad = ad(surfaceNet.s_fg.detach(), surfaceNet.s_fg_res.detach())[:,0].detach()
                            if(use_fg_new):
                                if(use_superResol):
                                    s = self.superResol(s[:,None,...], way = 'in')[:,0]
                                    s = fg(s)
                                    s_fg_new = self.superResol(s, way = 'out')[:,0].detach()
                                else:
                                    s_fg_new = fg(s)[:,0]
                        
                        if(attention_type == 'pixel'):
                            w = eNet(surfaceNet.s)[:,0].detach()
                            w_total += w
                            output += s * w
                            #w_total += w.sum((1,2,3))[...,None, None,None]
                            #output += s * w.sum((1,2,3))[...,None,None,None]
                            
                            #w_list[:,i_c] = w.sum((1,2,3)).cpu()
                            s_list[:,i_c] = output.cpu()
                            #print(w)
                            if(use_ad):
                                output_ad += s_ad * w.detach()
                        
                            
                        else:
                            w = eNet(data['embedding'][:,i_c,...].to(device)).detach()
                            if(draw_view):
                                fig, axes = plt.subplots(ncols = 2,nrows = 4)
                                e1 = data['embedding'][0,i_c,...]
                                p1 = e1[98:206].cpu().numpy().reshape(6,6,3)
                                p2 = e1[302:410].cpu().numpy().reshape(6,6,3)
                                
                                h1_r = e1[2:34].cpu().numpy()
                                h2_r = e1[206:238].cpu().numpy()
                                h1_g = e1[34:66].cpu().numpy()
                                h2_g = e1[238:270].cpu().numpy()
                                h1_b = e1[66:98].cpu().numpy()
                                h2_b = e1[270:302].cpu().numpy()
                                
                                axes[0,0].imshow(p1)
                                axes[0,1].imshow(p2)
                                axes[1,0].bar(range(32),h1_r)
                                axes[1,1].bar(range(32),h2_r)
                                axes[2,0].bar(range(32),h1_g)
                                axes[2,1].bar(range(32),h2_g)
                                axes[3,0].bar(range(32),h1_b)
                                axes[3,1].bar(range(32),h2_b)
                                
                                plt.show()
                                print('l2',((p1 - p2) ** 2).mean())
                                print('alpha', e1[0].cpu() * 180 / 3.14159)
                                print('h_r', np.abs(h1_r-h2_r).mean())
                                print('h_g', np.abs(h1_g-h2_g).mean())
                                print('h_b', np.abs(h1_b-h2_b).mean())
                                print('h_r', h1_r.var())
                                print('h_g', h1_g.var())
                                print('h_b', h1_b.var())
                            
                            w_list[:,i_c] = w[:,0].cpu()
                            if(use_fg_new):
                                s_list[:,i_c] = s_fg_new.cpu()
                            else:
                                s_list[:,i_c] = s.cpu()

                            w_total += w[...,None,None]
                            output += s * w[...,None,None]
                            if(use_ad):
                                output_ad += s_ad * w[...,None,None].detach()
                            if(use_fg_new):
                                output_fg_new += s_fg_new * w[...,None,None]
                        
                    output = output/(w_total + 1e-5)
                    #output = output/cubic_num
                    if(use_ad):
                        output_ad = output_ad/(w_total + 1e-5)
                    if(use_fg_new):
                        output_fg_new = output_fg_new/(w_total + 1e-5)
                    
                    
                if(use_fg):
                    if(use_fg_new):
                        output_old = output
                        output = output_fg_new
                        #output = output.pow(fg_tau) / (output.pow(fg_tau) + (1-output).pow(fg_tau))
                        if(use_mask):
                            output = self.threhold_adv(output)
                        
                    else:
                        #print('before fg',(output>min_prob).sum())
                        #print('before fg',(output).sum())
                        output_old = output
                        if(use_superResol):
                            o = self.superResol(output[:,None,...], way = 'in')[:,0]
                            o = fg(o)
                            output = self.superResol(o, way = 'out')[:,0].detach()
                        else:
                            output = fg(output)[:,0]
                        if(use_mask):
                            output = self.threhold_adv(output)
                        #print('after fg',(output>min_prob).sum())
                        #print('after fg',(output).sum())
                        #print('output', output.shape)
                        #output = output.pow(fg_tau) / (output.pow(fg_tau) + (1-output).pow(fg_tau))
                if(use_ad):
                    print('before fg',(output>min_prob).sum())
                    print('before fg',(output).sum())
                    output = output_ad
                    print('after fg',(output>min_prob).sum())
                    print('after fg',(output).sum())
                
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
                    #print('w_list_numpy',w_list_numpy)
                    #print('s_list_numpy',s_list_numpy)

                    cube_numpy = ((data['cvc'])).detach().cpu().numpy()
                    
                cube_numpy = 256 * (cube_numpy + 0.5)
                #cube_numpy = np.swapaxes(cube_numpy, 3, -1)
                new_cube = dense2sparse.generate_voxelLevelWeighted_coloredCubes(
                    viewPair_coloredCubes = cube_numpy, 
                    viewPair_surf_predictions = s_list_numpy, 
                    weight4viewPair = w_list_numpy
                )

                if(draw):
                    if(use_ad or use_fg):
                        print('start draw sample')
                        self.draw_sample(data, output[:,None], output_old[:,None], show_num = 1, use_fg = use_fg)
                        #self.draw_sample(data, output[:,None], output_old[:,None], show_num = 1, use_fg = use_fg, use_superResol = True)
                    
                    else:
                        self.draw_sample(data, output, show_num = 1)

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
                            min_prob = min_prob, 
                            rayPool_thresh = 0,
                            #rayPool_thresh = 0.5 * N_viewPairs4inference[0] * 2,
                            enable_centerCrop = True, 
                            cube_Dcenter = self._cube_Dcenter,
                            enable_rayPooling = enable_rayPooling, 
                            cameraPOs = test_dataset.cameraPOs_np, 
                            cameraTs = test_dataset.cameraTs_np, 
                            prediction_list = prediction_list, 
                            rgb_list = rgb_list, 
                            vxl_ijk_list = vxl_ijk_list, \
                            rayPooling_votes_list = rayPooling_votes_list, \
                            cube_ijk_np = cube_ijk_np, 
                            param_np = param_np, 
                            viewPair_np = viewPair_np)

                prediction_list, rgb_list, vxl_ijk_list, rayPooling_votes_list, cube_ijk_np, param_np, viewPair_np = updated_sparse_list_np
                
                
            #prediction_list_total.extend(prediction_list)
            #rgb_list_total.extend(rgb_list)
            #vxl_ijk_list_total.extend(vxl_ijk_list)
            #rayPooling_votes_list_total.extend(rayPooling_votes_list)
            
            print('end surfaceNet predict')
            print('pridict cost: ', time.time() - start_time)
            
            if(next_level == 'next'):
                self.updated_sparse_list_np = updated_sparse_list_np
            self.updated_sparse_dict[resol,use_superResol,tau_fg] = updated_sparse_list_np
            
            if not os.path.exists(model_path + model_name + 'model:%s'%str(model)):
                os.makedirs(model_path + model_name + 'model:%s'%str(model))
            npz_name = 'model:%s/ray:%s_min_prob:%s__resol:%s_fg:%s_ad:%s%s.npz'%(str(model),str(enable_rayPooling),str(min_prob),str(resol), str(use_fg), str(use_ad),addition_name+addition_name2)
            save_npz_file_path = model_path + model_name + npz_name
            dense2sparse.save_sparseCubes(save_npz_file_path, *updated_sparse_list_np)

            for tau in tau_list:
                for gamma in gamma_list:
                    print('tau', tau)
                    print('gamma', gamma)
                    vxl_mask_list = dense2sparse.filter_voxels(
                        vxl_mask_list=[],
                        prediction_list=prediction_list, 
                        prob_thresh= tau,
                        rayPooling_votes_list=rayPooling_votes_list, 
                        rayPool_thresh = gamma * N_viewPairs4inference[0] * 2) 

                    #vxl_maskDenoised_list = denoising.denoise_crossCubes(cube_ijk_np, 
                    #                                                     vxl_ijk_list, 
                    #                                                     vxl_mask_list = vxl_mask_list, 
                    #                                                     D_cube = self._cube_D)
                    vxl_maskDenoised_list = vxl_mask_list
                    if(enable_rayPooling):
                        if(viewPair_mode == 'inference'):
                            reconstruct_name = 'model:%s/ray:%s_tau:%s_gamma:%s__resol:%s_fg:%s_ad:%s%s.ply'%(str(model),str(enable_rayPooling),str(tau), str(gamma), str(resol), str(use_fg), str(use_ad) ,addition_name+addition_name2)
                        else:
                            reconstruct_name = 'model:%s/ray:%s_tau:%s_gamma:%s__resol:%s_fg:%s_ad:%s%s.ply'%(str(model),str(enable_rayPooling),str(tau), str(gamma), str(resol), str(use_fg),str(use_ad), addition_name+addition_name2)
                    else:
                        if(viewPair_mode == 'inference'):
                            reconstruct_name = 'model:%s/ray:%s_min_prob:%s__resol:%s_fg:%s_ad:%s%s.ply'%(str(model),str(enable_rayPooling),str(min_prob),str(resol), str(use_fg), str(use_ad),addition_name+addition_name2)
                        else:
                            reconstruct_name = 'model:%s/ray:%s_min_prob:%s__resol:%s_fg:%s_ad:%s%s.ply'%(str(model),str(enable_rayPooling),str(min_prob),str(resol), str(use_fg), str(use_ad),addition_name+addition_name2)

                    if not os.path.exists(model_path + model_name):
                        os.makedirs(model_path + model_name)

                    ply_filePath = model_path + model_name + reconstruct_name

                    #pdb.set_trace()
                    self.xyz_np, self.rgb_np, self.normal_np = dense2sparse.save_sparseCubes_2ply(vxl_maskDenoised_list, vxl_ijk_list, rgb_list, \
                                param_np, ply_filePath=ply_filePath, normal_list=None)
                    #self.xyz_dict[resol, tau] = self.xyz_np
                    #self.rgb_dict[resol, tau] = self.rgb_np

            
        #return self.xyzs, self.rgbs
        return self.xyz_np
    
    def rayPool(self,
                model = 9,
                resol = 0.4,
                super_fg = -0.6,
                min_fg = -0.9,
            tau_list = [0.5], 
            gamma_list = [1.0], 
            N_viewPairs4inference = [5],
            model_path = '',
            model_name = '',
            addition_name = '',
            addition_name2 = ''):
        
        prediction_list_1, rgb_list_1, vxl_ijk_list_1, rayPooling_votes_list_1, cube_ijk_np_1, param_np_1, viewPair_np_1 = self.updated_sparse_dict[resol,True,super_fg]
        prediction_list_2, rgb_list_2, vxl_ijk_list_2, rayPooling_votes_list_2, cube_ijk_np_2, param_np_2, viewPair_np_2 = self.updated_sparse_dict[resol,False,min_fg]
        
        #prediction_list_1
        prediction_list = prediction_list_1 + prediction_list_2
        rgb_list = rgb_list_1 + rgb_list_2
        vxl_ijk_list = vxl_ijk_list_1 + vxl_ijk_list_2
        rayPooling_votes_list = rayPooling_votes_list_1 + rayPooling_votes_list_2
        cube_ijk_np = np.concatenate((cube_ijk_np_1, cube_ijk_np_2), axis = 0)
        param_np = np.concatenate((param_np_1, param_np_2), axis = 0)
        viewPair_np = np.concatenate((viewPair_np_1, viewPair_np_1), axis = 0)
        
        dense2sparse = Dense2Sparse()
        #self._cube_D = 64
        
        for tau in tau_list:
            for gamma in gamma_list:
                print('tau', tau)
                print('gamma', gamma)
                vxl_mask_list = dense2sparse.filter_voxels(
                    vxl_mask_list=[],
                    prediction_list=prediction_list, 
                    prob_thresh= tau,
                    rayPooling_votes_list=rayPooling_votes_list, 
                    rayPool_thresh = gamma * N_viewPairs4inference[0] * 2) 

                vxl_maskDenoised_list = denoising.denoise_crossCubes(cube_ijk_np, 
                                                                     vxl_ijk_list, 
                                                                     vxl_mask_list = vxl_mask_list, 
                                                                     D_cube = self._cube_D)########################
                vxl_maskDenoised_list = vxl_mask_list
                reconstruct_name = 'model:%s/comb__tau:%s_gamma:%s_%s.ply'%(str(model),str(tau), str(gamma),addition_name+addition_name2)
                    
                if not os.path.exists(model_path + model_name):
                    os.makedirs(model_path + model_name)

                ply_filePath = model_path + model_name + reconstruct_name

               
                dense2sparse.save_sparseCubes_2ply(vxl_maskDenoised_list, vxl_ijk_list, rgb_list, \
                            param_np, ply_filePath=ply_filePath, normal_list=None)
               
    
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
    
    def threholdSurf(self, data, threshold = 0.5):
    
        threshold_surf = (data >= threshold)
        threshold_surf = threshold_surf * 1.0
        return threshold_surf
    
    def threhold_adv(self, data, threshold_1 = 0.2, mean_surf = 0.5):
        
        batch_size = data.shape[0]
        len_shape = len(data.shape)
        mask = (data >= threshold_1).type(torch.cuda.FloatTensor)
        mask_surf = mask * data
        mean_mask_surf = mask_surf.view(batch_size,-1).sum(dim = 1) / mask.view(batch_size,-1).sum(dim = 1)
        if(len_shape == 4):
            mask_surf = (mask_surf * 0.5) / mean_mask_surf[:,None,None,None]
        elif(len_shape == 5):
            mask_surf = (mask_surf * 0.5) / mean_mask_surf[:,None,None,None,None]
        
        return mask_surf
    
    def superResol(self, inputs, way):
        
        #dim = input.shape[-1]
        if(way == 'in'):
            a1 = nn.functional.interpolate
            out = a1(inputs, scale_factor = 0.5)
            #out = o1(inputs)
            return out
        elif(way == 'out'):
            o1 = nn.functional.interpolate
            out = o1(inputs, scale_factor = 2.0)
            return out
    
    
    
    def draw_sample(self, 
        data, 
        output, 
        output_old = None,
        cvc_num = 1,
        show_num = 4,
        save_image = False, 
        use_fg = False,
        train_model = 'none',
        file_root = 'labData/',
        mode = 'train',
        epoch_num = 10000,
        model_num = 10000,
        use_superResol = False,
        detach = False):

        if(self._datasetName != 'MVS2d'):
            batch_size, view_num, _,_,_,image_size = data['cvc'].shape

        if(cvc_num > view_num):
            cvc_num = view_num
        if(show_num > batch_size):
            show_num = batch_size
        
        
        if(detach):
            cvc_3d = data['cvc'].detach().numpy()
            embedding = data['embedding'].detach().numpy()
            surface_3d = data['surface'].detach().numpy()
            idxs = data['idx_validCubes'].detach()
        else:
            print(use_superResol)
            if(use_superResol):
                cvc_3d = np.zeros((batch_size,view_num,6,image_size/2,image_size/2,image_size/2))
                for i in range(view_num):
                    cvc_3d[:,i] = self.superResol(data['cvc'].to(device)[:,i], way = 'in').detach().cpu().numpy()
                    
            else:
                cvc_3d = data['cvc'].numpy()
            embedding = data['embedding'].numpy()
            surface_3d = data['surface'].numpy()
            idxs = data['idx_validCubes']
        
        if(use_fg):
            output_old_3d = output_old.cpu().detach().numpy()
        
        #output = self.threhold_adv(output)
        output_3d = output.cpu().detach().numpy()
        output_surf_3d = self.threholdSurf(output_3d,threshold = 0.7)
        
        for layer_h in [2,5,8,11,14]:
            cvc = cvc_3d[...,layer_h]
            surface = surface_3d[...,layer_h]
            output = output_3d[...,layer_h]
            output_surf = output_surf_3d[...,layer_h]
            if(use_fg):
                output_surf_old_3d = self.threholdSurf(output_old_3d,threshold = 0.5)
                output_old = output_old_3d[...,layer_h]
                output_surf_old = output_surf_old_3d[...,layer_h]

            for i in range(show_num):
                idx = idxs[i].item()

                surface_one = surface[i,0]
                show_length = 2
                fig, axes = plt.subplots(ncols = (cvc_num * show_length + 5))


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
                    axes[cvc_num * show_length + 2].imshow(output_surf_old[i,0], cmap = plt.cm.gray)
                    axes[cvc_num * show_length + 3].imshow(output[i,0], cmap = plt.cm.gray)
                    axes[cvc_num * show_length + 4].imshow(output_surf[i,0], cmap = plt.cm.gray)
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
                    #plt.savefig(path)
                    plt.show()
                    #plt.clf()
                else:
                    path_dirc = file_root + 'test/model_%r'%(model_num)
                    if not os.path.exists(path_dirc):
                        os.makedirs(path_dirc)
                    path = file_root + 'test/model_%r/epoch_%s__cvc_id_%r__show_id_%r.png'%(model_num, str(epoch_num).zfill(3), idx, i)
                    #plt.savefig(path)
                    plt.show()
                    plt.clf()
reconstruct = Reconstruct()
if __name__ == '__main__':

    torch.cuda.device_count()
    dataset_dict = {}
    model_lists = [9]
   
  
    
    use_cuda = True
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    surfaceNet = SurfaceNet_3d_old().to(device)

    eNet_p = EmbeddingNet_3d_new_big().to(device)
    attention_type = 'new_big'
   
    
    eNet = EmbeddingNet_3d_new_big().to(device)
    print('new_big')

    fg = FineGenerator_3d_res(residual_type = 'res_detach', activate_type = 'new').to(device)
    torch.cuda.empty_cache()

    epoch_name = 'epoch3'
    model_path = ''
    load_model(model_path = './experiment/network/' + model_path,
                       model_name2 = 'eNet_' + epoch_name,
                       model_name5 = 'eNet_' + epoch_name,
                       train_model = 'none')


    
    epoch_name = 'epoch5'
    model_path = ''
    #model_path = 'DTU/Wgan/053001/'
    load_model(model_path = './experiment/network/' + model_path,
                       model_name1 = 'SurfaceNet_' + epoch_name, 
                       model_name3 = 'fgNet_' + epoch_name,
                       train_model = 'none')
   
    reconstruct.params = [
                        
                        {'resol':4.8,
                            'use_superResol':True,
                           'min_prob':0.4, 
                           'tau_list':[0.4],
                           'gamma_list':[0.5],
                           'tau_fg':[-0.6],
                           'use_fg':True,
                           'use_fg_new':False,
                           'enable_rayPooling':True,

                           'cube_D':64,
                           'batch_size':10,
                           'next_level':'none'},
                       {'resol':1.2,
                            'use_superResol':True,
                           'min_prob':0.5, 
                           'tau_list':[0.5],
                           'gamma_list':[0.5],
                           'tau_fg':[-0.6],
                           'use_fg':True,
                           'use_fg_new':False,
                           'enable_rayPooling':True,

                           'cube_D':64,
                           'batch_size':10,
                           'next_level':'none'},
                        {'resol':0.6,
                            'use_superResol':True,
                           'min_prob':0.7, 
                           'tau_list':[0.7],
                           'gamma_list':[0.5],
                           'tau_fg':[-0.6],
                           'use_fg':True,
                           'use_fg_new':False,
                           'enable_rayPooling':True,

                           'cube_D':64,
                           'batch_size':10,
                           'next_level':'none'},
                    ]

    
   
    
  
   
    for N_view in [3]:
        for model in model_lists:
            for sparsity in [7]:
                for density in [1]:
                    reconstruct.multi_reconstruct( 
                                                test_dataset_dict = dataset_dict,
                                                model = model, 
                                                N_viewPairs4inference = [N_view],
                                                sparsity = sparsity,
                                                addition_name = '',
                                                addition_name2 = '__N_view:'+str(N_view),
                                                model_path = './experiment/output/',
                                                model_name = '',
                                                debug = False,
                                                use_combine = False,
                                                )
    