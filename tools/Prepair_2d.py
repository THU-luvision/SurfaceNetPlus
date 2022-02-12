from Parameter_2d import Params
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
import torch.optim as optim

import sys
sys.path.append("./utils")
import earlyRejection
import image
import utils
import time


class SimilarityNet():
    def __init__(self):
        pass


import os
import cPickle as pickle
import rayPooling
import sys
#import camera
from plyfile import PlyData, PlyElement


import scipy.misc

def save_cycle_png(model_list, batch_size):
    
    for model_num in model_list:

        dataset = Dataset(model_num)
        #batch_size = 64

        dataloader = DataLoader(dataset, batch_size= batch_size,
                                shuffle=False, num_workers=8)

        for i_data, data in enumerate(dataloader, 0):
            #print(data['cvc'].shape)
            batch_num, cubic_num,_,_,_ = data['cvc'].size()
            output = 0
            w_total = 0
            for i_c in range(cubic_num):
            #for i_c in range(1):
                s = surfaceNet(data['cvc'][:,i_c,...].to(device))
                w = eNet(data['embedding'][:,i_c,...].to(device))

                #w = w.detach()
                #s = s.detach()
                #w_list.append(w)
                w_total += (w[...,None,None])

                output += s * w[...,None,None]

            #w_total = w_total.detach()
            output = output/(w_total + 1e-15)

            output_cpu = output.detach().cpu().numpy()
            surface_cpu = data['surface'].numpy()

            #print(surface_cpu.shape)
            for i_batch_num in range(batch_num):
                data_root = './cyclegan'
                data_name_truth =  data_root + '/truth/' + str(model_num) + '_' + str(i_data * batch_size + i_batch_num)
                data_name_output = data_root + '/output/' + str(model_num) + '_' + str(i_data * batch_size + i_batch_num)
                scipy.misc.imsave(data_name_truth + '.png', surface_cpu[i_batch_num,0])
                scipy.misc.imsave(data_name_output + '.png', output_cpu[i_batch_num,0])
                
#save_cycle_png(model_list = model_list, batch_size = 32)

class Image(Params):
    def __init__(self):
        super(Image, self).__init__()
        pass
        
    
    def readImages(self, 
                   datasetFolder, 
                   imgNamePattern, 
                   viewList, 
                   return_list = True):
        """
        Only select the images of the views listed in the viewList.
        We assume that the view index is large or equal than 0
            &&
            the images' sizes are equal.

        ---------
        inputs:
            datasetFolder: where the dataset locates
            imgNamePattern: different dataset have different name patterns for images. Remember to include the subdirecteries, e.g. "x/x/xx.png"
                    Replace '#' --> '{:03}'; '@' --> '{}'
            viewList: list the view index, such as [11, 1, 30, 6]
            return_list: True.  Return list if true else np.

        ---------
        outputs:
            imgs_list: list of the images
                or
            imgs_np: np array with shape of (len(viewList), img_h, img_w, 3)

        ---------
        usages:
        >>> imgs_np = readImages(".", "test/Lasagne0#.jpg", [6,6], return_list = False)     # doctest need to run in the local dir
        loaded img ./test/Lasagne0006.jpg
        loaded img ./test/Lasagne0006.jpg
        >>> imgs_np.shape
        (2, 225, 225, 3)
        """

        imgs_list = []

        for i, viewIndx in enumerate(viewList):
            # we assume the name pattern looks like 'x/x/*001*.xxx', if {:04}, add one 0 in the pattern: '*0#*.xxx'
            imgPath = os.path.join(datasetFolder, imgNamePattern.replace('#', '{:03}'.format(viewIndx)).replace('@', '{}'.format(viewIndx))) 
            img = scipy.misc.imread(imgPath)    # read as np array
            imgs_list.append(img)
            #print('loaded img ' + imgPath)

        return imgs_list if return_list else np.stack(imgs_list)

class Scene():
    def __init__(self):
        pass

    def initializeCubes(self,resol, cube_D, cube_Dcenter, cube_overlapping_ratio, BB):
        """
        generate {N_cubes} 3D overlapping cubes, each one has {N_cubeParams} embeddings
        for the cube with size of cube_D^3 the valid prediction region is the center part, say, cube_Dcenter^3
        E.g. cube_D=32, cube_Dcenter could be = 20. Because the border part of each cubes don't have accurate prediction because of ConvNet.

        ---------------
        inputs:
            resol: resolusion of each voxel in the CVC (mm)
            cube_D: size of the CVC (Colored Voxel Cube)
            cube_Dcenter: only keep the center part of the CVC, because of the boundery effect of ConvNet.
            cube_overlapping_ratio: pertantage of the CVC are covered by the neighboring ones
            BB: bounding box, numpy array: [[x_min,x_max],[y_min,y_max],[z_min,z_max]]
        outputs: 
            cubes_param_np: (N_cubes, N_params) np.float32
            cube_D_mm: scalar

        ---------------
        usage:
        >>> cubes_param_np, cube_D_mm = initializeCubes(resol=1, cube_D=22, cube_Dcenter=10, cube_overlapping_ratio=0.5, BB=np.array([[3,88],[-11,99],[-110,-11]]))
        xyz bounding box of the reconstructed scene: [ 3 88], [-11  99], [-110  -11]
        >>> print cubes_param_np[:3] 
        [([   3.,  -11., -110.], [0, 0, 0],  1.)
         ([   3.,  -11., -105.], [0, 0, 1],  1.)
         ([   3.,  -11., -100.], [0, 0, 2],  1.)]
        >>> print cubes_param_np['xyz'][18:22]
        [[   3.  -11.  -20.]
         [   3.  -11.  -15.]
         [   3.   -6. -110.]
         [   3.   -6. -105.]]
        >>> np.allclose(cubes_param_np['xyz'][18:22], cubes_param_np[18:22]['xyz'])
        True
        >>> print cube_D_mm
        22
        """

        cube_D_mm = resol * cube_D   # D size of each cube along each axis, 
        cube_Center_D_mm = resol * cube_Dcenter   # D size of each cube's center that is finally remained 
        cube_stride_mm = cube_Center_D_mm * cube_overlapping_ratio # the distance between adjacent cubes, 
        safeMargin = (cube_D_mm - cube_Center_D_mm)/2

        print('xyz bounding box of the reconstructed scene: {}, {}, {}'.format(*BB))
        N_along_axis = lambda _min, _max, _resol: int(math.ceil((_max - _min) / _resol))
        N_along_xyz = [N_along_axis( (BB[_axis][0] - safeMargin), (BB[_axis][1] + safeMargin), cube_stride_mm) for _axis in range(3)]   # how many cubes along each axis
        # store the ijk indices of each cube, in order to localize the cube
        cubes_ijk = np.indices(tuple(N_along_xyz))
        N_cubes = cubes_ijk.size / 3   # how many cubes

        cubes_param_np = np.empty((N_cubes,), dtype=[('xyz', np.float32, (3,)), ('ijk', np.uint32, (3,)), ('resol', np.float32)])    # attributes for each CVC (colored voxel cube)
        cubes_param_np['ijk'] = cubes_ijk.reshape([3,-1]).T  # i/j/k grid index
        cubes_xyz_min = cubes_param_np['ijk'] * cube_stride_mm + (BB[:,0][None,:] - safeMargin)
        cubes_param_np['xyz'] = cubes_xyz_min    # x/y/z coordinates (mm)
        cubes_param_np['resol'] = resol

        return cubes_param_np, cube_D_mm

    def quantizePts2Cubes(self,pts_xyz, resol, cube_D, cube_Dcenter, cube_overlapping_ratio, BB = None):
        """
        generate overlapping cubes covering a set of points which is denser, so that we need to quantize the pts' coords.

        --------
        inputs:
            pts_xyz: generate the cubes around these pts
            resol: resolusion of each voxel in the CVC (mm)
            cube_D: size of the CVC (Colored Voxel Cube)
            cube_Dcenter: only keep the center part of the CVC, because of the boundery effect of ConvNet.
            cube_overlapping_ratio: pertantage of the CVC are covered by the neighboring ones
            BB: bounding box, numpy array: [[x_min,x_max],[y_min,y_max],[z_min,z_max]]

        --------
        outputs: 
            cubes_param_np: (N_cubes, N_params) np.float32
            cube_D_mm: scalar

        --------
        examples:
        >>> pts_xyz = np.array([[-1, 2, 0], [0, 2, 0], [1, 2, 0], [0,1,0], [0,0,0], [1,0,0], [2.1,0,0]])
        >>> #TODO quantizePts2Cubes(pts_xyz, resol=2, cube_D=3, cube_Dcenter = 2, cube_overlapping_ratio = 0.5)
        """

        cube_D_mm = resol * cube_D   # D size of each cube along each axis, 
        cube_Center_D_mm = resol * cube_Dcenter   # D size of each cube's center that is finally remained 
        cube_stride_mm = cube_Center_D_mm * cube_overlapping_ratio # the distance between adjacent cubes, 
        if BB is not None:
            safeMargin = cube_D_mm/2 # a little bit bigger than the BB
            inBB = np.array([np.logical_and(pts_xyz[:, _axis] >= (BB[_axis, 0] - safeMargin), pts_xyz[:, _axis] <= (BB[_axis, 1] + safeMargin)) for _axis in range(3)]).all(axis=0)
            pts_xyz = pts_xyz[inBB]
        shift = pts_xyz.min(axis=0)[None,...] # (1, 3), make sure the cube_ijk is non-negative, and try to cover the pts in the middle of the cubes.
        cubes_ijk_floor = (pts_xyz - shift) // cube_stride_mm # (N_pts, 3)
        cubes_ijk_ceil = ((pts_xyz - shift) // cube_stride_mm + 1)  # for each pt consider 2 neighboring cubesalong each axis.
        cubes_ijk = np.vstack([cubes_ijk_floor, cubes_ijk_ceil])  # (2*N_pts, 3)
        cubes_ijk_1d = cubes_ijk.view(dtype=cubes_ijk.dtype.descr * 3)
        cubes_ijk_unique = np.unique(cubes_ijk_1d).view(cubes_ijk_floor.dtype).reshape((-1, 3))  # (N_cubes, 3)
        N_cubes = cubes_ijk_unique.shape[0]   # how many cubes

        cubes_param_np = np.empty((N_cubes,), dtype=[('xyz', np.float32, (3,)), ('ijk', np.uint32, (3,)), ('resol', np.float32)])    # attributes for each CVC (colored voxel cube)
        cubes_param_np['ijk'] = cubes_ijk_unique  # i/j/k grid index
        cubesCenter_xyz = cubes_param_np['ijk'] * cube_stride_mm + shift
        cubes_param_np['xyz'] = cubesCenter_xyz - cube_D_mm/2    # (N_cubes, 3) min of x/y/z coordinates (mm)
        cubes_param_np['resol'] = resol

        return cubes_param_np, cube_D_mm


    def readPointCloud_xyz(self,pointCloudFile = 'xx/xx.ply'):
        pcd = PlyData.read(pointCloudFile)  # pcd for Point Cloud Data
        pcd_xyz = np.c_[pcd['vertex']['x'], pcd['vertex']['y'], pcd['vertex']['z']]
        return pcd_xyz

    def readBB_fromModel(self,objFile = 'xx/xx.obj'):
        mesh = mesh_util.load_obj(filename= objFile)
        BB = np.c_[mesh.v.min(axis=0), mesh.v.max(axis=0)]  # (3, 2)
        return BB


class Camera():
    def __init__(self):
        self.test =1
    
    def __readCameraPO_as_np_DTU__(self,cameraPO_file):
        """ 
        only load a camera PO in the file
        ------------
        inputs:
            cameraPO_file: the camera pose file of a specific view
        outputs:
            cameraPO: np.float64 (3,4)
        ------------
        usage:
        >>> p = __readCameraPO_as_np_DTU__(cameraPO_file = './test/cameraPO/pos_060.txt') 
        >>> p # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        array([[  1.67373847e+03,  -2.15171320e+03,   1.26963515e+03,
            ...
                  6.58552305e+02]])
        """
        cameraPO = np.loadtxt(cameraPO_file, dtype=np.float64, delimiter = ' ')
        return cameraPO
    
    def __readCameraPOs_as_np_Middlebury__(self, cameraPO_file, viewList):
        """ 
        load camera POs of multiple views in one file
        ------------
        inputs:
            cameraPO_file: the camera pose file of a specific view
            viewList: view list 
        outputs:
            cameraPO: np.float64 (N_views,3,4)
        ------------
        usage:
        >>> p = __readCameraPOs_as_np_Middlebury__(cameraPO_file = './test/cameraPO/dinoSR_par.txt', viewList=[3,8]) 
        >>> p # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
            array([[[ -1.22933223e+03,   3.08329199e+03,   2.02784015e+02,
            ...
            6.41227584e-01]]])
        """
        with open(cameraPO_file) as f:
            lines = f.readlines() 

        cameraPOs = np.empty((len(lines), 3, 4)).astype(np.float64)
        for _n, _l in enumerate(lines):
            if _n == 0:
                continue
            _params = np.array(_l.strip().split(' ')[1:], dtype=np.float64) 
            _K = _params[:9].reshape((3,3))
            _R = _params[9:18].reshape((3,3))
            _t = _params[18:].reshape((3,1))
            cameraPOs[_n] = np.dot(_K, np.c_[_R,_t])
        return cameraPOs[viewList]
    
    def readCameraPOs_as_np(self, 
                            datasetFolder,
                            datasetName,
                            poseNamePattern,
                            #model,
                            viewList):
        """
        inputs:
          datasetFolder: 'x/x/x/middlebury'
          datasetName: 'DTU' / 'Middlebury'
          #model: 1..128 / 'dinoxx'
          viewList: [3,8,21,...]
        output:
          cameraPOs (N_views,3,4) np.flost64
        """
        cameraPOs = np.empty((len(viewList),3,4), dtype=np.float64)

        if 'Middlebury' in datasetName:
            cameraPOs = self.__readCameraPOs_as_np_Middlebury__(cameraPO_file = os.path.join(datasetFolder, poseNamePattern), viewList=viewList)
        else: # cameraPOs are stored in different files
            for _i, _view in enumerate(viewList):
                # if 'DTU' in datasetName:
                _cameraPO = self.__readCameraPO_as_np_DTU__(cameraPO_file = os.path.join(datasetFolder, poseNamePattern.replace('#', '{:03}'.format(_view)).replace('@', '{}'.format(_view))))
                cameraPOs[_i] = _cameraPO
        return cameraPOs
    
    def __cameraP2T__(self, cameraPO):
        """
        cameraPO: (3,4)
        return camera center in the world coords: cameraT (3,0)
        >>> P = np.array([[798.693916, -2438.153488, 1568.674338, -542599.034996], \
                      [-44.838945, 1433.912029, 2576.399630, -1176685.647358], \
                      [-0.840873, -0.344537, 0.417405, 382.793511]])
        >>> t = np.array([555.64348632032, 191.10837560939, 360.02470478273])
        >>> np.allclose(__cameraP2T__(P), t)
        True
        """
        homo4D = np.array([np.linalg.det(cameraPO[:,[1,2,3]]), -1*np.linalg.det(cameraPO[:,[0,2,3]]), np.linalg.det(cameraPO[:,[0,1,3]]), -1*np.linalg.det(cameraPO[:,[0,1,2]]) ])
        cameraT = homo4D[:3] / homo4D[3]
        return cameraT

    
    def cameraPs2Ts(self, cameraPOs):
        """
        convert multiple POs to Ts. 
        ----------
        input:
            cameraPOs: list / numpy 
        output:
            cameraTs: list / numpy
        """
        if type(cameraPOs) is list:
            N = len(cameraPOs)
        else:                
            N = cameraPOs.shape[0]
        cameraT_list = []    
        for _cameraPO in cameraPOs:
            cameraT_list.append(self.__cameraP2T__(_cameraPO))

        return cameraT_list if type(cameraPOs) is list else np.stack(cameraT_list)
    
    def calculate_angle_p1_p2_p3(self,p1,p2,p3,return_angle=True, return_cosine=True):
        """
        calculate angle <p1,p2,p3>, which is the angle between the vectors p2p1 and p2p3 

        Parameters
        ----------
        p1/p2/p3: numpy with shape (3,)
        return_angle: return the radian angle
        return_cosine: return the cosine value

        Returns
        -------
        angle, cosine

        Examples
        --------
        """
        unit_vector = lambda v: v / np.linalg.norm(v)
        angle = lambda v1,v2: np.arccos(np.clip(np.dot(unit_vector(v1), unit_vector(v2)), -1.0, 1.0))
        cos_angle = lambda v1,v2: np.clip(np.dot(unit_vector(v1), unit_vector(v2)), -1.0, 1.0)

        vect_p2p1 = p1-p2
        vect_p2p3 = p3-p2
        return angle(vect_p2p1, vect_p2p3) if return_angle else None , \
                cos_angle(vect_p2p1, vect_p2p3) if return_cosine else None


    def k_combination_np(self, iterable, k = 2):
        """
        list all the k-combination along the output rows:
        input: [2,5,8], list 2-combination to a numpy array
        output: np.array([[2,5],[2,8],[5,8]])

        ----------
        usages:
        >>> k_combination_np([2,5,8])
        array([[2, 5],
               [2, 8],
               [5, 8]])
        >>> k_combination_np([2,5,8]).dtype
        dtype('int64')
        >>> k_combination_np([2.2,5.5,8.8,9.9], k=3)
        array([[ 2.2,  5.5,  8.8],
               [ 2.2,  5.5,  9.9],
               [ 2.2,  8.8,  9.9],
               [ 5.5,  8.8,  9.9]])
        """
        combinations = []
        for _combination in itertools.combinations(iterable, k):
            combinations.append(_combination)
        return np.asarray(combinations) 
    
    def viewPairAngles_wrt_pts(self,cameraTs, pts_xyz):
        """
        given a set of camera positions and a set of points coordinates, output the angle between camera pairs w.r.t. each 3D point.

        -----------
        inputs:
            cameraTs: (N_views, 3) camera positions
            pts_xyz: (N_pts, 3) 3D points' coordinates

        -----------
        outputs:
            viewPairAngle_wrt_pts: (N_pts, N_viewPairs) angle 

        -----------
        usages:
        >>> pts_xyz = np.array([[0,0,0],[1,1,1]], dtype=np.float32)     # p1 / p2
        >>> cameraTs = np.array([[0,0,1], [0,1,1], [1,0,1]], dtype=np.float32)      # c1/2/3
        >>> viewPairAngles_wrt_pts(cameraTs, pts_xyz) * 180 / math.pi    # output[i]: [<c1,pi,c2>, <c1,pi,c3>, <c2,pi,c3>]
        array([[ 45.,  45.,  60.],
               [ 45.,  45.,  90.]], dtype=float32)
        """

        unitize_array = lambda array, axis: array/np.linalg.norm(array, axis=axis, ord=2, keepdims=True)
        calc_arccos = lambda cos_values: np.arccos(np.clip(cos_values, -1.0, 1.0))  # TODO does it need clip ?
        N_views = cameraTs.shape[0]
        vector_pts2cameras = pts_xyz[:,None,:] - cameraTs[None,...]   # (N_pts, 1, 3) - (1, N_views, 3) ==> (N_pts, N_views, 3)
        unit_vector_pts2cameras = unitize_array(vector_pts2cameras, axis = -1)    # (N_pts, N_views, 3)  unit vector along axis=-1

        # do the matrix multiplication for the (N_pats,) tack of (N_views, 3) matrixs 
        ## (N_pts, N_views, 3) * (N_pts, 3, N_views) ==> (N_pts, N_views, N_views)
        # viewPairCosine_wrt_pts = np.matmul(unit_vector_pts2cameras, unit_vector_pts2cameras.transpose((0,2,1)))
        viewPairs = self.k_combination_np(range(N_views), k = 2)     # (N_combinations, 2)
        viewPairCosine_wrt_pts = np.sum(np.multiply(unit_vector_pts2cameras[:, viewPairs[:,0]], unit_vector_pts2cameras[:, viewPairs[:,1]]), axis=-1)    # (N_pts, N_combinations, 3) elementwise multiplication --> (N_pts, N_combinations) sum over the last axis
        viewPairAngle_wrt_pts = calc_arccos(viewPairCosine_wrt_pts)     # (N_pts, N_combinations)
        return viewPairAngle_wrt_pts
    
    #def viewPairAngles_p0s_pts(self, projection_M, )
    
    def perspectiveProj(self, 
                        projection_M, 
                        xyz_3D, 
                        return_int_hw = True, 
                        return_depth = False):
        """ 
        perform perspective projection from 3D points to 2D points given projection matrix(es)
                support multiple projection_matrixes and multiple 3D vectors
        notice: [matlabx,matlaby] = [width, height]

        ----------
        inputs:
        projection_M: numpy with shape (3,4) / (N_Ms, 3,4), during calculation (3,4) will --> (1,3,4)
        xyz_3D: numpy with shape (3,) / (N_pts, 3), during calculation (3,) will --> (1,3)
        return_int_hw: bool, round results to integer when True.

        ----------
        outputs:
        img_h, img_w: (N_pts,) / (N_Ms, N_pts)

        ----------
        usages:

        inputs: (N_Ms, 3,4) & (N_pts, 3), return_int_hw = False/True

        >>> np.random.seed(201611)
        >>> Ms = np.random.rand(2,3,4)
        >>> pts_3D = np.random.rand(2,3)
        >>> pts_2Dh, pts_2Dw = perspectiveProj(Ms, pts_3D, return_int_hw = False)
        >>> np.allclose(pts_2Dw, np.array([[ 1.35860185,  0.9878389 ],
        ...        [ 0.64522543,  0.76079278 ]]))
        True
        >>> pts_2Dh_int, pts_2Dw_int = perspectiveProj(Ms, pts_3D, return_int_hw = True)
        >>> np.allclose(pts_2Dw_int, np.array([[1, 1], [1, 1]]))
        True

        inputs: (3,4) & (3,)

        >>> np.allclose(
        ...         np.r_[perspectiveProj(Ms[1], pts_3D[0], return_int_hw = False)],
        ...         np.stack((pts_2Dh, pts_2Dw))[:,1,0])
        True
        """
        self.projection_M = projection_M
        
        if projection_M.shape[-2:] != (3,4):
            raise ValueError("perspectiveProj needs projection_M with shape (3,4), however got {}".format(projection_M.shape))

        if xyz_3D.ndim == 1:
            xyz_3D = xyz_3D[None,:]

        if xyz_3D.shape[1] != 3 or xyz_3D.ndim != 2:
            raise ValueError("perspectiveProj needs xyz_3D with shape (3,) or (N_pts, 3), however got {}".format(xyz_3D.shape))
        # perspective projection
        N_pts = xyz_3D.shape[0]
        xyz1 = np.c_[xyz_3D, np.ones((N_pts,1))].astype(np.float64) # (N_pts, 3) ==> (N_pts, 4)
        pts_3D = np.matmul(projection_M, xyz1.T) # (3, 4)/(N_Ms, 3, 4) * (4, N_pts) ==> (3, N_pts)/(N_Ms,3,N_pts)
        # the result is vector: [w,h,1], w is the first dim!!! (matlab's x/y/1')
        pts_2D = pts_3D[...,:2,:]
        self.pts_3D = pts_3D
        pts_2D /= pts_3D[...,2:3,:] # (2, N_pts) /= (1, N_pts) | (N_Ms, 2, N_pts) /= (N_Ms, 1, N_pts)
        self.pts_2D = pts_2D
        #print(self.pts_2D)
        if return_int_hw: 
            pts_2D = pts_2D.round().astype(np.int64)  # (2, N_pts) / (N_Ms, 2, N_pts)
        img_w, img_h = pts_2D[...,0,:], pts_2D[...,1,:] # (N_pts,) / (N_Ms, N_pts)
        if return_depth:
            depth = pts_3D[...,2,:]
            return img_h, img_w, depth
        return img_h, img_w
    
    def perspectiveProj_cubesCorner(self, projection_M, cube_xyz_min, cube_D_mm, return_int_hw = True, return_depth = False):
        """ 
        perform perspective projection from 3D points to 2D points given projection matrix(es)
                support multiple projection_matrixes and multiple 3D vectors
        notice: [matlabx,matlaby] = [width, height]

        ----------
        inputs:
        projection_M: numpy with shape (3,4) / (N_Ms, 3,4), during calculation (3,4) will --> (1,3,4)
        cube_xyz_min: numpy with shape (3,) / (N_pts, 3), during calculation (3,) will --> (1,3)
        cube_D_mm: cube with shape D^3
        return_int_hw: bool, round results to integer when True.
        return_depth: bool

        ----------
        outputs:
        img_h, img_w: (N_Ms, N_pts, 8)

        ----------
        usages:

        inputs: (N_Ms, 3, 4) & (N_pts, 3), return_int_hw = False/True, outputs (N_Ms, N_pts, 8)

        >>> np.random.seed(201611)
        >>> Ms = np.random.rand(2,3,4)
        >>> pts_3D = np.random.rand(2,3)
        >>> pts_2Dh, pts_2Dw = perspectiveProj_cubesCorner(Ms, pts_3D, cube_D_mm = 1, return_int_hw = False)
        >>> np.allclose(pts_2Dw[:,:,0], np.array([[ 1.35860185,  0.9878389 ],
        ...        [ 0.64522543,  0.76079278 ]]))
        True
        >>> pts_2Dh_int, pts_2Dw_int = perspectiveProj_cubesCorner(Ms, pts_3D, cube_D_mm = 1, return_int_hw = True)
        >>> np.allclose(pts_2Dw_int[:,:,0], np.array([[1, 1], [1, 1]]))
        True

        inputs: (3,4) & (3,), outputs (1,1,8)

        >>> np.allclose(
        ...         perspectiveProj_cubesCorner(Ms[1], pts_3D[0], cube_D_mm = 1, return_int_hw = False)[0],
        ...         pts_2Dh[1,0])        # (1,1,8)
        True
        """

        if projection_M.shape[-2:] != (3,4):
            raise ValueError("perspectiveProj needs projection_M with shape (3,4), however got {}".format(projection_M.shape))

        if cube_xyz_min.ndim == 1:
            cube_xyz_min = cube_xyz_min[None,:]     # (3,) --> (N_pts, 3)

        if cube_xyz_min.shape[1] != 3 or cube_xyz_min.ndim != 2:
            raise ValueError("perspectiveProj needs cube_xyz_min with shape (3,) or (N_pts, 3), however got {}".format(cube_xyz_min.shape))

        N_pts = cube_xyz_min.shape[0]
        cubeCorner_shift = np.indices((2, 2, 2)).reshape((3, -1)).T[None,:,:] * cube_D_mm    # (3,2,2,2) --> (1,8,3)
        cubeCorner = cube_xyz_min[:,None,:] + cubeCorner_shift      # (N_pts, 1, 3) + (1,8,3) --> (N_pts, 8, 3)
        img_h, img_w = self.perspectiveProj(projection_M = projection_M, xyz_3D = cubeCorner.reshape((N_pts*8, 3)), return_int_hw = return_int_hw, return_depth = return_depth)    # img_w/h: (N_Ms, N_pts*8) 
        img_w = img_w.reshape((-1, N_pts, 8))
        img_h = img_h.reshape((-1, N_pts, 8))
        return img_h, img_w

    
    def rotateImageVector(self, pts_3D, image_vector):
        '''
        pts_3D:
            shape:(M_view * N_points * 3)/(N_points * 3)
        image_vector:
            shape:(M_view * N_points * 3 * img_h * img_w)/(N_points * 3 * img_h * img_w)
        ----------------------------------------------------
        pts_3D = np.random.rand(1,2,3)
        image_vector = np.zeros((1,2,3,4,5))
        image_vector[...,:,:] = pts_3D[...,None,None]
        camera_test = Camera()
        q = camera_test.rotateImageVector(pts_3D, image_vector)
        print(q.shape)
        print(q)
        '''
        if(len(image_vector.shape) == 4):
            image_vector = np.moveaxis(image_vector,1,-1)
            N,img_h,img_w,_ = image_vector.shape
            matrix_x = np.zeros((N,img_h,img_w,3,3))
            matrix_yz = np.zeros((N,img_h,img_w,3,3))
        elif(len(image_vector.shape) == 5):
            image_vector = np.moveaxis(image_vector,2,-1)
            M,N,img_h,img_w,_ = image_vector.shape
            matrix_x = np.zeros((M,N,img_h,img_w,3,3))
            matrix_yz = np.zeros((M,N,img_h,img_w,3,3))
        else:
            raise ValueError('inputs shape is wrong')
        a = pts_3D[...,0]
        b = pts_3D[...,1]
        c = pts_3D[...,2]
        #print(pts_3D[...,1:])
        norm_bc = np.linalg.norm(pts_3D[...,1:], axis = -1)
        norm_abc = np.linalg.norm(pts_3D[...,:], axis = -1)



        matrix_x[...,0,0] = 1.0
        matrix_x[...,1,1] = (c/norm_bc)[...,None,None]
        matrix_x[...,1,2] = (b/norm_bc)[...,None,None]
        matrix_x[...,2,1] = (-b/norm_bc)[...,None,None]
        matrix_x[...,2,2] = (c/norm_bc)[...,None,None]

        matrix_yz[...,1,1] = 1.0
        matrix_yz[...,0,0] = (norm_bc/norm_abc)[...,None,None]
        matrix_yz[...,0,2] = (a/norm_abc)[...,None,None]
        matrix_yz[...,2,0] = (-a/norm_abc)[...,None,None]
        matrix_yz[...,2,2] = (norm_bc/norm_abc)[...,None,None]

        self.matrix_R = np.matmul(matrix_x, matrix_yz)
        image_vector = np.matmul(image_vector[...,None,:],self.matrix_R)
        image_vector = image_vector[...,0,:]
        image_vector = np.moveaxis(image_vector,-1,-3)
        return(image_vector)
    
    
    def inverseImageVector(self,
                           projection_M,
                           image_vector):
        '''
         pts_3D:
            shape:(M_view  * 3)
        image_vector:
            shape:(M_view * N_points * 3 * img_h * img_w)
        ----------------------------------------------------
        '''
        image_vector = np.moveaxis(image_vector,-3,-1)
        #N_Ms = projection_M.shape[0]
        N_Ms, N_pts, img_h, img_w,_ = image_vector.shape
        image_vector_new = np.ones((N_Ms, N_pts, img_h, img_w,4))
        image_vector_new[...,0:3] = image_vector
        
        projection_new = np.zeros((N_Ms, 4,4))
        projection_new[:,0:3,:] = projection_M
        projection_new[:,3,:] = np.array(([[0,0,0,1]]))
        projection_new = np.linalg.inv(projection_new)
        
        image_vector_inverse = np.matmul(projection_new, image_vector_new[...,None])
        image_vector_inverse = image_vector_inverse[...,0:3,0]
        image_vector_inverse = np.moveaxis(image_vector_inverse,-1,-3)
        return(image_vector_inverse)
    
    def generateVectorImage(self, 
                            projection_M, 
                            xyz_3D, 
                            img_shape = (50,50), 
                            vector_type = 'world',# world/camera/camera_rotate
                            ):
        '''
        np.random.seed(201611)
        #Ms = (np.random.rand(1,3,4) + ) / 10
        Ms = np.array([[[2607.429996,-3.844898,1498.178098,-533936.661373],
                        [-192.076910,2862.552532,681.798177,23434.686572],
                        [-0.241605,-0.030951,0.969881,22.540121]]])
        Ms = np.array([[[1,0,0,0],[0,1,0,0],[0,0,0.2,0]]])
        pts_3D = np.random.rand(2,3)
        pts_3D = np.array([[10.0,10.0,10.0]])

        camera_test = Camera()
        image_vector = camera_test.generateVectorImage(Ms, pts_3D)
        image_vector_show = image_vector/2 + 0.5
        np.moveaxis(image_vector_show[0,0],0,-1)
        import matplotlib.pyplot as plt
        plt.imshow(np.transpose(image_vector_show[0,0], (1, 2, 0)))
        plt.show()
        '''
        if projection_M.shape[-2:] != (3,4):
            raise ValueError("perspectiveProj needs projection_M with shape (3,4), however got {}".format(projection_M.shape))

        if xyz_3D.ndim == 1:
            xyz_3D = xyz_3D[None,:]

        if xyz_3D.shape[1] != 3 or xyz_3D.ndim != 2:
            raise ValueError("perspectiveProj needs xyz_3D with shape (3,) or (N_pts, 3), however got {}".format(xyz_3D.shape))
        # perspective projection
        N_pts = xyz_3D.shape[0]
        self.N_pts = N_pts
        N_Ms = projection_M.shape[0]
        self.N_Ms = N_Ms
        xyz1 = np.c_[xyz_3D, np.ones((N_pts,1))].astype(np.float64) # (N_pts, 3) ==> (N_pts, 4)
        pts_3D = np.matmul(projection_M, xyz1.T) # (3, 4)/(N_Ms, 3, 4) * (4, N_pts) ==> (3, N_pts)/(N_Ms,3,N_pts)
        # the result is vector: [w,h,1], w is the first dim!!! (matlab's x/y/1')
        pts_3D = pts_3D.swapaxes(1,2)
        pts_3D_new = pts_3D[:, :, :, np.newaxis, np.newaxis]
        
        img_h,img_w = img_shape
        x = np.arange(0, img_w, 1.0)
        y = np.arange(0, img_h, 1.0)
        xx, yy = np.meshgrid(x, y)
        XY=np.array([yy.flatten(),xx.flatten()]).T.reshape(img_h, img_w, 2)
        XY = np.moveaxis(XY,2,0)
        Z = np.ones((1, img_h, img_w))
        XYZ = np.concatenate((XY, Z), axis = 0)
        XYZ = XYZ[np.newaxis, np.newaxis, :, :,:]
        
        if(vector_type == 'camera_rotate'):
            image_vector = self.rotateImageVector(pts_3D, pts_3D_new - XYZ)
            image_vector_norm = np.linalg.norm(image_vector, axis = 2)[:,:,np.newaxis,:,:]
            image_vector = image_vector/image_vector_norm
            #print(image_vector.shape)
            #print(image_vector[1,1,:,2,3])
            return image_vector
        elif(vector_type == 'world'):
            image_vector = self.inverseImageVector(projection_M, -pts_3D_new + XYZ)
            image_vector_norm = np.linalg.norm(image_vector, axis = 2)[:,:,np.newaxis,:,:]
            image_vector = image_vector/image_vector_norm
            #print(image_vector.shape)
            #print(image_vector[1,1,:,2,3])
            return image_vector


class SparseCubes():
    def __init__(self):
        pass
    
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
            saved_pts['red'], saved_pts['green'], saved_pts['blue'] = rgb_np[:,0], rgb_np[:,1], rgb_np[:,2]
        if normal_np is not None:
            saved_pts['nx'], saved_pts['ny'], saved_pts['nz'] = normal_np[:,0], normal_np[:,1], normal_np[:,2] 

        el_vertex = PlyElement.describe(saved_pts, 'vertex')
        outputFolder = os.path.dirname(ply_filePath)
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        PlyData([el_vertex]).write(ply_filePath)
        print('saved ply file: {}'.format(ply_filePath))
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
        save2ply(ply_filePath, xyz_np, rgb_np, normal_np)
        return 1



from plyfile import PlyData, PlyElement

class PLY2volumn(Params):
    def __init__(self, model_num):
        super(PLY2volumn, self).__init__()
        self.model_num = model_num
        self.load_modelSpecific_params(self._datasetName, model_num)
    
    def ply2array(self):
        
        
        if(self._datasetName == 'DTU'):
            pcd_name = os.path.join(self.datasetFolder, self.stl_name)
            pcd = PlyData.read(pcd_name)
            count = pcd.elements[0].count

            self.pcd_xyz = np.c_[pcd['vertex']['x'], pcd['vertex']['y'], pcd['vertex']['z']]
    
        elif(self._datasetName == 'MVS2d'):
            #pcd_name = os.path.join(self.datasetFolder, self.stl_name)
            pcd = np.load(self.stl_name)
            
            count,_,_ = pcd[0].shape
            self.pcd_xyz = np.zeros((count, 3))
            self.pcd_xyz[:,0] = pcd[0,:,0,1]
            self.pcd_xyz[:,2] = pcd[0,:,0,0]
            
    def points2voxel(self, min_point, max_point, resol):
        
        if(self._datasetName == 'DTU'):
            self.min_point = min_point
            self.max_point = max_point
            #shift = np.array([[resol/2, resol/2, resol/2]])
            quanti_points = ((self.pcd_xyz - min_point)//resol).astype(int)
            max_check_points = (max_point + resol * self._cube_D  - self.pcd_xyz)>=0
            a,b,c = tuple(((max_point + resol * self._cube_D + 3 - min_point)//resol + self._cube_D).astype(int))
            index_list = (quanti_points>=0)
            index = index_list[:,0] * index_list[:,1] * index_list[:,2] * max_check_points[:,0] * max_check_points[:,1] * max_check_points[:,2]
            quanti_points = quanti_points[index]
            voxel = np.zeros((a,b,c)).astype(int)
            counts,_ = quanti_points.shape
            for i in range(counts):
                px,py,pz = tuple(quanti_points[i])
                voxel[px,py,pz] += 1
            #print(counts)
            return(voxel)
        elif(self._datasetName == 'MVS2d'):
            self.min_point = min_point
            self.max_point = max_point
            #shift = np.array([[resol/2, resol/2, resol/2]])
            quanti_points = ((self.pcd_xyz - min_point)//resol).astype(int)
            max_check_points = (max_point + resol * self._cube_D  - self.pcd_xyz)>=0
            a,b,c = tuple(((max_point + resol * self._cube_D + 3 - min_point)//resol + self._cube_D).astype(int))
            index_list = (quanti_points>=0)
            index = index_list[:,0] * index_list[:,1] * index_list[:,2] * max_check_points[:,0] * max_check_points[:,1] * max_check_points[:,2]
            quanti_points = quanti_points[index]
            voxel = np.zeros((a,1,c)).astype(int)
            counts,_ = quanti_points.shape
            for i in range(counts):
                px,_,pz = tuple(quanti_points[i])
                voxel[px,0,pz] += 1
            #print(counts)
            return(voxel)
        
    def init(self, min_point, max_point, resol):
        self.ply2array()
        self.voxel = self.points2voxel(min_point,max_point,resol)
        return (self.voxel)
    
    def gen_idx(self, voxel_point, resol, threhold = 2):
        a,b,c = tuple(((voxel_point - self.min_point)//resol).astype(int))
        self.box = self.voxel[a:a + self._cube_D, b:b + self._cube_D, c:c + self._cube_D]
        self.surface = np.zeros(self.box.shape)
        self.surface[self.box >= threhold] = 1.0
        
        if(self._datasetName == 'DTU'):
            return(self.surface[None,:,:,:])
        elif(self._datasetName == 'MVS2d'):
            return(np.moveaxis(self.surface,0,1))



class CVC(Params):
    def __init__(self):
        super(CVC, self).__init__()
        pass
    
    def _colorize_cube_(self, view_set, cameraPOs_np, model_imgs_np, xyz, resol, densityCube, colorize_cube_D, visualization_ON=False):
        """ 
        generate colored cubes of a perticular densityCube  
        inputs: 
        output: [views_N, 3, colorize_cube_D, colorize_cube_D, colorize_cube_D]. 3 is for RGB
        """
        
        if(self._datasetName == 'DTU'):
            min_x,min_y,min_z = xyz
            indx_xyz = range(0,colorize_cube_D)
            ##meshgrid: indexing : {'xy', 'ij'}, optional     ##Cartesian ('xy', default) or matrix ('ij') indexing of output.    
            indx_x,indx_y,indx_z = np.meshgrid(indx_xyz,indx_xyz,indx_xyz,indexing='ij')  
            indx_x = indx_x * resol + min_x
            indx_y = indx_y * resol + min_y
            indx_z = indx_z * resol + min_z
            homogen_1s = np.ones(colorize_cube_D**3, dtype=np.float64)
            pts_4D = np.vstack([indx_x.flatten(),indx_y.flatten(),indx_z.flatten(),homogen_1s])

            N_views = len(view_set)
            colored_cubes = np.zeros((N_views,3,colorize_cube_D,colorize_cube_D,colorize_cube_D))
            # only chooce from inScope views
            # center_pt_xyz1 = np.asarray([grid_D*resol/2 + min_x, grid_D*resol/2 + min_y, grid_D*resol/2 + min_z, 1])
            # center_pt_3D = np.dot(cameraPOs_np,center_pt_xyz1)
            # center_pt_wh = center_pt_3D[:,:-1] / center_pt_3D[:,-1:]# the result is vector: [w,h,1], w is the first dim!!!
            # valid_views = (center_pt_wh[:,0]<max_w) & (center_pt_wh[:,1]<max_h) & (center_pt_wh[:,0]>0) & (center_pt_wh[:,1]>0)      
            # while valid_views.sum() < N_randViews: ## if only n views can see this pt, where n is smaller than N_randViews, randomly choose some more
            #     valid_views[random.randint(1,cameraPOs_np.shape[0]-1)] = True
            # valid_view_list = list(valid_views.nonzero()[0]) ## because the cameraPOs_np[0] is zero, don't need +1 here
            # view_list = random.sample(valid_view_list,N_randViews)    

            for _n, _view in enumerate(view_set):
                # perspective projection
                projection_M = cameraPOs_np[_view]  ## use viewIndx
                pts_3D = np.dot(projection_M, pts_4D)
                pts_3D[:-1] /= pts_3D[-1] # the result is vector: [w,h,1], w is the first dim!!!
                pts_2D = pts_3D[:-1].round().astype(np.int32)
                pts_w, pts_h = pts_2D[0], pts_2D[1]
                # access rgb of corresponding model_img using pts_2D coordinates
                pts_RGB = np.zeros((colorize_cube_D**3, 3))
                img = model_imgs_np[_view]  ## use viewIndx
                max_h, max_w, _ = img.shape
                inScope_pts_indx = (pts_w<max_w) & (pts_h<max_h) & (pts_w>1) & (pts_h>1)
                
                pts_h_float, pts_w_float = pts_3D[:-1][0], pts_3D[:-1][1]
                pts_h_down, pts_w_down = np.floor(pts_h_float).astype(np.int32), np.floor(pts_w_float).astype(np.int32)
                pts_h_up, pts_w_up = np.ceil(pts_h_float).astype(np.int32), np.ceil(pts_w_float).astype(np.int32)
                
                
                #pts_RGB[inScope_pts_indx] = img[pts_h_up[inScope_pts_indx],pts_w_up[inScope_pts_indx]] * (1 - (pts_h_up - pts_h_float)) * (1 - (pts_w_up - pts_w_float)) +\
                #                            img[pts_h_up[inScope_pts_indx],pts_w_down[inScope_pts_indx]] * (1 - (pts_h_up - pts_h_float)) * (1 - (-pts_w_down + pts_w_float)) +\
                #                            img[pts_h_down[inScope_pts_indx],pts_w_up[inScope_pts_indx]] * (1 - (-pts_h_down + pts_h_float)) * (1 - (pts_w_up - pts_w_float)) +\
                #                            img[pts_h_down[inScope_pts_indx],pts_w_down[inScope_pts_indx]] * (1 - (-pts_h_down + pts_h_float)) * (1 - (-pts_w_down + pts_w_float))
                
                pts_RGB[inScope_pts_indx] = img[pts_h[inScope_pts_indx],pts_w[inScope_pts_indx]]
                colored_cubes[_n] = pts_RGB.T.reshape((3,colorize_cube_D,colorize_cube_D,colorize_cube_D))
            if visualization_ON:    
                visualize_N_densities_pcl([densityCube]+[colored_cubes[n] for n in range(0,len(5))])

            return colored_cubes
        
        elif(self._datasetName == 'MVS2d'):
            min_x,_,min_z = xyz
            indx_xyz = range(0,colorize_cube_D)
            ##meshgrid: indexing : {'xy', 'ij'}, optional     ##Cartesian ('xy', default) or matrix ('ij') indexing of output.    
            indx_x,indx_y,indx_z = np.meshgrid(indx_xyz,[0],indx_xyz,indexing='ij')  
            indx_x = indx_x * resol + min_x
            indx_y = indx_y * resol 
            indx_z = indx_z * resol + min_z
            homogen_1s = np.ones(colorize_cube_D**2, dtype=np.float64)
            pts_4D = np.vstack([indx_x.flatten(),indx_y.flatten(),indx_z.flatten(),homogen_1s])

            N_views = len(view_set)
            colored_cubes = np.zeros((N_views,3,colorize_cube_D,1,colorize_cube_D))
            # only chooce from inScope views
            # center_pt_xyz1 = np.asarray([grid_D*resol/2 + min_x, grid_D*resol/2 + min_y, grid_D*resol/2 + min_z, 1])
            # center_pt_3D = np.dot(cameraPOs_np,center_pt_xyz1)
            # center_pt_wh = center_pt_3D[:,:-1] / center_pt_3D[:,-1:]# the result is vector: [w,h,1], w is the first dim!!!
            # valid_views = (center_pt_wh[:,0]<max_w) & (center_pt_wh[:,1]<max_h) & (center_pt_wh[:,0]>0) & (center_pt_wh[:,1]>0)      
            # while valid_views.sum() < N_randViews: ## if only n views can see this pt, where n is smaller than N_randViews, randomly choose some more
            #     valid_views[random.randint(1,cameraPOs_np.shape[0]-1)] = True
            # valid_view_list = list(valid_views.nonzero()[0]) ## because the cameraPOs_np[0] is zero, don't need +1 here
            # view_list = random.sample(valid_view_list,N_randViews)    

            for _n, _view in enumerate(view_set):
                # perspective projection
                projection_M = cameraPOs_np[_view]  ## use viewIndx
                pts_3D = np.dot(projection_M, pts_4D)
                pts_3D[:-1] /= pts_3D[-1] # the result is vector: [w,h,1], w is the first dim!!!
                pts_2D = pts_3D[:-1].round().astype(np.int32)
    
                
                pts_h, pts_w = pts_2D[0], pts_2D[1]
                # access rgb of corresponding model_img using pts_2D coordinates
                pts_RGB = np.zeros((colorize_cube_D**2, 3))
                img = model_imgs_np[_view]  ## use viewIndx
                max_h, max_w, _ = img.shape
                inScope_pts_indx = (pts_w<max_w) & (pts_h<max_h) & (pts_w>=0) & (pts_h>=0)
                
                pts_h_float, pts_w_float = pts_3D[:-1][0], pts_3D[:-1][1]
                pts_h_down, pts_w_down = np.floor(pts_h_float).astype(np.int32), np.floor(pts_w_float).astype(np.int32)
                pts_h_up, pts_w_up = np.ceil(pts_h_float).astype(np.int32), np.ceil(pts_w_float).astype(np.int32)
                
                #pts_RGB[inScope_pts_indx] = img[pts_h_up[inScope_pts_indx],pts_w[inScope_pts_indx]].astype(np.float) * (1 - (pts_h_up[inScope_pts_indx][None,:,None] - pts_h_float[inScope_pts_indx][None,:,None]))  +\
                #                            img[pts_h_down[inScope_pts_indx],pts_w[inScope_pts_indx]].astype(np.float) * (1 - (-pts_h_down[inScope_pts_indx][None,:,None] + pts_h_float[inScope_pts_indx][None,:,None])) 
                
                #pts_RGB[inScope_pts_indx] = pts_RGB[inScope_pts_indx].round()
                #a = pts_RGB[inScope_pts_indx]
                #pts_RGB[inScope_pts_indx] = img[pts_h[inScope_pts_indx],pts_w_up[inScope_pts_indx]] * (1 - (pts_w_up[inScope_pts_indx][None,:,None] - pts_w_float[inScope_pts_indx][None,:,None]))  +\
                #                            img[pts_h[inScope_pts_indx],pts_w_down[inScope_pts_indx]] * (1 - (-pts_w_down[inScope_pts_indx][None,:,None] + pts_w_float[inScope_pts_indx][None,:,None])) 
                #pts_RGB[inScope_pts_indx] = pts_RGB[inScope_pts_indx].round().astype(np.uint8)
                
                pts_RGB[inScope_pts_indx] = img[pts_h[inScope_pts_indx],pts_w[inScope_pts_indx]]
                #print(a - pts_RGB[inScope_pts_indx])
                
                colored_cubes[_n] = pts_RGB.T.reshape((3,colorize_cube_D,1,colorize_cube_D))
                
            if visualization_ON:    
                visualize_N_densities_pcl([densityCube]+[colored_cubes[n] for n in range(0,len(5))])


            return colored_cubes

    def gen_coloredCubes(self, 
                         selected_viewPairs, 
                         xyz, 
                         resol, 
                         cameraPOs, 
                         models_img, 
                         colorize_cube_D, 
                         visualization_ON = False, 
                         occupiedCubes_01=None):     
        """
        inputs: 
        selected_viewPairs: (N_select_viewPairs, 2)
        xyz, resol: parameters for each occupiedCubes (N,params)
        occupiedCubes_01: multiple occupiedCubes (N,)+(colorize_cube_D,)*3
        return:
        coloredCubes = (N*N_select_viewPairs,3*2)+(colorize_cube_D,)*3 
        """
        N_select_viewPairs = selected_viewPairs.shape[0]
        
        if(self._datasetName == 'DTU'):
            coloredCubes = np.zeros((N_select_viewPairs*2,3)+(colorize_cube_D,)*3, dtype=np.float32) # reshape at the end
        elif(self._datasetName == 'MVS2d'):
            coloredCubes = np.zeros((N_select_viewPairs*2,3)+(colorize_cube_D,colorize_cube_D), dtype=np.float32) # reshape at the end

        if visualization_ON:
            if occupiedCubes_01 is None:
                print 'error: [func]gen_coloredCubes, occupiedCubes_01 should not be None when visualization_ON==True'
            occupiedCube_01 = occupiedCubes_01
        else:
            occupiedCube_01 = None
        ##randViewIndx = random.sample(range(1,cameraPOs.shape[0]),N_randViews)


        #(N_select_viewPairs, 2) ==> (N_select_viewPairs*2,). 
        selected_views = selected_viewPairs.flatten() 
        # because selected_views could include duplicated views, this case is not the best way. But if the N_select_viewPairs is small, it doesn't matter too much
        coloredCubes = self._colorize_cube_(view_set = selected_views, \
                cameraPOs_np = cameraPOs, model_imgs_np = models_img, xyz = xyz, resol = resol, \
                visualization_ON=visualization_ON, colorize_cube_D=colorize_cube_D, densityCube = occupiedCube_01)


        
        if(self._datasetName == 'DTU'):
            return coloredCubes.reshape((N_select_viewPairs,3*2)+(colorize_cube_D,colorize_cube_D,colorize_cube_D))
        elif(self._datasetName == 'MVS2d'):
            return coloredCubes.reshape((N_select_viewPairs,3*2)+(colorize_cube_D,colorize_cube_D))


    def preprocess_augmentation(self, gt_sub, X_sub, mean_rgb, augment_ON = True, crop_ON = True):
        # X_sub /= 255.
        X_sub = X_sub.astype(np.float32)
        X_sub -= mean_rgb  ##.5

        if augment_ON:
            X_sub += np.random.randint(-30,30,1) # illumination argmentation
            X_sub += np.random.randint(-5,5,mean_rgb.shape) # color channel argmentation
            gt_sub, X_sub = data_augment_rand_rotate(gt_sub, X_sub) # randly rotate multiple times
            gt_sub, X_sub = data_augment_rand_rotate(gt_sub, X_sub)
            gt_sub, X_sub = data_augment_rand_rotate(gt_sub, X_sub)
            ##gt_sub, X_sub = data_augment_scipy_rand_rotate(gt_sub, X_sub) ## take a lot of time
        if crop_ON:
            gt_sub, X_sub = data_augment_crop([gt_sub, X_sub], random_crop=augment_ON) # smaller size cube       
        return gt_sub, X_sub


class ViewPairSelection(Params):
    def __init__(self):
        super(ViewPairSelection, self).__init__()
        self.camera = Camera()
        self.min_angle = 90
        self.view_pair_num = 5
    
    def __argmaxN_viewPairs__(self, viewPairs, w_viewPairs, N_argmax):
        """
        inputs:
            viewPairs:  (N_viewPairs, 2) np.int 
            w_viewPairs:  (N_validCubes, N_viewPairs)
            N_argmax: argmax_N
        outputs:
            argmaxN_viewPairs: (N_validCubes, N_argmax, 2)
            argmaxN_w: (N_validCubes, N_argmax)   

        -----------
        usages:
        >>> w_viewPairs = np.array([[3,1,2], [0,-1,70]])
        >>> viewPairs = utils.k_combination_np(range(3), k = 2) # the 3 positions corresponding to viewPairs [[0,1], [0,2], [1,2]]
        >>> __argmaxN_viewPairs__(viewPairs, w_viewPairs, 1)   # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        (array([[[0, 1]],
        <BLANKLINE>
               [[1, 2]]]), array([[ 3],
                                  [70]]))
        >>> __argmaxN_viewPairs__(viewPairs, w_viewPairs, 2)   # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        (array([[[1, 2],
                 [0, 1]],
        <BLANKLINE>
                [[0, 1],
                 [1, 2]]]), array([[ 2,  3],
                                   [ 0, 70]]))
        """

        N_validCubes, N_viewPairs = w_viewPairs.shape
        indice_cube, _ = np.indices((N_validCubes, N_argmax))   # (2, N_validCubes, N_argmax)
        indice_N_max = w_viewPairs.argsort(axis=1)[:, -1*N_argmax:]    # (N_validCubes, N_viewPairs) np.float32 --> (N_validCubes, N_argmax) np.int
        argmaxN_viewPairs = np.repeat(viewPairs[None,...], N_validCubes, axis=0)[indice_cube, indice_N_max] # (N_viewPairs, 2) --> (N_validCubes, N_viewPairs, 2) --> (N_validCubes, N_argmax, 2)
        argmaxN_w = w_viewPairs[indice_cube, indice_N_max]  # (N_validCubes, N_argmax)
        return argmaxN_viewPairs, argmaxN_w

    
    def viewPairSelection_easy(self, 
                         projection_M, 
                         xyz_3D,
                         min_angle = 90,
                         view_pair_num = 5
                         ):
        '''
        inputs:
            projection_M
            xyz_3D
        outputs:
            
        '''
        self.viewRejection(projection_M, xyz_3D)
        pts_middle_3D = np.mean(xyz_3D, axis = 0)[None,...]
        self.projection_M_r = projection_M
        cameraTs_np = self.camera.cameraPs2Ts(
            cameraPOs = self.projection_M_r)
        #theta_viewPairs = self.camera.viewPairAngles_wrt_pts(
        #    cameraTs = cameraTs_np, 
        #    pts_xyz = xyz_3D)
        viewPairQualified = list(itertools.combinations(self.quaified_view,2))
        random.shuffle(viewPairQualified)
        #pdb.set_trace()
        
        projection_M_pair = np.zeros((2,3,4))
        qualified_projection_M_pair = []
        pairIndex = []
        
        for ele in viewPairQualified:
            projection_M_pair[0] = self.projection_M_r[ele[0]]
            projection_M_pair[1] = self.projection_M_r[ele[1]]
            cameraTs_np = self.camera.cameraPs2Ts(
                cameraPOs = projection_M_pair)
            theta_viewPairs = self.camera.viewPairAngles_wrt_pts(
                cameraTs = cameraTs_np, 
                pts_xyz = pts_middle_3D)
            if(theta_viewPairs > np.pi * min_angle / 180.0):
                continue
            else:
                qualified_projection_M_pair.append(projection_M_pair)
                pairIndex.append(ele)
            if(len(pairIndex)>=view_pair_num):
                break
        return (pairIndex, qualified_projection_M_pair)
   
    def viewRejection(self,
                     projection_M,
                     xyz_3D,
                     image_size = (1200,1600),
                     patch_size = (56, 56),
                     ):
        img_h, img_w = self.camera.perspectiveProj(projection_M,xyz_3D,return_int_hw = False)
        N_Ms, N_pts = img_h.shape
        img_h_min = patch_size[0]/2
        img_w_min = patch_size[1]/2
        img_h_max = image_size[0] - patch_size[0]/2
        img_w_max = image_size[1] - patch_size[1]/2
        #img_h = np.array([[255,55,64],[222,1200,2]])
        #img_w = np.array([[255,55,64],[222,1200,2]])
        self.quaified_view = []
        self.deleted_view = []
        for i in range(N_Ms):
            bool_h = all(((j > img_h_min)&(j < img_h_max)) for j in img_h[i,:])
            bool_w = all(((j > img_w_min)&(j < img_w_max)) for j in img_w[i,:])
            if(bool_h&bool_w):
                self.quaified_view.append(i)
            else:
                self.deleted_view.append(i)
                self.quaified_view.append(i)#TODO :delete it

        self.projection_M_r = np.delete(projection_M, self.deleted_view, 0)
        if(len(self.quaified_view)<3):
            raise ValueError('Do not have enough view pairs.Need to select points properly')
        return(self.projection_M_r, self.quaified_view)

    def cal_w_viewPairs(self,
                        cameraTs_np,
                        validCubes, 
                        cubeCenters_xyz, 
                        middle_point = np.array([400,0,400]),
                        ):
        
        '''
        calculate the viewPair score in order to select the viewPair in front of the mode
        
        '''
        t_test = cameraTs_np
        t_v_test = self.camera.k_combination_np(t_test)
        t_c_test = cubeCenters_xyz[validCubes]
        t_m_test = t_v_test.mean(axis = 1)

        num_viewPair,_ = t_m_test.shape
        num_validCubes,_ = t_c_test.shape

        vect_p2p1 = t_m_test - middle_point
        vect_p2p3 = t_c_test - middle_point
        unit_vector = lambda v: v / np.linalg.norm(v, axis = 1)[...,None]

        angle_viewPairs = np.zeros((num_validCubes, num_viewPair))
        for i in range(num_validCubes):
            dot = unit_vector(vect_p2p3)[i][None,...] * unit_vector(vect_p2p1)
            cos_angle = (dot.sum(axis = 1))

            angle_viewPairs[i] = cos_angle

        theta_viewPairs = self.camera.viewPairAngles_wrt_pts(cameraTs = t_test, pts_xyz = t_c_test)[..., None]  # (N_validCubes, N_viewPairs, 1)
        thre_angle = self.thre_angle
        valid_viewPair = theta_viewPairs < (thre_angle * np.pi /180.)
        score_viewPair = angle_viewPairs * valid_viewPair[:,:,0]
        
        return score_viewPair
    
    def viewPairSelection(self, 
                          cameraTs_np, 
                          e_viewPairs, 
                          d_viewPairs, 
                          validCubes, 
                          cubeCenters_xyz, 
                          viewPair_relativeImpt_fn, 
                          batchSize, 
                          N_viewPairs4inference, 
                          viewPairs):
        """

        ------------
        inputs:
        e_viewPairs: patches' embedding  (N_cubes, N_views, D_embedding)
        d_viewPairs
        validCubes
        viewPair_relativeImpt_fn
        batchSize
        viewPairs: (N_viewPairs, 2) np.int
        ------------
        outputs:
            w_viewPairs: 
        """

        N_cubes, N_viewPairs = d_viewPairs.shape[:2]
        N_validCubes = validCubes.sum()
        D_embedding = e_viewPairs.shape[-1]
        theta_viewPairs = camera.viewPairAngles_wrt_pts(cameraTs = cameraTs_np, pts_xyz = cubeCenters_xyz[validCubes])[..., None]  # (N_validCubes, N_viewPairs, 1)
        d_viewPairs = d_viewPairs[validCubes][..., None]  # (N_validCubes, N_viewPairs, 1)
        w_viewPairs = np.empty((N_validCubes, N_viewPairs), dtype = np.float32)

        # TODO: change fn to accept pair-by-pair inputs rather than the cube-by-cube inputs
        for _batch in utils.yield_batch_npBool(N_all = N_validCubes, batch_size = int(math.floor(float(batchSize) / N_viewPairs))):
            N_batch = _batch.sum()
            # (N_cubes, N_views, D_embedding) --> (N_validCubes, N_views, D_embedding) --> (N_batch, N_viewPairs * 2, D_embedding) --> (N_batch, N_viewPairs, 2 * D_embedding)
            _e_viewPairs = e_viewPairs[validCubes][_batch][:,viewPairs.flatten()].reshape((N_batch, N_viewPairs, 2 * D_embedding))
            N_features = 2 * D_embedding + 2
            # (N_batch, N_viewPairs, N_features),  
            features_viewPairs = np.concatenate([_e_viewPairs, d_viewPairs[_batch], theta_viewPairs[_batch]], axis=-1).astype(np.float32).reshape((N_batch*N_viewPairs, N_features))

            # TODO: check        
            w_viewPairs[_batch] = viewPair_relativeImpt_fn(features_viewPairs, n_samples_perGroup = N_viewPairs)    # (N_batch*N_viewPairs, N_features) --> (N_batch, N_viewPairs) 

        # select N_argmax viewPairs
        selected_viewPairs, selected_similNet_weight = __argmaxN_viewPairs__(viewPairs = viewPairs, w_viewPairs = w_viewPairs, N_argmax = N_viewPairs4inference)

        return selected_viewPairs, selected_similNet_weight
        

    def viewPairSelection_random(self, 
                         cameraTs_np, 
                         e_viewPairs, 
                         d_viewPairs, 
                         validCubes, 
                         cubeCenters_xyz, 
                         inScope_cubes_vs_views,
                         viewPairs, 
                         N_validCubes, 
                         N_random):
        """
        inputs:
            viewPairs:  (N_viewPairs, 2) np.int 
            N_validCubes: int
            N_argmax: N_random pairs
        outputs:
            randomN_viewPairs: (N_validCubes, N_random, 2)

        -----------
        select = ViewPairSelection()
        viewPair = np.array([[2,3],[1,4],[3,6],[9,9]])
        select.viewPairSelection_random(viewPair,9,3)
        """

        N_cubes, N_viewPairs = d_viewPairs.shape[:2]
        N_validCubes = validCubes.sum()
        D_embedding = e_viewPairs.shape[-1]
        theta_viewPairs = self.camera.viewPairAngles_wrt_pts(cameraTs = cameraTs_np, pts_xyz = cubeCenters_xyz[validCubes])[..., None]  # (N_validCubes, N_viewPairs, 1)
        d_viewPairs = d_viewPairs[validCubes][..., None]  # (N_validCubes, N_viewPairs, 1)
        
        #e_viewPair_valid = np.zeros((N_validCubes, N_viewPairs, D_embedding * 2))
        e_viewPair_valid = e_viewPairs[validCubes][:,viewPairs.flatten()].reshape((N_validCubes, N_viewPairs, 2 * D_embedding))
        
        inScope_cubes_viewPairs = inScope_cubes_vs_views[validCubes][:,viewPairs.flatten()].reshape((N_validCubes, N_viewPairs, 2))
        inScope_cubes_viewPairs = inScope_cubes_viewPairs[...,0] * inScope_cubes_viewPairs[...,1]
        score_viewPairs =  self.cal_w_viewPairs( middle_point = self.BB.mean(axis = 1),
                                            cameraTs_np = cameraTs_np,
                                            validCubes = validCubes, 
                                            cubeCenters_xyz = cubeCenters_xyz, 
                                            )
        w_viewPairs = score_viewPairs * inScope_cubes_viewPairs
        #w_viewPairs = np.random.rand(N_validCubes, N_viewPairs) * inScope_cubes_viewPairs
        
        #pdb.set_trace()
        
        indice_cube, _ = np.indices((N_validCubes, N_random))   # (2, N_validCubes, N_argmax)
        indice_N_max = w_viewPairs.argsort(axis=1)[:, -1*N_random:]    # (N_validCubes, N_viewPairs) np.float32 --> (N_validCubes, N_argmax) np.int
        
        randomN_viewPairs = np.repeat(viewPairs[None,...], N_validCubes, axis=0)[indice_cube, indice_N_max] # (N_viewPairs, 2) --> (N_validCubes, N_viewPairs, 2) --> (N_validCubes, N_argmax, 2)
        random_theta_viewPairs = theta_viewPairs[indice_cube, indice_N_max] # (N_viewPairs, 2) --> (N_validCubes, N_viewPairs, 2) --> (N_validCubes, N_argmax, 2)
        random_d_viewPairs = d_viewPairs[indice_cube, indice_N_max] # (N_viewPairs, 2) --> (N_validCubes, N_viewPairs, 2) --> (N_validCubes, N_argmax, 2)
        random_e_viewPairs = e_viewPair_valid[indice_cube, indice_N_max] # (N_viewPairs, 2) --> (N_validCubes, N_viewPairs, 2) --> (N_validCubes, N_argmax, 256)
        
        random_embeddings = np.concatenate((random_theta_viewPairs, random_d_viewPairs, random_e_viewPairs), axis = 2)

        
        return randomN_viewPairs, random_embeddings




class Volumn(Params):
    def __init__(self, resol = 1.0):
        super(Volumn, self).__init__()
        self.resol = resol
        self.init_tools()
        #self.init_data(images_list, cameraPOs_np, cameraTs_np)
        
    def init_tools(self):
        self.scene = Scene()
        self.image = Image()
        self.sparseCubes = SparseCubes()
        self.camera = Camera()
        self.similarityNet = SimilarityNet()
        self.viewPairSelection = ViewPairSelection()
        
    def init_data(self, images_list = None, cameraPOs_np = None, cameraTs_np = None):
        #self.images_list = images_list
        self.cameraPOs_np = cameraPOs_np
        self.cameraTs_np = cameraTs_np
        
    def init_volumn(self,images_list, cameraPOs_np, cameraTs_np):
        
        #self._cube_Dcenter = _cube_Dcenter
        self.init_data(images_list, cameraPOs_np, cameraTs_np)
        
        if self.initialPtsNamePattern is None:
            self.cubes_param_np, self.cube_D_mm = self.scene.initializeCubes(resol = self.resol, 
                                                              cube_D = self._cube_D, 
                                                              cube_Dcenter = self._cube_Dcenter, 
                                                              cube_overlapping_ratio = self._cube_overlapping_ratio, 
                                                              BB = self.BB)  # (N_cubes,N_params), scalar. the scene is divided into multiple overlapping cubes, each of which has several attributes, such as param_np["xyz"/"ijk"/"resol"]
        else:
            self.initial_pts_xyz = self.scene.readPointCloud_xyz(pointCloudFile = os.path.join(self.datasetFolder, self.initialPtsNamePattern))
            self.cubes_param_np, self.cube_D_mm = self.scene.quantizePts2Cubes(pts_xyz = self.initial_pts_xyz, 
                                                                resol = self.resol, 
                                                                cube_D = self._cube_D, \
                                                                cube_Dcenter = self._cube_Dcenter,
                                                                cube_overlapping_ratio = self._cube_overlapping_ratio, 
                                                                BB = self.BB)
        
        #self.sparseCubes.save2ply(os.path.join(self.outputFolder, 'initialCubes.ply'), xyz_np = cubes_param_np['xyz'] + cube_D_mm/2)  # save the cube positions to ply file
        self.img_h_cubesCorner, self.img_w_cubesCorner = self.camera.perspectiveProj_cubesCorner(
            projection_M = self.cameraPOs_np, 
            cube_xyz_min = self.cubes_param_np['xyz'], 
            cube_D_mm = self.cube_D_mm, 
            return_int_hw = False, 
            return_depth = False)       # img_w/h_cubesCorner (N_views, N_cubes, 8)
        self.img_h_cubesCenter, self.img_w_cubesCenter = self.camera.perspectiveProj(
            projection_M = self.cameraPOs_np, \
            xyz_3D = self.cubes_param_np['xyz'] + self.cube_D_mm/2, \
            return_int_hw = False, 
            return_depth = False)    # img_w/h: (N_Ms, N_pts) 
        self.N_views, self.N_cubes = self.img_h_cubesCorner.shape[:2]

        self.D_embedding = self._D_imgPatchEmbedding 
        
        self.viewPairs = utils.k_combination_np(range(self.N_views), k = 2)     # (N_viewPairs, 2)
        self.N_viewPairs = self.viewPairs.shape[0]
        
        self.patches_mean_bgr = self._MEAN_PATCHES_BGR

    
    def train_rejection(self, dataset):
        train_validCubics = []
        cubic_num,_ = self.cubes_param_np['xyz'].shape
        for i in range(cubic_num):
            #pdb.set_trace()
            loc_min = self.cubes_param_np['xyz'][i]
            loc_max = loc_min + self.resol * self._cube_D
            flag_min = ((dataset.ply.pcd_xyz[::20] - loc_min) >=0)
            flag_max = ((loc_max - dataset.ply.pcd_xyz[::20]) >=0)
            flag = flag_min * flag_max
            flag_in = flag.prod(axis = 1)
            flag_all_in = flag_in.sum()
            output = ((flag_all_in >= 10) and self.validCubes[i])
            train_validCubics.append(output)
            
        self.validCubes = np.array(train_validCubics)
        self.validCubesIndex = [i for i, x in enumerate(self.validCubes) if x]
        self.N_validCubes = self.validCubes.sum()
    
    
    def early_rejection(self, images_list = None, dataset = None):
        
        if(self._datasetName == 'DTU'):
            if(self._use_old_early_rejection):
                self.patch2embedding_fn, self.embeddingPair2simil_fn = self.similarityNet.similarityNet_inference(
                    model_file = self._pretrained_similNet_model_file, \
                    imgPatch_hw_size = (self._imgPatch_hw_size, )*2 )
                self.patches_embedding, self.inScope_cubes_vs_views = earlyRejection.patch2embedding( \
                        images_list, 
                        self.img_h_cubesCorner, 
                        self.img_w_cubesCorner, 
                        self.patch2embedding_fn, 
                        self.patches_mean_bgr, \
                        self.N_cubes, 
                        self.N_views, 
                        self.D_embedding, 
                        patchSize = self._imgPatch_hw_size, \
                        batchSize = self._batchSize_similNet_patch2embedding, \
                        cubeCenter_hw = np.stack([self.img_h_cubesCenter, self.img_w_cubesCenter], axis=0))    # (N_cubes, N_views, D_embedding), (N_cubes, N_views)

                self.dissimilarity = earlyRejection.embeddingPairs2simil(
                        embeddings = self.patches_embedding, 
                        embeddingPair2simil_fn = self.embeddingPair2simil_fn, 
                        inScope_cubes_vs_views = self.inScope_cubes_vs_views, 
                        viewPairs = self.viewPairs, 
                        N_views = self.N_views,
                        batchSize = self._batchSize_similNet_embeddingPair2simil)   # (N_cubes, N_viewPairs), TODO: need to set the dissimil value of the viewPairs with at least one invalid_view to 0.


                self.validCubes = earlyRejection.selectFromSimilarity(
                    dissimilarityProb = self.dissimilarity, 
                    N_viewPairs4inference = self.N_viewPairs4inference[0])# (N_cubes,) np.bool  ######%%%%%%%!!!!!
            #self.validCubes = self.train_rejection(dataset = dataset)


                self.validCubesIndex = [i for i, x in enumerate(self.validCubes) if x]

                self.N_validCubes = self.validCubes.sum()
                print("\nEarly rejection step reduced the # of cubes from {} to {}.".format(self.N_cubes, self.N_validCubes))
            else:
                image_h, image_w, _ = images_list[0].shape
                index = (self.img_w_cubesCenter >= (self.resol * self._cube_D * 3**0.5/2)) * (self.img_w_cubesCenter <= (image_w - self.resol * self._cube_D * 3**0.5/2)) *\
                        (self.img_h_cubesCenter >= (self.resol * self._cube_D * 3**0.5/2)) * (self.img_h_cubesCenter <= (image_h - self.resol * self._cube_D * 3**0.5/2))
                self.inScope_cubes_vs_views = index.T
                self.patches_embedding = np.zeros((self.N_cubes, self.N_views, self.D_embedding), dtype=np.float32)
                for i_view, image in enumerate(images_list):
                    for i_cubic in range(self.N_cubes):
                        loc_w = int(self.img_w_cubesCenter[i_view,i_cubic])
                        loc_h = int(self.img_h_cubesCenter[i_view,i_cubic])
                        if((loc_h >= (self.D_embedding/12)) and (loc_h <= (image_h - self.D_embedding/12)) and (loc_w >= (self.D_embedding/12)) and (loc_w <= (image_w - self.D_embedding/12))):
                            patch_image = np.concatenate((image[(loc_h - self.D_embedding/12):(loc_h + self.D_embedding/12),loc_w,:], image[loc_h,(loc_w - self.D_embedding/12):(loc_w + self.D_embedding/12),:]), axis = 1)
                            self.patches_embedding[i_cubic, i_view] = (patch_image.flatten())/256.0
                
                self.dissimilarity = np.zeros((self.N_cubes, (self.N_views * (self.N_views-1)) / 2))

                self.validCubes = (index.sum(axis = 0) >= 5)
                self.validCubesIndex = [i for i, x in enumerate(self.validCubes) if x]
                self.N_validCubes = self.validCubes.sum()
               
        elif(self._datasetName == 'MVS2d'):
            image_shape_0, image_shape_1, _ = images_list[0].shape  
            index = (self.img_w_cubesCenter >= (self.resol * self._cube_D/(2**0.5))) * (self.img_w_cubesCenter <= (image_shape_0 - self.resol * self._cube_D/(2**0.5)))
            self.inScope_cubes_vs_views = index.T
            self.patches_embedding = np.zeros((self.N_cubes, self.N_views, self.D_embedding), dtype=np.float32)
            for i_view, image in enumerate(images_list):
                for i_cubic in range(self.N_cubes):
                    loc = int(self.img_w_cubesCenter[i_view,i_cubic])
                    if((loc >= self.D_embedding/6) and (loc <= (image_shape_0 - self.D_embedding/6))):
                        self.patches_embedding[i_cubic, i_view] = (image[(loc - self.D_embedding/6):(loc + self.D_embedding/ 6),0,:].flatten())/256.0
            self.dissimilarity = np.zeros((self.N_cubes, (self.N_views * (self.N_views-1)) / 2))
            
            self.validCubes = (index.sum(axis = 0) >= 5)
            self.validCubesIndex = [i for i, x in enumerate(self.validCubes) if x]
            self.N_validCubes = self.validCubes.sum()
        
    
    def viewSelection(self,mode = 'train'):
        
        if(mode == 'train'):
            self.viewPairs4Reconstr, self.embeddings = self.viewPairSelection.viewPairSelection_random(
                cameraTs_np = self.cameraTs_np,  \
                e_viewPairs = self.patches_embedding,  \
                d_viewPairs = self.dissimilarity,  \
                validCubes = self.validCubes,  \
                inScope_cubes_vs_views = self.inScope_cubes_vs_views,
                cubeCenters_xyz = self.cubes_param_np['xyz'] + self.cube_D_mm / 2., \
                viewPairs = self.viewPairs,
                N_validCubes = self.N_validCubes,
                N_random = self.N_viewPairs4inference[0])
            
        elif(mode == 'inference'):
            
            self.viewPairs4Reconstr, self.w_viewPairs4Reconstr = viewPairSelection.viewPairSelection( \
                cameraTs_np = self.cameraTs_np,  \
                e_viewPairs = self.patches_embedding,  \
                d_viewPairs = self.dissimilarity,  \
                validCubes = self.validCubes,  \
                cubeCenters_xyz = self.cubes_param_np['xyz'] + self.cube_D_mm / 2., \
                viewPair_relativeImpt_fn = viewPair_relativeImpt_fn,  \
                batchSize = params.__batchSize_viewPair_w,  \
                N_viewPairs4inference = N_viewPairs4inference, \
                viewPairs = viewPairs)     # (N_validCubes, N_viewPairs4inference, 2), (N_validCubes, N_viewPairs4inference)

            if params.__weighted_fusion is False:
                w_viewPairs4Reconstr[:] = 1.0 / N_viewPairs4inference


class Recorder():
    def __init__(self):
        self.init_data()
    
    def init_data(self):
        
        self.model_list = []
        
        self.images_list_dict = {}
        self.cameraPOs_np = None
        self.cameraTs_np = None
        self.volumn_dict = {}
        self.ply_dict = {}
    
    def record_flag(self, model_num):
        '''
        check whether the model has been initiatied
        
        input:the i'th model number
        output:bool
            True:has been initiated
        '''
        if model_num in self.model_list:
            return True
        else:
            self.model_list.append(model_num)
            return False
    
    def record_data(self, 
                    model_num,
                    images_list,
                    cameraPOs_np,
                    cameraTs_np,
                    volumn,
                    ply_dict = None,
                    ):
        
        self.record_flag(model_num)
        
        #self.images_list_dict[model_num] = images_list
        self.cameraPOs_np = cameraPOs_np
        self.cameraTs_np = cameraTs_np
        self.volumn_dict[model_num] = volumn
        #self.ply_dict[model_num] = ply_dict
        
        
from torch.utils import data

class Dataset(data.Dataset, Params):
    
    def __init__(self, model_num, recorder, reconstruct = False, transform = None, resol = 1.0):
        super(Dataset, self).__init__()
        self.recorder = recorder
        self.resol = resol
        self.reconstruct = reconstruct
        self.init_tools(model_num)
        self.init_data(model_num)
        self.transform = transform
        
        
        
    
    def init_tools(self, model_num):
        self.camera = Camera()
        self.select = ViewPairSelection()
        self.image = Image()
        self.volumn = Volumn(resol = self.resol)
        self.cvc = CVC()
        self.ply = PLY2volumn(model_num)
        
    
    def init_data(self, model_num):
        
        ttt0 = time.time()
        ##########################################################33
        self.load_modelSpecific_params(self._datasetName, model_num)
        if (not(self.recorder.record_flag(model_num)))or self.reconstruct:   
            self.images_list = self.image.readImages(
                datasetFolder = self.datasetFolder, 
                imgNamePattern = self.imgNamePattern, 
                viewList = self.viewList, 
                return_list = True)    

            self.cameraPOs_np = self.camera.readCameraPOs_as_np(
                datasetFolder = self.datasetFolder, 
                datasetName = self._datasetName, 
                poseNamePattern = self.poseNamePattern, 
                #model = self.model, 
                viewList = self.viewList)  # (N_views, 3, 4) np

            self.cameraTs_np = self.camera.cameraPs2Ts(cameraPOs = self.cameraPOs_np)


            ##########################################################
            self.volumn.init_volumn(self.images_list,
                                   self.cameraPOs_np,
                                   self.cameraTs_np)
            self.volumn.early_rejection(images_list = self.images_list)

            #####################################################################
            self.ply.init(min_point = self.volumn.cubes_param_np['xyz'][0], 
                         max_point = self.volumn.cubes_param_np['xyz'][-1],
                         resol = self.volumn.cubes_param_np['resol'][0])

            print('ply_init finished')
            #if not(self.reconstruct):
            ttt1 = time.time()
            print('time cost: ', time.time() - ttt0)
            self.volumn.train_rejection(self)##################################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            print('train_rejection finished')
            ttt2 = time.time()
            print('time cost: ', time.time() - ttt1)

            #######################################################
            self.volumn.viewSelection(mode = 'train')

            
            self.recorder.record_data(model_num,
                                None,
                                self.cameraPOs_np,
                                self.cameraTs_np,
                                self.volumn,
                                #self.ply
                                )
            print('init_data_finished')
            print('time cost: ', time.time() - ttt2)
        
        else:
            self.images_list = self.image.readImages(
                datasetFolder = self.datasetFolder, 
                imgNamePattern = self.imgNamePattern, 
                viewList = self.viewList, 
                return_list = True)
            #self.images_list = self.recorder.images_list_dict[model_num] 
            self.cameraPOs_np = self.recorder.cameraPOs_np
            self.cameraTs_np = self.recorder.cameraTs_np
            self.volumn = self.recorder.volumn_dict[model_num]
            self.ply.init(min_point = self.volumn.cubes_param_np['xyz'][0], 
                         max_point = self.volumn.cubes_param_np['xyz'][-1],
                         resol = self.volumn.cubes_param_np['resol'][0])
            #self.ply = self.recorders.ply_dict[model_num] 
        
        
    def __len__(self):
        return self.volumn.N_validCubes
    
    def __getitem__(self, index):
        
        cvc, embeddings, idx, idx_validCubes= self.transform_input(index)
        cvc = cvc/256.0 - 0.5
        surface = self.transform_label(index)
        
        sample = {'cvc':torch.from_numpy(cvc).float(), 
                  'embedding':torch.from_numpy(embeddings).float(),
                  'surface':torch.from_numpy(surface).float(),
                  'idx': idx,
                  'idx_validCubes':idx_validCubes
                 }
        
        if self.transform:
            sample = self.transform(sample)
        
        return (sample)
    
    def transform_input(self, idx):
        
        idx_validCubes = self.volumn.validCubesIndex[idx]
        self._CVCs1_sub = self.cvc.gen_coloredCubes(
                                        selected_viewPairs = self.volumn.viewPairs4Reconstr[idx],
                                        xyz = self.volumn.cubes_param_np['xyz'][idx_validCubes],
                                        resol = self.volumn.cubes_param_np['resol'][idx_validCubes],
                                        colorize_cube_D = self.volumn._cube_D,\
                                        cameraPOs=self.cameraPOs_np, \
                                        models_img=self.images_list, \
                                        visualization_ON = False) 
        #self.cubes_param_np = 
        return (self._CVCs1_sub, 
                self.volumn.embeddings[idx], 
                np.array(idx),
                np.array(idx_validCubes))
    
    
    def transform_label(self, idx):
        idx_validCubes = self.volumn.validCubesIndex[idx]
        self.surface = self.ply.gen_idx(voxel_point = self.volumn.cubes_param_np['xyz'][idx_validCubes],
                                    resol = self.volumn.cubes_param_np['resol'][idx_validCubes])
        return (self.surface)



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
                print(prediction[_n].shape)
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
        #self.save2ply(ply_filePath, xyz_np, rgb_np, normal_np)
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



