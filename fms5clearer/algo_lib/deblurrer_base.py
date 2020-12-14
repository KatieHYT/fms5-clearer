import numpy as np 
import cv2
import scipy as sp
from scipy.sparse import csc_matrix
from skimage.restoration import inpaint

from abc import ABCMeta
from abc import abstractmethod

class DeblurrerBase(object):
    __metaclass__ = ABCMeta


    def __init__(self, ):
        pass
    
    def remove_ringing_pattern(self, clearer_img, blurer_img, iter_num):
        '''
        clearer_img: clearer but typically would have ringing
        blurrer_img: not clearer enough but w/o ringing or with less ringing
        '''
        # ringing pattern = the clearer(with ringing) - blurrer(w/o ringing)
        ring_pattern = clearer_img - blurer_img;
        # filter out some noise?
        for i in range(iter_num):
            ring_pattern_filt = cv2.bilateralFilter(ring_pattern.astype('float32'), 5, 8, 1)
            # clearer - ringing pattern
            result = clearer_img - ring_pattern_filt;
            ring_pattern = clearer_img - result
        return result

    def _saturation_fill(self, img, ind):
        '''
        !!!WARNING!!! This function would force inplace rewrite img
        '''
        [m,n] = img.shape
        resh_img = img.reshape(m*n, 1)
        resh_ind = ind.reshape(m*n, 1)
        b_ind = (resh_ind == 0).nonzero()[0]
        i = range(m*n)
        in_ind = np.setdiff1d(i, b_ind)
        k1 = -4*np.ones((m*n), dtype=int)
        k2 = np.ones((m*n-1), dtype=int)
        k3 = np.ones((m*n-n), dtype=int)
        is1 = i[0:m*n-1]
        is2 = i[1:m*n]
        is3 = i[0:m*n-n]
        is4 = i[0+n:m*n]
        total_i = np.concatenate((i, is1, is2, is3, is4),axis=0)
        total_j = np.concatenate((i, is2, is1, is4, is3),axis=0)
        total_k = np.concatenate((k1, k2, k2, k3, k3))
        A = csc_matrix((total_k[:], (total_i,total_j)), shape=(m*n,m*n))
        F = -A[:,b_ind]*resh_img[b_ind]
        resh_img[in_ind,0] = sp.sparse.linalg.spsolve(A[in_ind][:,in_ind],F[in_ind])
        tile_img = resh_img.reshape(m,n);
        return tile_img

    def _mirror_fill(self, data, invalid=None):
        """
        Replace the value of invalid 'data' cells (indicated by 'invalid')
        by the value of the nearest valid data cell
        
        Input:
            data:    numpy array of any dimension
            invalid: a binary array of same shape as 'data'. True cells set where data
                     value should be replaced.
                     If None (default), use: invalid  = np.isnan(data)
        
        Output:
            Return a filled array.
        """
        import numpy as np
        import scipy.ndimage as nd
        
        if invalid is None: invalid = np.isnan(data)
        
        ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
        return data[tuple(ind)]
    
    def _inpainting_fill(self, img, mask):
        """ 
        
        """
        
        img[np.where(mask ==1)] =1
        image_result = inpaint.inpaint_biharmonic(img, mask,
                                          multichannel=False)
        return image_result
        
    def get_sat_idx(self, B, sat_threshold, dilate_kernel = (31,31)):
        '''
        get saturation index, and return a dilate one 
        '''
        ind = B > sat_threshold
        se = np.ones(dilate_kernel, np.uint8)        
        ind = cv2.dilate(ind.astype('uint8'), se).astype('bool')
        return ind


