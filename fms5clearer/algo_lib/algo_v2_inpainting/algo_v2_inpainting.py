import yaml
import numpy as np
import cv2
import scipy as sp
from scipy.sparse import csc_matrix

from fms5clearer.func_box.deblur_func import FTL1_v4, FTL2, run_bm3d_deblurring
from fms5clearer.param_lib.param_TV import params as params_TV
from fms5clearer.algo_lib.deblurrer_base import DeblurrerBase

class Algo(DeblurrerBase):
    def __init__(self, algo_param_path):
        super(Algo, self).__init__()
        with open(algo_param_path, 'r') as stream:
            self.algo_param = yaml.load(stream)
        self.sat_threshold = self.algo_param['sat_threshold']
        self.dilate_kernel = self.algo_param['dilate_kernel']
        self.dilate_kernel2 = self.algo_param['dilate_kernel2']
        self.bm3d_sigma = self.algo_param['bm3d_sigma']
        self.para = params_TV(mu2 = 30, mu3 = 60)

    def run_deblur(self, b_img, kernel):
        self.ind = self.get_sat_idx(b_img, self.sat_threshold, self.dilate_kernel)
        deblur_sat_result_dict = self.deblur_sat(b_img, kernel, self.para)
        self.out_sat = deblur_sat_result_dict['sat_result']
        self.B_w_sat_fill = deblur_sat_result_dict['B_w_sat_fill']
        self.out_nonsat = self.deblur_non_sat(self.B_w_sat_fill, kernel, para = self.para)
        
        out_combine = np.clip(self.out_sat * self.ind + self.out_nonsat, 0,1)
        return out_combine

    def deblur_sat(self, B_ori, kernel, para):
        result_dict = {}
        if np.sum(self.ind.astype('uint8')) >0:
            print('====Using sat mode')
            # bcs img would be rewrited by "self._saturation_fill", we make a copy.
            _B = B_ori.copy()
            B_sat_fill = self._inpainting_fill(_B, self.ind)
            print('>>>>sat fill')
            B_res = B_ori - B_sat_fill
            
            _ind = np.zeros(self.ind.shape, dtype='bool')
            print('>>>>>> deblur_sat_TV: 1/2')
            # blurrer
            self.im_res_1 = FTL1_v4(B_res, B_res, kernel, _ind, para, mu = para.mu1, mini_iter = para.mini_iter_sat)
            print('>>>>>> deblur_sat_TV: 2/2')
            # clearer, which could lead more ringing
            self.im_res_3 = FTL1_v4(B_res, B_res, kernel, _ind, para, mu = para.mu3, mini_iter = para.mini_iter_sat)

            result = self.remove_ringing_pattern(self.im_res_3, self.im_res_1, iter_num = 1)
        else:
            B_sat_fill = B_ori.copy()
            print('====Using non-sat mode')
            result = np.zeros(B_ori.shape)
        result_dict['B_w_sat_fill'] = B_sat_fill
        result_dict['sat_result'] = result

        return result_dict

    def deblur_non_sat(self, B_sat_fill, kernel, para):
        print('>>>>>> deblur_nonsat_TV: 1/4')
        im_L2 = FTL2(B_sat_fill, kernel, mu=para.L2ratio, relchg_cond = 5e-3)
        print('>>>>>> deblur_nonsat_TV: 2/4')
        im_1 = FTL1_v4(im_L2, B_sat_fill, kernel, self.ind, para, mu = para.mu1, mini_iter = para.mini_iter_nonsat)
        print('>>>>>> deblur_nonsat_TV: 3/4')
        im_2 = FTL1_v4(im_L2, B_sat_fill, kernel, self.ind, para, mu = para.mu2, mini_iter = para.mini_iter_nonsat)
        print('>>>>>> deblur_nonsat_TV: 4/4')
        im_3 = FTL1_v4(im_L2, B_sat_fill, kernel, self.ind, para, mu = para.mu3, mini_iter = para.mini_iter_nonsat)
        result_a = self.remove_ringing_pattern(im_2, im_1, iter_num = 6)
        result_b = self.remove_ringing_pattern(im_3, im_1, iter_num = 6)
        result_TV = result_a/2 + result_b/2
       
        _result_dict1 = run_bm3d_deblurring(blur_img = B_sat_fill, sigma = self.bm3d_sigma, blur_kernel = kernel)
        _im = _result_dict1['result_wo_clip']
        result_bm3d = _im.copy()       
        
        self._sat_idx = self.get_sat_idx(_im, 1, dilate_kernel = self.dilate_kernel2)
        result = result_TV * self._sat_idx + result_bm3d  * (1-self._sat_idx)

        return result

