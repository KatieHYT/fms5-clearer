from pprint import pprint
import yaml
import json
import os 
import numpy as np 
import skimage.io as io
from datetime import datetime
from tqdm import tqdm
from shutil import copyfile

from fms5clearer.algorithm_builder import algo_builder
from fms5clearer.func_box.tile_generator import tile_generator_direct

class DeblurrerFormosa5():
    def __init__(self, img_range, if_tile_gen):
        '''
        params:
            img_range:
                1   : 0-1 (nothing change) 
                255 : 0-255
                4095: 0-4095

        '''
        self.img_range = img_range
        self.if_tile_gen = if_tile_gen
    
    def prepare_img(self, raw_img_path, border):
        """
        Load image and then pad it with border to tackle blocking artifact due to total variation method.
        """
        raw_img = io.imread(raw_img_path)
        raw_img = self.normalB(raw_img)
        assert len(raw_img.shape) == 2, 'img dim must be (W,H)'
        assert border >= 0, 'border must be >= 0'
        
        if border == 0:
            img = raw_img.copy()
        else: 
            img_w = raw_img.shape[0]
            img_h = raw_img.shape[1]
            img = np.empty(shape =  (img_w+border*2, img_h+border*2))
            canvas_w = img.shape[0]
            canvas_h = img.shape[1]
            num = int(np.ceil(canvas_w/img_w))
            img[border:border+img_w, border:border+img_h] = raw_img
            
            # border left, 
            num = int(np.ceil(img_w/border))
            for i in range(num):
                w1 = (i+1)*border
                w2 = (i+2)*border
                h1 = 0
                h2 = border
                reflect_patch = np.fliplr(raw_img[(i)*border:(i+1)*border , h1:h2])
                img[w1:w2, h1:h2] = reflect_patch

            # border right
            for i in range(num):
                w1 = (i+1)*border
                w2 = (i+2)*border
                h1 = -border
                reflect_patch = np.fliplr(raw_img[(i)*border:(i+1)*border , h1:])
                img[w1:w2, h1:] = reflect_patch

            # border top 
            num = int(np.ceil(img_w/border))
            for i in range(num+2):
                w1 = 0
                w2 = border
                h1 = (i)*border
                h2 = (i+1)*border
                reflect_patch = np.flipud(img[w2:w2+border, h1:h2])
                img[w1:w2, h1:h2] = reflect_patch

            # border down
            for i in range(num+2):
                w1 = -border*1
                h1 = (i)*border
                h2 = (i+1)*border
                reflect_patch = np.flipud(img[w1-border:w1, h1:h2])
                img[w1:, h1:h2] = reflect_patch

        print('Done prepared image.')
        return img
    
    def prepare_kernel(self, k_path):
        """
        Load kernel and make its range [0,1].
        """
        raw_k = io.imread(k_path)
        kernel = self.normalK(raw_k)
        print('Done prepared kernel.')
        return kernel

    def normalB(self, b_img):
        return b_img* (1/self.img_range)
    
    def normalB_inv(self, I, img_range):
        _I = I * img_range
        I = _I.astype('uint8')
        return I

    def normalK(self, kernel):
        return kernel.astype('float32') / kernel.sum()
    
    def get_content_idx(self, shape, dm, dn):
        if shape[0]%2==0:
            x = (int(dm/2), int(shape[0]-dm/2))
        else:
            x = (int(dm+1/2), int(shape[0]-(dm-1)/2))

        if shape[1]%2==0:
            y = (int(dn/2), int(shape[1]-dn/2))
        else:
            y = (int((dn+1)/2), int(shape[1]-(dn-1)/2))

        return x, y

    def build_algo(self, algo_dict):
        ALGO = algo_builder(algo_dict['algo_name'])
        AG = ALGO(algo_param_path = algo_dict['algo_param_path'])
        return AG
    
    def save_out(self, output_img, args_dict, algo_dict, save_dir):
        print('Saving out...to {}'.format(save_dir))
        save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        path_input = os.path.join(save_dir, 'input_img.png')
        path_kernel = os.path.join(save_dir, 'kernel.png')
        path_output = os.path.join(save_dir, 'output_img.png')
        path_notes = os.path.join(save_dir, 'notes.json')
        path_setting = os.path.join(save_dir, 'setting.yaml')
        path_algo_param = os.path.join(save_dir, 'algo_param.yaml')
        copyfile(args_dict['raw_img_path'], path_input)
        copyfile(args_dict['preset_kernel_path'], path_kernel)
        copyfile(args_dict['setting_dict_path'], path_setting)
        copyfile(algo_dict['algo_param_path'], path_algo_param)
        io.imsave(path_output, output_img)
        with open(path_notes, 'w') as f:
            json.dump(args_dict, f)
        print('Done saving to {}'.format(save_dir))

    def run_algo(self, b_img_norm, kernel_norm, if_tile_gen):
        assert len(b_img_norm.shape) == 2, 'img shape should be (W,H, C)'
        if if_tile_gen == True:
            b_img_norm_pad, dm, dn = tile_generator_direct(b_img_norm)
            content_x_minmax, content_y_minmax = self.get_content_idx(b_img_norm_pad.shape, dm, dn)
            _out = self.AG.run_deblur(b_img_norm_pad, kernel_norm)
            out = _out[content_x_minmax[0]:content_x_minmax[1], content_y_minmax[0]:content_y_minmax[1]]
        else:
            out = self.AG.run_deblur(b_img_norm, kernel_norm)
        return out
    
    def batch_process(self, img, kernel, content_patch_w, content_patch_h, border):
        img_w = img.shape[0]
        img_h = img.shape[1]
        content_img_w = img_w-border*2
        content_img_h = img_h-border*2
        patch_num_w = int(np.ceil(content_img_w/content_patch_w))
        patch_num_h = int(np.ceil(content_img_h/content_patch_h))
        out_img = np.empty(shape= (content_img_w, content_img_h))
        pbar1 = tqdm(total=patch_num_w)
        pbar2 = tqdm(total=patch_num_h)

        for i in range(patch_num_w):
            for j in range(patch_num_h):
                w1, h1 = content_patch_w*i, content_patch_h*j
                w2 = img_w if i == patch_num_w else content_patch_w*(i+1)
                h2 = img_h if j == patch_num_h else content_patch_h*(j+1)
                _img = img[w1:w2+border*2,h1:h2+border*2]
                out = self.run_algo(_img, kernel, if_tile_gen = self.if_tile_gen)
                y_est = out.copy()
                y_est = y_est[border:content_patch_w+border:, border:content_patch_h+border]
                out_img[w1:w2, h1:h2] = y_est
                pbar1.update()
            pbar2.update()
        pbar1.close()
        pbar2.close()
        print('Done deblurring!')
        out_img = np.clip(out_img, 0, 1)
        return out_img 
    
    def run(self, args_dict, algo_dict, outer_dict):
        st = datetime.now()
        print('Start: ', st)
        pprint('Using algo dict:')
        pprint(algo_dict)
        pprint('Using outer dict:')
        pprint(outer_dict)
        img = self.prepare_img(border = outer_dict['border'], raw_img_path = args_dict['raw_img_path'])
        kernel = self.prepare_kernel(k_path = args_dict['preset_kernel_path'])
        self.AG = self.build_algo(algo_dict)
        _out = self.batch_process(img = img, kernel = kernel,\
             content_patch_w = outer_dict['content_patch_w'], content_patch_h = outer_dict['content_patch_h'],\
             border = outer_dict['border']) 
        out_img = self.normalB_inv(_out, 255)
        ed = datetime.now()
        t_cost = ed-st
        print('End: ', ed, 'Cost:', t_cost)
        args_dict['time_cost'] = str(t_cost)
        self.save_out(output_img = out_img, args_dict = args_dict, algo_dict = algo_dict, save_dir = args_dict['save_dir'])
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_img_path', default = None)
    parser.add_argument('--preset_kernel_path', default = None)
    parser.add_argument('--save_dir', default = None)
    parser.add_argument('--setting_dict_path', default = None)

    args = parser.parse_args()
    args_dict = args.__dict__
    pprint('Using args:')
    pprint(args)
    
    with open(args_dict['setting_dict_path'], 'r') as stream:
        setting_dict = yaml.load(stream)
        
    algo_dict = setting_dict['algo']
    outer_dict = setting_dict['outer']
    DBLR = DeblurrerFormosa5(img_range = outer_dict['img_range'], if_tile_gen = outer_dict['if_tile_gen'])
    DBLR.run(args_dict, algo_dict, outer_dict)
