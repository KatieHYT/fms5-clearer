import os
import yaml
from fms5clearer.DeblurrerFormosa5 import DeblurrerFormosa5

# RAW_IMG_PATH = os.path.join('./fms5-clearer/sample/input/test4095_800.png')
RAW_IMG_PATH = './fms5-clearer/sample/input/test255_800.png'
PRESET_KERNEL_PATH = './fms5-clearer/sample/input/testkernel.png'
SAVE_DIR='/local_vol/test_out'
SETTING_DICT_PATH='./fms5-clearer/sample/setting/setting_test.yaml'

args_dict = {}
args_dict['save_dir'] = SAVE_DIR
args_dict['setting_dict_path'] = SETTING_DICT_PATH
args_dict['raw_img_path'] = RAW_IMG_PATH
args_dict['preset_kernel_path'] = PRESET_KERNEL_PATH

with open(args_dict['setting_dict_path'], 'r') as stream:
    setting_dict = yaml.load(stream)
algo_dict = setting_dict['algo']
outer_dict = setting_dict['outer']

DBLR = DeblurrerFormosa5(img_range = outer_dict['img_range'], if_tile_gen = outer_dict['if_tile_gen'])
DBLR.run(args_dict, algo_dict, outer_dict)
