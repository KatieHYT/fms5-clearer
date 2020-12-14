# fms5-clearer

## Getting Started (Ubuntu)
0. Install Docker  
   Here's the [official guide.](https://docs.docker.com/get-docker/)  
1. Clone this repository
   ```bash
   git clone https://github.com/KatieHYT/fms5-clearer.git
   ```
2. Build and run image using Dockerfile from the root directory
   ```bash
   bash build_env.sh
   ```
3. Attach the container
   ```bash
   bash exec_cntnr.sh
   ```
4. Run a sample
   ```bash
   python test.py
   ```
5. Go to directory fms5-clearer/test_out and check the result  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; kernel  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img src="./sample/input/testkernel.png">  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; input image / output image  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img src="./sample/input/test255_800.png" width="300">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img src="./sample/output/output_img.png" width="300">

## Getting Started (Windows)
For windows, you might need to install git-bash, and then just follow the above steps for Ubuntu.  
You might encounter some error message like "the input device is not a TTY. If you are using mintty, try prefixing the command with 'winpty'" while running step-3.  
If so, edit exec_cntnr.sh, add 'winpty' before the command:
```
winpty docker exec -it fms5_cntnr bash  
```
This "might" help to resolve the error according to [this stackoverflow page.](https://stackoverflow.com/questions/48623005/docker-error-the-input-device-is-not-a-tty-if-you-are-using-mintty-try-prefi)



## Running Your Own Data from test.py
1. Create or modify your setting file if needed  
&nbsp;&nbsp;Go to fms5-clearer/sample/setting/
```
   content_patch_w: 800 ----> image width of your input image 
   content_patch_h: 800 ----> image height of your input image 
   img_range: 255 ----> either 4095(0-4095) or 255(0-255)
```

2. Specify your path  
&nbsp;&nbsp;Go to fms5-clearer/test.py
```
RAW_IMG_PATH = './your/own/input_img.png' ----> the path to your input image 
PRESET_KERNEL_PATH = './your/own/kernel.png' ----> the path to your kernel
SETTING_DICT_PATH='./your/own/settting.yaml' ----> the path to your setting
```

