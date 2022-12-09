#%% Imports
import os
import subprocess
import glob
# os.listdir()
# os.chdir("video")

import shutil

os.chdir("runs/2022-12-08_12-53-22_SAC_HalfCheetah-v2_Gaussian_autotune/episode10000")

#shutil.copy('drone_1.png', 'test.png')

print(glob.glob('*.png'))
for file_name in glob.glob('*.png'):
    num = file_name.replace('drone_','')
    num = num.replace('.png','')
    num = int(num)
    try:
        shutil.copy(file_name, "drone_%03d.png" % num)
        os.remove(file_name)
    except:
        pass    
#    os.rename(file_name, "/drone_%03d.png" % num)
#
##os.rename(src, dst, *, src_dir_fd=None, dst_dir_fd=None)
#
subprocess.call([
    'ffmpeg', '-framerate', '8', '-i', 'drone_%03d.png', '-r', '30', '-pix_fmt', 'yuv420p',
    'video.mp4'
])
#subprocess.call([
#    'ffmpeg', '-framerate', '8', '-pattern_type', 'glob', '-i', '*.png', '-r', '30', '-pix_fmt', 'yuv420p',
#    'video.mp4'
#])
#for file_name in glob.glob("*.png"):
#    os.remove(file_name)
