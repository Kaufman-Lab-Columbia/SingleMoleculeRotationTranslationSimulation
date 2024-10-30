# -*- coding: utf-8 -*-
"""

Takes all frame files output from simulation and generates a single .tif containing them all.

@author: Alec

"""

import os
import sys
import numpy as np
from skimage import io

# Get user input, initiate output name and pass if folder is not for lc_frames or rc_frames
# folder = input("What is the folder path?\n")
folder = os.getcwd()

check = folder.rpartition('\\')[2]

if check == 'lc_frames':
    outpath = folder.rpartition('\\')[0] + r"\LC.tif" 
elif check == 'rc_frames':
    outpath = folder.rpartition('\\')[0] + r"\RC.tif" 
elif check =='lc_frames1':
        outpath = folder.rpartition('\\')[0] + r"\LC1.tif"
elif check =='lc_frames2':
        outpath = folder.rpartition('\\')[0] + r"\LC2.tif"
elif check =='rc_frames1':
        outpath = folder.rpartition('\\')[0] + r"\RC1.tif"
elif check =='rc_frames2':
        outpath = folder.rpartition('\\')[0] + r"\RC2.tif"

else:
    sys.exit('Terminating... Not a LC or RC Frames Folder...')

# Compute number of files
count = 0
for filename in os.listdir(folder):
    filepath = os.path.join(folder, filename)
    if os.path.isfile(filepath):
        count += 1
        
# Create array --> assuming movie is 256x512
movie_arr = np.empty((count, 300, 300))

frame = 0
for filename in os.listdir(folder):
    
    # if frame % 500 == 0:
    #     print("On frame: ", frame, "\n")
    
    filepath = os.path.join(folder, filename)
    frame_arr = np.loadtxt(filepath, delimiter=',', usecols=np.arange(300))
    movie_arr[frame, :, :] = frame_arr
    frame += 1

# Save output
io.imsave(outpath, movie_arr) 