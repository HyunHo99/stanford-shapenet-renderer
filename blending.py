from PIL import Image
import os
import glob
import argparse
import cv2 as cv

parser = argparse.ArgumentParser()
parser.add_argument('--work_dir', type=int, default=30,
                    help='number of views to be rendered')

work_dir="./"
out_dir = os.path.join(work_dir, 'blended')
os.makedirs(out_dir, exist_ok=True)
os.makedirs(os.path.join(out_dir, 'image'), exist_ok=True)

image_list = glob.glob(os.path.join(work_dir, 'image/pert_*.png'))
if(len(image_list)==0):
    print("Cannot find input images!")
    
image_list.sort()

for i in range(0, len(image_list), 2):
    img_1 = cv.imread(image_list[i])
    img_2 = cv.imread(image_list[i+1])
    img_blended = img_1*0.5 + img_2*0.5
    cv.imwrite(os.path.join(out_dir, 'image', '{:0>3d}.png'.format(i//2)), img_blended)