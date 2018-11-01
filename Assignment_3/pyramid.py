import matplotlib.pyplot as plt
import numpy as np
import cv2
import copy
import argparse
from scipy.spatial import Delaunay
import os
import scipy.ndimage
import math

# Apply kernel and downsample
def pyramid_down(img,sigma=1):
    blur = cv2.GaussianBlur(img,(5,5),0)
    #downsampling
    return blur[::2,::2,:]

# Upsampling and Resizing
def pyramid_up(img,dest_size=None):
    upsampled_image=scipy.ndimage.zoom(img,[2,2,1], order=3)
    
    if(dest_size is None):
        dest_size=upsampled_image.shape
    return upsampled_image[0:dest_size[0],0:dest_size[1],:]

def gaussian_pyramid(img,levels=5):
    height, width, depth=img.shape
    levels=min(levels,int(math.log(min(height,width),2)))
    image_list=[img]

    for level in range(levels):
        img=pyramid_down(img)
        image_list.append(img)

    return image_list

def laplacian_pyramid(img=None,levels=None,gaussian_pyramid_list=None):

    laplacian_pyramid_list=[]

    if(gaussian_pyramid_list is None):
        gaussian_pyramid_list=gaussian_pyramid(img,levels)

    levels=len(gaussian_pyramid_list)
    for idx in range(levels-1):
        laplacian_pyramid_list.append(gaussian_pyramid_list[idx]-pyramid_up(gaussian_pyramid_list[idx+1],gaussian_pyramid_list[idx].shape))
    laplacian_pyramid_list.append(gaussian_pyramid_list[levels-1])

    return laplacian_pyramid_list


def reconstruct_from_laplacian(laplacian_pyramid):
    x=len(laplacian_pyramid)

    lap=laplacian_pyramid[x-1]
    reconstructed_list=[lap]

    for i in range(x-2,-1,-1):
        lap=pyramid_up(lap,laplacian_pyramid[i].shape)
        h,w,d=lap.shape
        lap=laplacian_pyramid[i][0:h,0:w,0:d]+lap
        reconstructed_list.append(lap)

    return reconstructed_list


def main():
    parser=argparse.ArgumentParser(description='Image Pyramid')
    parser.add_argument('--source', dest='source_image', help="Enter Source Image Path", required=True, type=str)
    parser.add_argument('--levels', dest='levels', help="Enter no. of levels for pyramid", default=5, type=int)
    parser.add_argument('--output', dest='output_image_folder', help="Enter Output Image folder", required=True, type=str)
    args=parser.parse_args()

    os.system("mkdir -p "+args.output_image_folder)
    img=cv2.imread(args.source_image, cv2.IMREAD_COLOR)
    (height,width,depth)=img.shape

    gaussian_pyramid_list=gaussian_pyramid(img,levels=args.levels)
    laplacian_pyramid_list=laplacian_pyramid(gaussian_pyramid_list=gaussian_pyramid_list)

    counter=0
    for image in gaussian_pyramid_list:
        counter+=1
        path=os.path.join(args.output_image_folder,"gaussian_"+str(counter)+".jpg")
        cv2.imwrite(path,cv2.resize(image,(width,height)))
    counter=0
    for image in laplacian_pyramid_list:
        counter+=1
        path=os.path.join(args.output_image_folder,"laplacian_"+str(counter)+".jpg")
        cv2.imwrite(path,cv2.resize(image,(width,height)))

if __name__=='__main__':
    main()