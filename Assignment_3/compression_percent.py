import numpy as np
import cv2
import copy
import argparse
from scipy.spatial import Delaunay
import os
import scipy.ndimage
import math
import pyramid

parser=argparse.ArgumentParser(description='Image Mosaicing')
parser.add_argument('--source', dest='source_image', help="Enter Source Image Path", required=True, type=str)
parser.add_argument('--levels', dest='levels', help="Enter no. of levels for pyramid", default=3, type=int)
parser.add_argument('--percent', dest='percent', help="Enter percent of co-efficient to keep", default=5, type=int)
parser.add_argument('--output', dest='output_image_folder', help="Enter Output Image folder", required=True, type=str)
args=parser.parse_args()

os.system("mkdir -p "+args.output_image_folder)

src_img=cv2.imread(args.source_image, cv2.IMREAD_COLOR)
(src_height,src_width,src_depth)=src_img.shape

src_gaussian_pyramid_list=pyramid.gaussian_pyramid(src_img,levels=args.levels)
src_laplacian_pyramid_list=pyramid.laplacian_pyramid(gaussian_pyramid_list=src_gaussian_pyramid_list)

x=len(src_laplacian_pyramid_list)
counter=0

for image in src_laplacian_pyramid_list:
    counter+=1
    if(counter==x):
        break
    ravel=np.ravel(image)
    to_keep=int(ravel.shape[0]*(args.percent/100.0))
    #val stores largest to_keep value
    val=ravel[np.argsort(ravel)[-to_keep]]
    image[image[:,:,:]<val]=0

reconstructed_image=pyramid.reconstruct_from_laplacian(src_laplacian_pyramid_list)

for i in range(x-1,-1,-1):
    path=os.path.join(args.output_image_folder,"reconstructed_"+str(i)+".jpg")
    cv2.imwrite(path,cv2.resize(reconstructed_image[i],(src_width,src_height)))
