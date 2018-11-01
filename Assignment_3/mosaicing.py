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
parser.add_argument('--dest', dest='dest_image', help="Enter dest Image Path", required=True, type=str)
parser.add_argument('--levels', dest='levels', help="Enter no. of levels for pyramid", default=3, type=int)
parser.add_argument('--output', dest='output_image_folder', help="Enter Output Image folder", required=True, type=str)
args=parser.parse_args()

os.system("mkdir -p "+args.output_image_folder)

src_img=cv2.imread(args.source_image, cv2.IMREAD_COLOR)
(src_height,src_width,src_depth)=src_img.shape

dest_img=cv2.imread(args.dest_image, cv2.IMREAD_COLOR)
(dest_height,dest_width,dest_depth)=dest_img.shape


if(src_height!=dest_height):
    min_ht=min(src_height,dest_height)
    min_ht*=1.0
    f_src=min_ht/src_height
    f_dest=min_ht/dest_height
    src_img=cv2.resize(src_img,dsize=None,fx=f_src,fy=f_src)
    (src_height,src_width,src_depth)=src_img.shape

    dest_img=cv2.resize(dest_img,dsize=None,fx=f_dest,fy=f_dest)
    (dest_height,dest_width,dest_depth)=dest_img.shape

src_gaussian_pyramid_list=pyramid.gaussian_pyramid(src_img,levels=args.levels)
src_laplacian_pyramid_list=pyramid.laplacian_pyramid(gaussian_pyramid_list=src_gaussian_pyramid_list)

dest_gaussian_pyramid_list=pyramid.gaussian_pyramid(dest_img,levels=args.levels)
dest_laplacian_pyramid_list=pyramid.laplacian_pyramid(gaussian_pyramid_list=dest_gaussian_pyramid_list)

# mask_img=np.zeros((src_height,(src_width+dest_width)//2,3),dtype='float32')
# # mask_img[:,0:src_width//2,:]=1
# mask_img[:,0:src_width//2,:]=255
# mask_gaussian_pyramid_list=pyramid.gaussian_pyramid(mask_img,levels=args.levels)
# mask_laplacian_pyramid_list=pyramid.laplacian_pyramid(gaussian_pyramid_list=mask_gaussian_pyramid_list)


combined_laplace=[]
counter=0
# for slap,dlap,mlap in zip(src_laplacian_pyramid_list,dest_laplacian_pyramid_list,mask_laplacian_pyramid_list):
for slap,dlap in zip(src_laplacian_pyramid_list,dest_laplacian_pyramid_list):
    (sheight,swidth,sdepth)=slap.shape
    (dheight,dwidth,ddepth)=dlap.shape
    # (mheight,mwidth,mdepth)=mlap.shape
    # print(sheight,swidth,sdepth,"src")
    # print(dheight,dwidth,ddepth,"dst")
    # print(mheight,mwidth,mdepth,"mask")
    part_dest=(dwidth//2)

    # comblap=np.concatenate((slap[:,0:swidth//2,:]*mlap[:,0:swidth//2,:],dlap[:,dwidth-part_dest:,:]*(255-mlap[:,mwidth-part_dest:,:])),axis=1)
    comblap=np.concatenate((slap[:,0:swidth//2,:],dlap[:,dwidth-part_dest:,:]),axis=1)
    # print(comblap.shape)
    combined_laplace.append(comblap)
    # path=os.path.join(args.output_image_folder,"mask_"+str(counter)+".jpg")
    # cv2.imwrite(path,cv2.resize(mlap,((src_width+dest_width)//2,src_height)))
    # counter+=1

counter=0
(height,width,_)=combined_laplace[0].shape
for image in combined_laplace:
    counter+=1
    path=os.path.join(args.output_image_folder,"combined_laplacian_"+str(counter)+".jpg")
    cv2.imwrite(path,cv2.resize(image,(width,height)))


reconstructed_image=pyramid.reconstruct_from_laplacian(combined_laplace)
x=len(reconstructed_image)

for i in range(x-1,-1,-1):
    path=os.path.join(args.output_image_folder,"debug_"+str(i)+".jpg")
    cv2.imwrite(path,cv2.resize(reconstructed_image[i],(width,height)))

path=os.path.join(args.output_image_folder,"seamless_mosaic.jpg")
cv2.imwrite(path,reconstructed_image[x-1])

path=os.path.join(args.output_image_folder,"seam_mosaic.jpg")
seam_comb=np.concatenate((src_img[:,0:src_width//2,:],dest_img[:,dest_width//2:dest_width,:]),axis=1)
cv2.imwrite(path,seam_comb)

