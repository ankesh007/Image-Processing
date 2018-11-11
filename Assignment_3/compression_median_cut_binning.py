import numpy as np
import cv2
import copy
import argparse
from scipy.spatial import Delaunay
import os
import scipy.ndimage
import math
import pyramid
import median_cut

parser=argparse.ArgumentParser(description='Image Mosaicing')
parser.add_argument('--source', dest='source_image', help="Enter Source Image Path", required=True, type=str)
parser.add_argument('--levels', dest='levels', help="Enter no. of levels for pyramid", default=5, type=int)
parser.add_argument('--to_keep', dest='to_keep', help="No. of bins/colors", default=10, type=int)
parser.add_argument('--mode', dest='mode', help="<median_cut,bin>", default="median_cut", type=str)
parser.add_argument('--output', dest='output_image_folder', help="Enter Output Image folder", required=True, type=str)
args=parser.parse_args()

os.system("mkdir -p "+args.output_image_folder)

src_img=cv2.imread(args.source_image, cv2.IMREAD_COLOR)
(src_height,src_width,src_depth)=src_img.shape

src_gaussian_pyramid_list=pyramid.gaussian_pyramid(src_img,levels=args.levels)
src_laplacian_pyramid_list=pyramid.laplacian_pyramid(gaussian_pyramid_list=src_gaussian_pyramid_list)

x=len(src_laplacian_pyramid_list)

def do_binning(img,bins):
	assert(len(img.shape)==2)
	height,width=img.shape
	new_img=np.copy(img)
	color_list=[]
	for h in range(height):
		for w in range(width):
			color_list.append((img[h,w],h,w))

	color_list=sorted(color_list)
	# print(color_list)

	bins=min(bins,len(color_list))
	per_bin=(len(color_list)+bins-1)//bins

	for i in range(bins):
		upp=min(len(color_list),(i+1)*per_bin)
		counter=0
		value=0

		for j in range(i*per_bin,upp):
			counter+=1
			value+=int(color_list[j][0])
			# new_img[color_list[j][1],color_list[j][2]]=color_list[j][0]
		if(counter==0):
			continue
		value//=counter

		for j in range(i*per_bin,upp):
			new_img[color_list[j][1],color_list[j][2]]=value

	return new_img


def binning(img,bins):
	channels=img.shape[2]
	new_img=np.copy(img)
	for channel in range(channels):
		new_img[:,:,channel]=do_binning(img[:,:,channel],bins)
	return new_img

compressed_laplacian_pyramid_list=[]

for i in range(x):
	print(i)
	if(args.mode=="median_cut"):
		compressed_laplacian_pyramid_list.append(median_cut.median_cut_quantization(src_laplacian_pyramid_list[i],args.to_keep))
	else:
		compressed_laplacian_pyramid_list.append(binning(src_laplacian_pyramid_list[i],args.to_keep))

# for i in range(x-1,-1,-1):
#     path=os.path.join(args.output_image_folder,"lap_compressed_"+str(i)+".jpg")
#     cv2.imwrite(path,cv2.resize(compressed_laplacian_pyramid_list[i],(src_width,src_height)))
#     path=os.path.join(args.output_image_folder,"lap_original_"+str(i)+".jpg")
#     cv2.imwrite(path,cv2.resize(src_laplacian_pyramid_list[i],(src_width,src_height)))


compressed_image=pyramid.reconstruct_from_laplacian(compressed_laplacian_pyramid_list)
original_image=pyramid.reconstruct_from_laplacian(src_laplacian_pyramid_list)

for i in range(x-1,-1,-1):
    path=os.path.join(args.output_image_folder,"compressed_"+str(i)+".jpg")
    cv2.imwrite(path,cv2.resize(compressed_image[i],(src_width,src_height)))
    path=os.path.join(args.output_image_folder,"original_"+str(i)+".jpg")
    cv2.imwrite(path,cv2.resize(original_image[i],(src_width,src_height)))
