import numpy as np
import cv2
import sys

color_levels=256

if(len(sys.argv)<3):
	print("Usage: python <script> <input_image_path> <output_image_path>")

def getRep(pixel_color,rep_color):
	# print(pixel_color.shape)
	mini=1e18
	index=()

	for tuples in rep_color:
		dist=0
		for j in range(3):
			dist+=(int(pixel_color[j])-int(tuples[j]))**2
		if(dist<mini):
			mini=dist
			index=tuples
	return index

def median_cut_quantization(img_inp,keep_colors=10):	
	img=np.copy(img_inp)
	height,width,depth=img.shape

	x_cut=0
	y_cut=0
	z_cut=0

	mini=[1e18,1e18,1e18]
	maxi=[-1e18,-1e18,-1e18]

	for h in range(height):
		for w in range (width):
			for d in range(depth):
				mini[i]=min(mini[i],img[h,w,d])
				maxi[i]=max(maxi[i],img[h,w,d])

	accu=[[[0 for x in range(color_levels+1)] for y in range(color_levels+1)]for z in range(color_levels+1)]
	dp=[[[0 for x in range(color_levels+1)] for y in range(color_levels+1)]for z in range(color_levels+1)]

	for h in range(height):
		for w in range (width):
			pixel_color=img[h,w,:]
				accu[pixel_color[0]+1][pixel_color[1]+1][pixel_color[2]+1]+=1



	while(True):
		boxes=(1+x_cut)*(1+y_cut)*(1+z_cut)
		if(boxes>=keep_colors):
			break



	# for h in range(height):
	# 	for w in range (width):
	# 		dep=getRep(img[h,w,:],representative_color)
	# 		for d in range(depth):
	# 			img[h,w,d]=dep[d]

	return img

img=cv2.imread(sys.argv[1],cv2.IMREAD_COLOR)
cv2.imwrite(sys.argv[2],popularity_quantization(img,30))