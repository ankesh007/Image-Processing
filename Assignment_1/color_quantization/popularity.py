import numpy as np
import cv2
import sys

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

def popularity_quantization(img_inp,keep_colors=10):	
	img=np.copy(img_inp)
	color_histogram=dict()
	height,width,depth=img.shape
	# print(height,width,depth)
	for h in range(height):
		for w in range (width):
			tup=(img[h,w,0],img[h,w,1],img[h,w,2])
			if tup in color_histogram:
				color_histogram[tup]+=1
			else:
				color_histogram[tup]=1

	representative_color=[]
	counter=0

	for key,value in sorted(color_histogram.iteritems(),key=lambda (k,v):(v,k),reverse=True):
		counter+=1
		representative_color.append(key)
		print(key,value)
		if(counter==keep_colors):
			break

	# print(representative_color)
	# print(type(representative_color[0]))

	for h in range(height):
		for w in range (width):
			dep=getRep(img[h,w,:],representative_color)
			for d in range(depth):
				img[h,w,d]=dep[d]

	return img

img=cv2.imread(sys.argv[1],cv2.IMREAD_COLOR)
cv2.imwrite(sys.argv[2],popularity_quantization(img,30))