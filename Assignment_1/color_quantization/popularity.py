import numpy as np
import cv2
import sys

color_levels=256

if(len(sys.argv)<5):
	print("Usage: python <script> <input_image_path> <output_image_path> <color_palette> <dither(True/False)>")

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

def add_validate(val,error):
	val=int(val)
	val=int(val+error)
	val=min(val,color_levels-1)
	val=max(val,0)
	return val


def performDithering(h,w,d,error,height,width,img):
	if(h+1<height):
		img[h+1,w,d]=add_validate(img[h+1,w,d],error*3.0/8)
		if(w+1<width):
			img[h+1,w+1,d]=add_validate(img[h+1,w+1,d],error*1.0/4)

	if(w+1<width):
		img[h,w+1,d]=add_validate(img[h,w+1,d],error*3.0/8)

def popularity_quantization(img_inp,keep_colors=10,dither=False):	
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

	for h in range(height):
		for w in range (width):
			dep=getRep(img[h,w,:],representative_color)
			for d in range(depth):
				error=int(img[h,w,d])-dep[d]
				img[h,w,d]=dep[d]
				if(dither==True):
					performDithering(h,w,d,error,height,width,img)

	return img

color_palette=int(sys.argv[3])
dither=bool(sys.argv[4])
img=cv2.imread(sys.argv[1],cv2.IMREAD_COLOR)
cv2.imwrite(sys.argv[2],popularity_quantization(img,color_palette,dither))
