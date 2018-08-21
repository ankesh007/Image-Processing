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

def getSumRectangle(dp,i1,j1,k1,i2,j2,k2):
	return dp[i2][j2][k2]-(dp[i1-1][j2][k2]+dp[i2][j1-1][k2]+dp[i2][j2][k1-1])+(dp[i1-1][j1-1][k2]+dp[i2][j1-1][k1-1]+dp[i1-1][j2][k1-1])-dp[i1-1][j1-1][k1-1]

def getMedian(dp,i1,j1,k1,i2,j2,k2,axis):
	total_points=getSumRectangle(dp,i1,j1,k1,i2,j2,k2)
	lef_end=[i1,j1,k1]
	rig_end=[i2,j2,k2]
	temp_rig=[i2,j2,k2]

	lef=lef_end[axis],rig=rig_end[axis]
	ans=lef
	diff=1e18
	while(lef<rig):
		mid=(lef+rig)//2
		temp_rig[axis]=mid
		lef_part=getSumRectangle(dp,lef_end[0],lef_end[1],lef_end[2],temp_rig[0],temp_rig[1],temp_rig[2])
		rig_part=total_points-lef_part

		if(abs(lef_part-rig_part)<diff):
			diff=abs(lef_part-rig_part)
			ans=mid

		if(lef_part>rig_part):
			rig=mid-1
		else:
			lef=mid+1

	return ans

def median_cut_quantization(img_inp,keep_colors=10):	
	img=np.copy(img_inp)
	height,width,depth=img.shape

	# cut=[0 for i in range(3)]

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

	for i in range(1,color_levels+1,1):
		for j in range(1,color_levels+1,1):
			for k in range(1,color_levels+1,1):
				dp[i][j][k]=accu[i][j][k]+dp[i-1][j][k]+dp[i][j-1][k]+dp[i][j][k-1]+dp[i-1][j-1][k-1]-(dp[i-1][j-1][k]+dp[i][j-1][k-1]+dp[i-1][j][k-1])

	box_coord=[[(mini[i],maxi[i]) for i in range(3)]]
	boxes=1
	while(boxes<keep_colors):
		# total_boxes=(1+cut[0])*(1+cut[1])*(1+cut[2])
		# if(total_boxes>=keep_colors):
		# 	break
		boxes+=1
		index=-1
		axis=-1
		maxy=0

		for j in range(total_boxes):
			boxes=box_coord[j]
			for i in range(3):
				diff=boxes[i][1]-boxes[i][0]
				if(diff>maxy):
					maxy=diff
					axis=i
					index=j

		cut[axis]+=1

		if(maxy==1):
			print("Maximum Variation along any axis=1 pixel, hence stopping median cut")
			break

		one_box=list(box_coord[index])

		cut_val=getMedian(dp,one_box[0][0],one_box[1][0],one_box[2][0],one_box[0][1],one_box[1][1],one_box[2][1],axis)
		box_coord[index][axis][1]=cut_val
		print("boxes:",boxes,"axis:",axis,"maxy:",maxy,"cutval:",cut_val)

		new_box=list(one_box)
		box_coord.append(new_box)
		x=len(box_coord)
		box_coord[x-1][axis][0]=cut_val+1

	len_boxes=len(box_coord)
	representative_color=[[0 for in range(3)] for j in range(len_boxes)]
	count=[0 for i in range(len_boxes)]

	for h in range(height):
		for w in range (width):
			for z in range(len_boxes):
				flag=True
				for d in range(depth):
					if(img[h,w,d]>=box_coord[z][d][0] and img[h,w,d]<=box_coord[z][d][1]):
						continue
					else:
						flag=False
						break
				if(flag==True):
					count[z]+=1
					for d in range(depth):
						representative_color[z][d]+=int(img[h,w,d])

	for i in range(len_boxes):
		for d in range(depth):
			representative_color[i][d]=representative_color[i][d]//count[i]

	for h in range(height):
		for w in range (width):
			dep=getRep(img[h,w,:],representative_color)
			for d in range(depth):
				img[h,w,d]=dep[d]

	return img

img=cv2.imread(sys.argv[1],cv2.IMREAD_COLOR)
cv2.imwrite(sys.argv[2],median_cut_quantization(img,10))