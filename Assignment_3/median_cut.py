import numpy as np
import cv2
import sys
import copy

color_levels=256



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
	lef_end=[int(i1),int(j1),int(k1)]
	rig_end=[int(i2),int(j2),int(k2)]
	temp_rig=[int(i2),int(j2),int(k2)]

	lef=lef_end[axis]
	rig=rig_end[axis]
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

def median_cut_quantization(img_inp,keep_colors=10,dither=False):	
	img=np.copy(img_inp)
	height,width,depth=img.shape

	# cut=[0 for i in range(3)]

	mini=[1e18,1e18,1e18]
	maxi=[-1e18,-1e18,-1e18]

	for h in range(height):
		for w in range (width):
			for d in range(depth):
				mini[d]=min(mini[d],img[h,w,d])
				maxi[d]=max(maxi[d],img[h,w,d])

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

	box_coord=[[[mini[i],maxi[i]] for i in range(3)]]
	boxes=1
	while(boxes<keep_colors):
		# total_boxes=(1+cut[0])*(1+cut[1])*(1+cut[2])
		# if(total_boxes>=keep_colors):
		# 	break
		index=-1
		axis=-1
		maxy=0

		for j in range(boxes):
			box=box_coord[j]
			for i in range(3):
				diff=box[i][1]-box[i][0]
				if(diff>maxy):
					maxy=diff
					axis=i
					index=j

		if(maxy<=1):
			print("Maximum Variation along any axis<=1 pixel, hence stopping median cut")
			break

		one_box=copy.deepcopy(box_coord[index])
		# print(box_coord)
		cut_val=getMedian(dp,one_box[0][0],one_box[1][0],one_box[2][0],one_box[0][1],one_box[1][1],one_box[2][1],axis)
		box_coord[index][axis][1]=cut_val
		print("boxes:",boxes,"axis:",axis,"maxy:",maxy,"cutval:",cut_val,"index:",index)
		# for boxe in box_coord:
		# 	print(boxe)

		new_box=list(one_box)
		box_coord.append(new_box)
		x=len(box_coord)
		box_coord[x-1][axis][0]=cut_val+1
		boxes+=1
		# print(box_coord)

	TOTAL_BOXES=len(box_coord)
	representative_color=[[0 for i in range(3)] for j in range(TOTAL_BOXES)]
	count=[0 for i in range(TOTAL_BOXES)]

	for h in range(height):
		for w in range (width):
			for z in range(TOTAL_BOXES):
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
					break

	for i in range(TOTAL_BOXES):
		for d in range(depth):
			if(count[i]!=0):
				representative_color[i][d]=representative_color[i][d]//count[i]
			else:
				representative_color[i][d]=(int(box_coord[i][d][0])+box_coord[i][d][1])//2
			representative_color[i][d]=max(0,representative_color[i][d]-1)

	for h in range(height):
		for w in range (width):
			dep=getRep(img[h,w,:],representative_color)
			for d in range(depth):
				error=int(img[h,w,d])-dep[d]
				img[h,w,d]=dep[d]
				if(dither==True):
					performDithering(h,w,d,error,height,width,img)

	return img

if __name__=="__main__":
	if(len(sys.argv)<5):
		print("Usage: python <script> <input_image_path> <output_image_path> <color_palette> <dither(True/False)>")
		exit()
	color_palette=int(sys.argv[3])
	dither=False
	if sys.argv[4]=='True':
		dither=True
	img=cv2.imread(sys.argv[1],cv2.IMREAD_COLOR)
	cv2.imwrite(sys.argv[2],median_cut_quantization(img,color_palette,dither))