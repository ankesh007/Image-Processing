import matplotlib.pyplot as plt
import numpy as np
import cv2
import copy
import argparse
from scipy.spatial import Delaunay

parser=argparse.ArgumentParser(description='Image Morphing')
parser.add_argument('--source', dest='source_image', help="Enter Source Image Path", required=True, type=str)
parser.add_argument('--dest', dest='destination_image', help="Enter Dest Image Path", required=True, type=str)
parser.add_argument('--k', dest='k', help="Enter no. of Intermediate Images", default=10, type=int)
parser.add_argument('--output', dest='output_image_path', help="Enter Output Image Path", required=True, type=str)
args=parser.parse_args()

src_img=cv2.imread(args.source_image, cv2.IMREAD_COLOR)
dest_img=cv2.imread(args.destination_image, cv2.IMREAD_COLOR)
src_height, src_width, src_depth=src_img.shape
dest_height, dest_width, dest_depth=dest_img.shape

if(src_depth!=dest_depth):
    print("Errot: Incorrect Color Channels")
    exit(0)

depth=src_depth

src_window_name="get_source_features"
dest_window_name="get_dest_features"
morphing_window="morphing"
src_points=[]
dest_points=[]

# BGR Color Channel

def src_getPoints(event, x, y, flags, param):
    global src_points,src_img

    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(x,y)
        src_points.append((y,x))
        cv2.circle(src_img,(x,y),2,(0,0,255),2)
        cv2.imshow(src_window_name, src_img)

def dest_getPoints(event, x, y, flags, param):
    global dest_points,dest_img

    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(x,y)
        dest_points.append((y,x))
        cv2.circle(dest_img,(x,y),2,(0,0,255),2)
        cv2.imshow(dest_window_name, dest_img)

def get_features():
    global src_points,src_img,dest_points,dest_img
    cv2.namedWindow(src_window_name)
    cv2.setMouseCallback(src_window_name,src_getPoints)
    cv2.namedWindow(dest_window_name)
    cv2.setMouseCallback(dest_window_name,dest_getPoints)
    
    cv2.imshow(src_window_name, src_img)
    cv2.imshow(dest_window_name, dest_img)

    while (1):
        k=cv2.waitKey(20) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

    final_size=min(len(src_img),len(dest_img))
    src_points=src_points[0:final_size]
    dest_points=dest_points[0:final_size]

    src_points.append((0,0))
    src_points.append((0,src_width))
    src_points.append((src_height,0))
    src_points.append((src_height,src_width))

    dest_points.append((0,0))
    dest_points.append((0,dest_width))
    dest_points.append((dest_height,0))
    dest_points.append((dest_height,dest_width))

def getPos(points,baricentric_coord,triangle_indices):
    accu=np.zeros([2])
    points=np.array(points)
    for i in range(3):
        accu+=baricentric_coord[i]*points[triangle_indices[i]]
    return accu

def interpolate(pos,img,h,w):
    (inter_h,inter_w)=pos
    index_h=int(inter_h)
    index_w=int(inter_w)

    if(index_h>=h-1 or index_w>=w-1):
        return img[index_h,index_w,:]

    topleft = img[index_h, index_w, :].astype(np.float)
    topright = img[index_h, index_w + 1, :].astype(np.float)
    bottomleft = img[index_h + 1, index_w, :].astype(np.float)
    bottomright = img[index_h + 1, index_w + 1, :].astype(np.float)

    interpolate_1 = topright * (inter_w - index_w) + topleft * (index_w + 1 - inter_w)
    interpolate_2 = bottomright * (inter_w - index_w) + bottomleft * (index_w + 1 - inter_w)

    final_interpolate = interpolate_2 * (inter_h - index_h) + interpolate_1 * (index_h + 1 - inter_h)
    final_interpolate = final_interpolate.astype(np.uint8)
    return final_interpolate


get_features()
src_img=cv2.imread(args.source_image, cv2.IMREAD_COLOR)
dest_img=cv2.imread(args.destination_image, cv2.IMREAD_COLOR)
# cv2.namedWindow(morphing_window)
cv2.imwrite(args.output_image_path+str(0)+".jpg",src_img)
cv2.imwrite(args.output_image_path+str(args.k+1)+".jpg",dest_img)

for i in range(args.k):
    print(i+1)
    rig_wt=(i+1)/float(args.k+1)
    lef_wt=1.0-rig_wt

    new_width=int(lef_wt*src_width+rig_wt*dest_width)
    new_height=int(lef_wt*src_height+rig_wt*dest_height)

    interim_image=np.zeros([new_height,new_width,depth])
    interim_points=[]

    total_feature_points=len(src_points)

    for ii in range(total_feature_points):
        interim_r=int(src_points[ii][0]*lef_wt+dest_points[ii][0]*rig_wt)
        interim_c=int(src_points[ii][1]*lef_wt+dest_points[ii][1]*rig_wt)
        interim_points.append((interim_r,interim_c))

    triangulation=Delaunay(interim_points)
    triangulation_indices=triangulation.simplices
    # interim_points=np.array(interim_points)
    # plt.triplot(interim_points[:,0], interim_points[:,1], triangulation.simplices.copy())
    # plt.plot(interim_points[:,0], interim_points[:,1], 'o')
    # plt.show()
    all_points=[]
    for h in range(new_height):
        for w in range(new_width):
            all_points.append((h,w))

    all_points=np.array(all_points)
    point_index=triangulation.find_simplex(all_points)
    X=triangulation.transform[point_index,:2]
    Y=all_points - triangulation.transform[point_index,2]
    baricentric_coord=np.einsum('ijk,ik->ij', X, Y)
    baricentric_coord=np.c_[baricentric_coord, 1 - baricentric_coord.sum(axis=1)]

    counter=0

    for h in range(new_height):
        for w in range(new_width):
            src_pos=getPos(src_points,baricentric_coord[counter],triangulation_indices[point_index[counter]])
            dest_pos=getPos(dest_points,baricentric_coord[counter],triangulation_indices[point_index[counter]])
            counter+=1
            pixel_val=lef_wt*interpolate(src_pos,src_img,src_height,src_width)+rig_wt*interpolate(dest_pos,dest_img,dest_height,dest_width)
            interim_image[h,w,:]=pixel_val

    # cv2.imshow(morphing_window,interim_image)
    print(args.output_image_path+str(i+1)+".jpg")
    cv2.imwrite(args.output_image_path+str(i+1)+".jpg",interim_image)
    # while (1):
    #     k=cv2.waitKey(20) & 0xFF
    #     if k == 27:
    #         break

cv2.destroyAllWindows()


# cv2.imwrite(args.output_image_path, final_img)
