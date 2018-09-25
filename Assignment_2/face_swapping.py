import matplotlib.pyplot as plt
import numpy as np
import cv2
import copy
import argparse
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(description='Face Swapping')
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

    # src_points.append((0,0))
    # src_points.append((0,src_width))
    # src_points.append((src_height,0))
    # src_points.append((src_height,src_width))

    # dest_points.append((0,0))
    # dest_points.append((0,dest_width))
    # dest_points.append((dest_height,0))
    # dest_points.append((dest_height,dest_width))

def getPos(points,baricentric_coord,triangle_indices):
    accu=np.zeros([2])
    for i in range(3):
        accu+=baricentric_coord[i]*points[triangle_indices[i]]
    return accu

def interpolate(pos,img,h,w):
    (inter_h,inter_w)=pos
    index_h=int(inter_h)
    index_w=int(inter_w)

    if(index_h>=h-1 or index_w>=w-1):
        return img[min(index_h,h-1),min(index_w,w-1),:]
    if(index_h<0 or index_w<0):
        return img[max(index_h,0),max(index_w,0),:]

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

cv2.imwrite(args.output_image_path+str(0)+".jpg",src_img)
cv2.imwrite(args.output_image_path+str(args.k+1)+".jpg",dest_img)

src_points_np=np.array(src_points).astype(np.float32)
dest_points_np=np.array(dest_points).astype(np.float32)
# src_points_np=np.float32([[ 74.,240.]
#  ,[ 76.,297.]
#  ,[ 90.,330.]
#  ,[160.,350.]
#  ,[215.,370.]
#  ,[293.,378.]
#  ,[352.,368.]
#  ,[409.,295.]
#  ,[411.,245.]
#  ,[372.,184.]
#  ,[331.,150.]
#  ,[269.,138.]
#  ,[190.,140.]
#  ,[119.,147.]
#  ,[ 88.,176.]
#  ,[271.,251.]
#  ,[215.,196.]
#  ,[207.,302.]
#  ,[313.,307.]
#  ,[322.,208.]])

# dest_points_np=np.float32([[ 79.,157.]
#  ,[ 76.,204.]
#  ,[ 96.,234.]
#  ,[150.,249.]
#  ,[195.,253.]
#  ,[248.,249.]
#  ,[298.,234.]
#  ,[348.,183.]
#  ,[349.,142.]
#  ,[306., 91.]
#  ,[263., 73.]
#  ,[214., 64.]
#  ,[154., 68.]
#  ,[ 96., 76.]
#  ,[ 80.,105.]
#  ,[234.,161.]
#  ,[183.,114.]
#  ,[183.,206.]
#  ,[283.,198.]
#  ,[283.,124.]])
# [[ 71. 245.]
#  [ 85. 326.]
#  [139. 345.]
#  [230. 370.]
#  [320. 377.]
#  [404. 318.]
#  [408. 261.]
#  [367. 176.]
#  [324. 147.]
#  [268. 138.]
#  [184. 141.]
#  [104. 155.]
#  [275. 253.]
#  [322. 208.]
#  [312. 308.]
#  [217. 196.]
#  [209. 301.]]
# [[ 78. 159.]
#  [ 85. 226.]
#  [133. 244.]
#  [193. 254.]
#  [261. 247.]
#  [327. 211.]
#  [347. 159.]
#  [315.  99.]
#  [271.  75.]
#  [221.  65.]
#  [158.  69.]
#  [ 93.  81.]
#  [241. 162.]
#  [283. 123.]
#  [278. 200.]
#  [184. 114.]
#  [182. 205.]]

print(src_points_np)
print(dest_points_np)

def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst

# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = [] 
    t2Rect = []
    t2RectInt = []

    for i in range(3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    #img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)
    
    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    
    img2Rect = img2Rect * mask
    # print(( (1.0, 1.0, 1.0) - mask ))
    # print(( (1.0, 1.0, 1.0) - mask ).shape)
    # print(r2)
    # print(img2[r2[1]:(r2[1]+r2[3]), r2[0]:(r2[0]+r2[2])].shape)
    # print(img2.shape)

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
     
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect 


def face_swap(src_img,dest_img,src_points_np,dest_points_np,output_image_path,k=10):

    src_img=np.copy(src_img)
    dest_img=np.copy(dest_img)

    (dest_height,dest_width,_)=dest_img.shape
    (src_height,src_width,_)=src_img.shape
    # transformation, mask = cv2.findHomography(src_points_np, dest_points_np, cv2.RANSAC,5.0)
    # src_img2=cv2.warpPerspective(src_img,transformation,(src_width,src_height))
    # src_img=cv2.warpPerspective(src_img,transformation,(dest_width,dest_height))
    # src_points_np=cv2.perspectiveTransform(src_points_np.reshape(-1,1,2),transformation)
    # src_points_np=src_points_np.reshape(-1,2)
    cv2.imwrite(output_image_path+"Affine"+".jpg",src_img)
    # cv2.imwrite(output_image_path+"Affine2"+".jpg",src_img2)
    (src_height,src_width,_)=src_img.shape

    triangulation=Delaunay(dest_points_np)
    triangulation_indices=triangulation.simplices

    all_points=[]
    for h in range(dest_height):
        for w in range(dest_width):
            all_points.append((h,w))
    
    all_points=np.array(all_points)
    point_index=triangulation.find_simplex(all_points)
    all_points2=[]
    all_points_len=len(all_points)
    for i in range(all_points_len):
        if(point_index[i]==-1):
            continue
        all_points2.append(all_points[i])

    all_points=all_points2
    all_points=np.array(all_points)

    point_index=point_index[point_index[:]!=-1]


    X=triangulation.transform[point_index,:2]
    Y=all_points - triangulation.transform[point_index,2]
    baricentric_coord=np.einsum('ijk,ik->ij', X, Y)
    baricentric_coord=np.c_[baricentric_coord, 1 - baricentric_coord.sum(axis=1)]
    interim_image=np.copy(dest_img)
    img1Warped = np.copy(dest_img)

    dest_points_np_xy=dest_points_np[:,[1, 0]]
    src_points_np_xy=src_points_np[:,[1, 0]]
    hullIndex = cv2.convexHull(np.array(dest_points_np_xy), returnPoints = False)
    hull_dest=[]

    for ind in hullIndex:
        hull_dest.append(dest_points_np_xy[ind].tolist()[0])

    for i in range(len(triangulation_indices)):
        t1 = []
        t2 = []
        
        #get points for img1, img2 corresponding to the triangles
        for j in range(3):
            t1.append(src_points_np_xy[triangulation_indices[i][j]])
            t2.append(dest_points_np_xy[triangulation_indices[i][j]])
        
        warpTriangle(src_img, img1Warped, t1, t2)

    src_img=img1Warped
    cv2.imwrite(output_image_path+"_warped.jpg",src_img)

    dest_mask=np.zeros(dest_img.shape,dtype=dest_img.dtype)
    cv2.fillConvexPoly(dest_mask,np.int32(hull_dest),(255,255,255))
    bb = cv2.boundingRect(np.float32([hull_dest]))
    center=((bb[0]+int(bb[2]/2), bb[1]+int(bb[3]/2))) 
    cv2.imwrite(output_image_path+"_mask.jpg",dest_mask)

    for i in range(k):
        print(i)
        counter=0
        rig_wt=(i+1)/float(args.k+1)
        lef_wt=1.0-rig_wt

        for points in all_points:
            src_pos=getPos(dest_points_np,baricentric_coord[counter],triangulation_indices[point_index[counter]])
            dest_pos=getPos(dest_points_np,baricentric_coord[counter],triangulation_indices[point_index[counter]])
            counter+=1
            pixel_val=lef_wt*interpolate(src_pos,src_img,src_height,src_width)+rig_wt*interpolate(dest_pos,dest_img,dest_height,dest_width)
            interim_image[points[0],points[1],:]=pixel_val

        print(output_image_path+str(i+1)+".jpg")

        output = cv2.seamlessClone(interim_image, dest_img, dest_mask, center, cv2.NORMAL_CLONE)
        cv2.imwrite(output_image_path+str(i+1)+".jpg",output)


face_swap(src_img,dest_img,src_points_np,dest_points_np,args.output_image_path,args.k)
