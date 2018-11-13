import matplotlib.pyplot as plt
import numpy as np
import cv2
import copy
import argparse
import os
import scipy.ndimage
import math
import pyramid
import preprocess
from sklearn.feature_extraction.image import extract_patches_2d
import time
from panns import *
"""
https://github.com/ryanrhymes/panns
"""
#keep them odd
n_fine=5
n_coarse=3
n_channel=3
# half is used only with fine scale
half=(n_fine**2)//2
kappa=2
size=64
inf=10**20

approx_time=0
coh_time=0

def get_feature_matrix_(img_sm,neighbour,keep_half=False,last=False):
    height,width,_=img_sm.shape
    if last:
        return np.zeros((height*width,neighbour*neighbour*n_channel))
    padded_img = np.pad(img_sm, ((neighbour//2, neighbour//2),
                                 (neighbour//2, neighbour//2), (0,0)), mode='constant')
    patches=extract_patches_2d(padded_img,(neighbour,neighbour))    
    unrolled_patches = patches.reshape(
        patches.shape[0], -1)
    if keep_half:
        unrolled_patches=unrolled_patches[:,0:n_channel*half]
    return unrolled_patches


def get_feature_matrix(g_A,g_Ap,level=5):
    """
    Returns feature matrix given level of pyramid
    For last level 2 options -> 1. append 0's
    2. append same feature vector as l  
    """
    indices=[]
    height,width,_=g_A[level].shape
    feature_matrix=[]
    rev_map={}

    for h in range(height):
        for w in range(width):
            rev_map[(h,w)]=len(indices)
            indices.append((h,w))

    x=len(g_A)
    flag=True if level==(len(g_A)-1) else False
    upp=min(x-1,level+1)

    feat_A_l=get_feature_matrix_(g_A[level],n_fine)
    # feat_A_lprev=get_feature_matrix_(pyramid.pyramid_up(g_A[upp],g_A[level].shape),n_coarse,last=flag)
    feat_A_lprev=get_feature_matrix_(g_A[upp],n_coarse,last=flag)
    feat_Ap_l=get_feature_matrix_(g_Ap[level],n_fine,keep_half=True)
    # feat_Ap_lprev = get_features_matrix_(pyramid.pyramid_up(g_Ap[upp], g_Ap[level].shape), n_coarse,last=flag)
    feat_Ap_lprev = get_feature_matrix_(g_Ap[upp], n_coarse,last=flag)
    # print(feat_A_l[0])
    # print(feat_A_lprev[0])
    # print(feat_Ap_l[0])
    # print(feat_Ap_lprev[0])
    # print("******************")
    col_size=feat_A_l.shape[1]+feat_A_lprev.shape[1]+feat_Ap_l.shape[1]+feat_Ap_lprev.shape[1]
    feature_matrix=np.zeros((height*width,col_size))

    counter=0
    hh,ww,_=g_A[upp].shape
    for h in range(height):
        for w in range(width):
            new_counter=(h//2)*ww+w//2
            feature_matrix[counter] = np.concatenate(
                (feat_A_l[counter], feat_A_lprev[new_counter], feat_Ap_l[counter], feat_Ap_lprev[new_counter]))
            counter+=1

    # print(feature_matrix.shape)
    return feature_matrix,indices,rev_map


def get_feature_vector_(img,row,col,neighbour,keep_half=False,last=False):
    if last:
        return np.zeros((neighbour*neighbour*n_channel))
    img = np.pad(img, ((neighbour//2, neighbour//2),(neighbour//2, neighbour//2), (0, 0)), mode='constant')
    px_feature = img[row: row + neighbour,
                                      col: col + neighbour].flatten()

    if keep_half:
        px_feature = px_feature[:n_channel * half]

    return px_feature


def get_feature_vector(g_B,g_Bp,row,col,level):
    flag=True if level+1==len(g_B) else False
    x=len(g_B)
    upp=min(x-1,level+1)

    feat_B_l=get_feature_vector_(g_B[level],row,col,n_fine)
    feat_B_lprev = get_feature_vector_(g_B[upp], row//2, col//2, n_coarse,last=flag)
    feat_Bp_l = get_feature_vector_(g_Bp[level], row, col, n_fine,keep_half=True)
    feat_Bp_lprev = get_feature_vector_(g_Bp[upp], row//2, col//2, n_coarse, last=flag)
    # print("****")
    # print(g_B[upp].shape,row,col)
    # print(feat_B_l)
    # print(feat_B_lprev)
    # print(feat_Bp_l,"yehah")
    # print(feat_Bp_lprev)
    # print(np.concatenate((feat_B_l,feat_B_lprev,feat_Bp_l,feat_Bp_lprev)).shape)

    return np.concatenate((feat_B_l,feat_B_lprev,feat_Bp_l,feat_Bp_lprev))    
    
def best_approx_match(feature_matrix,pixel_feature,indices):
    """
    return co-ordinates,dis of
    best match in feature_matrix
    """
    x=len(indices)
    mini=inf
    index=0

    for i in range(x):
        norm=np.linalg.norm(pixel_feature-feature_matrix[i])
        norm=norm**2
        if(norm<mini):
            mini=norm
            index=i

    return (indices[index],mini)


def fast_approx_match(index_structure, pixel_feature, indices):
    """
    return co-ordinates,dis of
    best match in feature_matrix
    """
    best_match=index_structure.query(pixel_feature,1)
    # print(best_match)
    return (indices[best_match[0][0]], best_match[0][1]**2)

def best_coherence_match(h,w,mapper,feature_matrix,rev_map,pixel_feature):
    """
    return co-ordinates,dis of
    best coherence match
    """
    mini = inf
    index = (h,w)

    if(h-1>=0):
        val = mapper[(h-1, w)]
        nei=(val[0]+1,val[1])
        if(nei in rev_map):
            norm = np.linalg.norm(pixel_feature-feature_matrix[rev_map[nei]])
            norm = norm**2

            if(norm < mini):
                mini = norm
                index = nei

    if(w-1>=0):
        val = mapper[(h, w-1)]
        nei=(val[0],val[1]+1)
        if(nei in rev_map):
            norm = np.linalg.norm(pixel_feature-feature_matrix[rev_map[nei]])
            norm = norm**2

            if(norm < mini):
                mini = norm
                index = nei

    return (index, mini)



def best_match(g_B,g_Bp,level,h,w,feature_matrix,indices,mapper,rev_map,index_structure):
    pixel_feature=get_feature_vector(g_B,g_Bp,h,w,level)

    global approx_time,coh_time
    t1=time.time()
    # pixel_appr,norm_appr=best_approx_match(feature_matrix,pixel_feature,indices)
    pixel_appr, norm_appr = fast_approx_match(index_structure, pixel_feature, indices)
    t2 = time.time()
    approx_time+=t2-t1
    pixel_cohr,norm_cohr=best_coherence_match(h,w,mapper,feature_matrix,rev_map,pixel_feature)
    t1=time.time()
    coh_time+=t1-t2

    tot_level=len(g_B)

    if norm_cohr<=norm_appr*(1+(2**(level-tot_level))*kappa):
        # print("Coherence")
        return pixel_cohr
    
    # print("Approx")
    return pixel_appr

def resize(img):
    h,w,_=img.shape
    fact = size/h
    return cv2.resize(img,dsize=None,fx=fact,fy=fact)

def create_image_analogy(img_A,img_Ap,img_B,output_image_folder,levels=5):
    # global coh_time,approx_time
    gaussian_A=pyramid.gaussian_pyramid(img_A,levels=levels)
    gaussian_Ap = pyramid.gaussian_pyramid(img_Ap,levels=levels)
    gaussian_B = pyramid.gaussian_pyramid(img_B,levels=levels)

    levels=len(gaussian_A)

    gaussian_Bp=[]
    for image in gaussian_B:
        gaussian_Bp.append(np.zeros_like(image))
    
    for level in range(levels-1,-1,-1):
        print("Processing level:",level)
        height,width,_=gaussian_B[level].shape
        feature_matrix,indices,rev_map=get_feature_matrix(gaussian_A,gaussian_Ap,level)
        print(height,width)
        print("feature_matrix shape:",feature_matrix.shape)

        """
        Constructing index structure
        """  
        print("Constructing index structure")      
        dimension=(n_channel)*(2*n_coarse*n_coarse+n_fine*n_fine+half)
        print(dimension)
        index_structure = PannsIndex(dimension=dimension, metric='euclidean')
        index_structure.parallelize(True)
        index_structure.load_matrix(feature_matrix)
        index_structure.build(2)
        print("Finished Construction")

        mapper={}

        for h in range(height):
            for w in range(width):
                pixel=best_match(gaussian_B,gaussian_Bp,level,h,w,feature_matrix,indices,mapper,rev_map,index_structure)
                gaussian_Bp[level][h,w,:]=gaussian_Ap[level][pixel[0],pixel[1],:]
                mapper[(h,w)]=pixel
            print(h)
            # print(coh_time)
            # print(approx_time)
            # approx_time=0
            # coh_time=0

        # path = os.path.join(output_image_folder, "analogy_"+str(level)+".jpg")
        # cv2.imwrite(path,gaussian_Bp[level])

    return gaussian_Bp[0]



def image_analogy(img_A,img_Ap,img_B,output_image_folder,levels=5,mode="RGB"):
    img_A=resize(img_A)
    img_Ap=resize(img_Ap)
    img_B=resize(img_B)
    img_Bp=np.copy(img_B)
    # print(type(img_A[0,0,0]))

    if(mode=="RGB"):
        img_Bp=create_image_analogy(img_A,img_Ap,img_B,output_image_folder,levels)

    else:
        img_A/=255
        img_Ap/=255
        img_B/=255
        img_Bp/=255
        # print(img_A)
        img_A_YIQ=preprocess.convert_to_YIQ(img_A)
        img_Ap_YIQ=preprocess.convert_to_YIQ(img_Ap)
        img_B_YIQ=preprocess.convert_to_YIQ(img_B)
        img_Bp_YIQ=np.copy(img_B_YIQ)


        img_A_YIQ[:,:,0],img_Ap_YIQ[:,:,0]=preprocess.remap_luminance(img_A_YIQ[:,:,0],img_Ap_YIQ[:,:,0],img_B_YIQ[:,:,0])
        global n_channel
        n_channel=1
        img_Bp_YIQ[:,:,0:1]=create_image_analogy(img_A_YIQ[:,:,0:1],img_Ap_YIQ[:,:,0:1],img_B_YIQ[:,:,0:1],output_image_folder,levels)
        img_A=preprocess.convert_to_RGB(img_A_YIQ)*255
        img_Ap=preprocess.convert_to_RGB(img_Ap_YIQ)*255
        img_B=preprocess.convert_to_RGB(img_B_YIQ)*255
        img_Bp=preprocess.convert_to_RGB(img_Bp_YIQ)*255

        # print(img_A_YIQ.shape)
        # img_A_YIQ=preprocess.convert_to_RGB(img_A_YIQ)
        # img_A_YIQ*=255
    
    path = os.path.join(output_image_folder, "A"+".jpg")
    cv2.imwrite(path, img_A)
    path = os.path.join(output_image_folder, "Ap"+".jpg")
    cv2.imwrite(path, img_Ap)
    path = os.path.join(output_image_folder, "B"+".jpg")
    cv2.imwrite(path, img_B)
    path = os.path.join(output_image_folder, "Bp"+".jpg")
    cv2.imwrite(path, img_Bp)


def main():
    parser=argparse.ArgumentParser(description='Image Pyramid')
    parser.add_argument('--source', dest='source_image', help="Enter Source Image Path", required=True, type=str)
    parser.add_argument('--source_filtered', dest='filtered_source', help="Enter filtered Source Image Path", required=True, type=str)
    parser.add_argument('--dest', dest='dest_image', help="Enter dest Image Path", required=True, type=str)
    parser.add_argument('--levels', dest='levels', help="Enter no. of levels for pyramid", default=5, type=int)
    parser.add_argument('--output', dest='output_image_folder', help="Enter Output Image folder", required=True, type=str)
    parser.add_argument('--resize_height', dest='resize', help="Enter resize parameter", default=64, type=float)
    parser.add_argument('--mode', dest='mode', help="RGB or YIQ", default="RGB", type=str)
    parser.add_argument('--kappa', dest='kappa',
                        help="Enter Kappa", default=5.0, type=float)
    args=parser.parse_args()

    os.system("mkdir -p "+args.output_image_folder)

    A =cv2.imread(args.source_image, cv2.IMREAD_COLOR).astype(np.float)
    Ap = cv2.imread(args.filtered_source, cv2.IMREAD_COLOR).astype(np.float)
    B = cv2.imread(args.dest_image, cv2.IMREAD_COLOR).astype(np.float)
    global kappa,size
    kappa=args.kappa
    size=args.resize
    image_analogy(A,Ap,B,args.output_image_folder,args.levels,args.mode)


if __name__=='__main__':
    main()
