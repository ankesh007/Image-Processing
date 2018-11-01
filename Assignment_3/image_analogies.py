import matplotlib.pyplot as plt
import numpy as np
import cv2
import copy
import argparse
from scipy.spatial import Delaunay
import os
import scipy.ndimage
import math
import pyramid
from sklearn.feature_extraction.image import extract_patches_2d

n_fine=5
n_coarse=3

def return_feature(img_sm,neighbour):
    padded_img=np.pad(img_sm, neighbour//2, mode='symmetric')
    patches=extract_patches_2d(padded_img,(neighbour,neighbour))    
    unrolled_patches = patches.reshape(
        patches.shape[0], -1)
    
    return unrolled_patches

def get_feature_vector(gA,gAp,level):
    """
    Handle last level
    """
    feat_A_l=return_feature(gA[level],n_fine)
    feat_A_lprev=return_feature(pyramid.pyramid_up(gA[level+1],gA[level].shape),n_coarse)
    feat_Ap_l=return_feature(gAp[level],n_fine)
    feat_Ap_lprev = return_feature(pyramid.pyramid_up(
        gAp[level+1], gAp[level].shape), n_coarse)

    return np.concatenate((feat_A_l, feat_A_lprev, feat_Ap_l, feat_Ap_lprev), axis=1)

def get_feature_matrix(gA,gA_,level=5):
    indices=[]
    height,width=gA[level].shape
    feature_matrix=[]

    for h in height:
        for w in width:
            indices.append((h,w))

    feature_matrix=get_feature_vector(gA,gA_,level)
    return feature_matrix,indices

def best_match():

def create_image_analogy(img_a,img_af,levels=5):
    gaussian_a=pyramid.gaussian_pyramid(img_a,levels=levels)
    gaussian_af = pyramid.gaussian_pyramid(img_af,levels=levels)
    gaussian_b = pyramid.gaussian_pyramid(img_b,levels=levels)

    # features_a,features_af,features_b=get_features(img_a,img_af,img_b)
    gaussian_bf=[]

    for image in gaussian_b:
        gaussian_bf.append(np.zeros_like(image))
    
    nw=3


    for level in range(levels,-1,-1):
        height,width,_=gaussian_b[level].shape
        feature_matrix,indices=get_feature_matrix(gaussian_a,gaussian_af,level)

        for h in height:
            for w in width:
                pixel=best_match(gaussian_b,gaussian_bf,level,(h,w),feature_matrix,indices,nw)
                gaussian_bf[level][h,w]=gaussian_af[level][pixel[0]][pixel[1]]
    





def main():
    parser=argparse.ArgumentParser(description='Image Pyramid')
    parser.add_argument('--source', dest='source_image', help="Enter Source Image Path", required=True, type=str)
    parser.add_argument('--source_filtered', dest='filtered_source', help="Enter filtered Source Image Path", required=True, type=str)
    parser.add_argument('--dest', dest='dest_image', help="Enter dest Image Path", required=True, type=str)
    parser.add_argument('--levels', dest='levels', help="Enter no. of levels for pyramid", default=5, type=int)
    parser.add_argument('--output', dest='output_image_folder', help="Enter Output Image folder", required=True, type=str)
    args=parser.parse_args()

    os.system("mkdir -p "+args.output_image_folder)

    A =cv2.imread(args.source_image, cv2.IMREAD_COLOR)
    A_f = cv2.imread(args.filtered_source, cv2.IMREAD_COLOR)
    B = cv2.imread(args.dest, cv2.IMREAD_COLOR)

    create_image_analogy(A,A_f,B,args.levels)


if __name__=='__main__':
    main()
