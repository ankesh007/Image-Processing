import numpy as np
import cv2
import sys

if(len(sys.argv)<3):
	print("Usage: python <script> <input_image_path> <output_image_path>")

img=cv2.imread(sys.argv[1],0)
cv2.imwrite(sys.argv[2],dst)