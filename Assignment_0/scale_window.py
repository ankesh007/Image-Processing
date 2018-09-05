import numpy as np
import cv2
import sys
import copy
import argparse

# if(len(sys.argv)<3):
# 	print("Usage: python <script> <input_image_path> <output_image_path> <scale_factor:1 for no scaling>")
# 	exit(0)

def scale(img_inp, factor=1, interpolate=False):
    img = np.copy(img_inp)

    if (factor == 1):
        return img

    if (factor <= 0):
        print("Incorrect Factor Value - Must be >0")
        exit(0)

    height, width, depth = img.shape
    print(height, width)

    new_height = int(factor * height)
    new_width = int(factor * width)
    print(new_height, new_width)

    scaled_img = np.zeros((new_height, new_width, depth), np.uint8)

    for h in range(new_height):
        for w in range(new_width):
            inter_h = (h / factor)
            inter_w = (w / factor)
            index_h = int(inter_h)
            index_w = int(inter_w)

            if (interpolate == False):
                scaled_img[h, w, :] = img[index_h, index_w, :]
                continue

            if (index_h + 1 < height and index_w + 1 < width):
                topleft = img[index_h, index_w, :].astype(np.float)
                topright = img[index_h, index_w + 1, :].astype(np.float)
                bottomleft = img[index_h + 1, index_w, :].astype(np.float)
                bottomright = img[index_h + 1, index_w + 1, :].astype(np.float)

                interpolate_1 = topleft * (inter_w - index_w) + topright * (index_w + 1 - inter_w)
                interpolate_2 = bottomleft * (inter_w - index_w) + bottomright * (index_w + 1 - inter_w)

                final_interpolate = interpolate_1 * (inter_h - index_h) + interpolate_2 * (index_h + 1 - inter_h)
                final_interpolate = final_interpolate.astype(np.uint8)
                scaled_img[h, w, :] = final_interpolate

            else:
                scaled_img[h, w, :] = img[index_h, index_w, :]

    return scaled_img

parser = argparse.ArgumentParser(description='Image Scaling')
parser.add_argument('--input', dest='input_image_path', help="Enter Input Image Path", required=True, type=str)
parser.add_argument('--scale', dest='scale', help="Enter Scale Factor", default=1, type=float)
parser.add_argument('--interpolate', action='store_true', default='False', help="Perform Replication")
parser.add_argument('--replicate', action='store_true', default='False', help="Perform Replication")
parser.add_argument('--sizex', dest='sizex', default=50, help="Window size x", type=int)
parser.add_argument('--sizey', dest='sizey', default=50, help="Window size y", type=int)

args = parser.parse_args()
img = cv2.imread(args.input_image_path, cv2.IMREAD_COLOR)
height, width, depth = img.shape


# final_img = None
# if (args.interpolate == True):
#     final_img = scale(img, args.scale, True)
# else:
#     final_img = scale(img, args.scale, False)


def show_zoom(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(x,y)
        # z = final_img[int(y*args.scale)-args.sizey:int(y*args.scale) + args.sizey, int(x*args.scale)-args.sizex:int(x*args.scale) + args.sizex]
        lefty=max(0,y-args.sizey)
        righty=min(height,y+args.sizey)
        leftx=max(0,x-args.sizex)
        rightx=min(width,x+args.sizex)
        flag=False

        if(args.interpolate==True):
        	flag=True
        
        cv2.imshow('zoom', scale(img[lefty:righty,leftx:rightx],args.scale,flag))


cv2.namedWindow('src')
cv2.setMouseCallback('src',show_zoom)

while (1):
    cv2.imshow('src', img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
# scale_factor=float(sys.argv[3])
# img=cv2.imread(sys.argv[1],cv2.IMREAD_COLOR)
# cv2.imwrite(sys.argv[2],scale(img,scale_factor))


# cv2.imwrite(args.output_image_path, final_img)
