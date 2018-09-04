import numpy as np
import cv2

factor = 2
size = 100


def show_zoom(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(x,y)
        z1 = zoomimg1[y*factor-size:y*factor + size, x*factor-size:x*factor + size]
        z2 = zoomimg2[y*factor-size:y*factor + size, x*factor-size:x*factor + size]
        cv2.imshow('zoom1', z1)
        cv2.imshow('zoom2', z2)

fac = float(factor)
img = cv2.imread('src.jpg', cv2.IMREAD_COLOR)
h, w, r = img.shape
zoomimg1 = np.zeros((h * factor, w * factor, r), np.uint8)
zoomimg2 = np.zeros((h * factor, w * factor, r), np.uint8)
for i in range(h):
    for j in range(w):
        temp = img[i][j]
        for p in range(factor):
            for q in range(factor):
                zoomimg1[i * factor + p][j * factor + q] = temp
for i in range(h - 1):
    for j in range(w - 1):
        temp1 = img[i][j]
        temp2 = img[i][j + 1]
        temp3 = img[i + 1][j]
        temp4 = img[i + 1][j + 1]
        for p in range(factor):
            for q in range(factor):
                zoomimg2[i * factor + p][j * factor + q] = (((fac-q)*temp1+q*temp2)*(fac-p) + p*((fac-q)*temp3+q*temp4))/(fac*fac)
cv2.namedWindow('src')
cv2.setMouseCallback('src',show_zoom)

while (1):
    cv2.imshow('src', img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
