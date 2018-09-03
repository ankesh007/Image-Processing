import cv2
import numpy as np
import sys

sigma = 0.1
k = 1.6
tau = 0.99
eps = 0.67
phi = 5.5


if(len(sys.argv)<3):
  print("Usage: python <script> <input_image_path> <output_image_path>")

def DoG(src, sigma, k, tau):
    g1 = cv2.GaussianBlur(src, (5, 5), sigma)
    g2 = cv2.GaussianBlur(src, (5, 5), k * sigma)

    dog = g1 - tau * g2
    return dog


def XDoG(src, sigma, k, tau, eps, phi0):
    dog = DoG(src, sigma, k, tau)
    out = dog.copy()

    out[dog >= eps * 255] = 255
    out[dog < eps * 255] = 255 * (1 + np.tanh(phi * (dog[dog < eps * 255]/255-eps)))

    cv2.imshow('dog', dog)
    cv2.imshow('xdog', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return out


# img = cv2.imread('src.jpg', cv2.IMREAD_COLOR)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('bg', img)
# XDoG(img, sigma, k, tau, eps, phi)


img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(sys.argv[2], XDoG(img, sigma, k, tau, eps, phi))
