import cv2
import numpy as np

img1 = cv2.imread('frame217.png',0)
img2 = cv2.imread('frame218.png',0)
img3 = cv2.imread('frame057.png',0)

ret,thresh = cv2.threshold(img1, 127, 255, 0)
ret2,thresh2 = cv2.threshold(img2,127,255,0)
ret3,thresh3 = cv2.threshold(img3,127,255,0)
_,contours,hierarchy = cv2.findContours(thresh,3,2)
cnt1 = contours[0]
_,contours2,hierarchy2 = cv2.findContours(thresh2,3,2)
cnt2 = contours2[0]
_,contours3,hierarchy3 = cv2.findContours(thresh3,3,2)
cnt3 = contours3[0]

ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
ret1 = cv2.matchShapes(cnt1,cnt3,1,0.0)
ret2 = cv2.matchShapes(cnt1,cnt1,1,0.0)

print (ret,ret1,ret2)