# -*- coding: UTF-8 -*-
import cv2
from matplotlib import pyplot as plt
#
# img = cv2.imread('/Users/shiwakaga/Amodel_Data/train/instrument1/amodel/frame056_ins_1.png')
# a = cv2.copyMakeBorder(img,0,50,0,50,cv2.BORDER_CONSTANT,value=[0,0,0])
# gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
# # cv2.imshow("draw_img0", gray)
#
# ret,binary = cv2.threshold(a, 127, 255, 0)
#
# _,contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #只检测出最外轮廓即c0
#
# # draw_img = []
# color = [(0,255,0), (255,0,255), (0,0,255), (0,255,255),(0,0,0),(255,255,255)]
#
# print ("contours 数量：",len(contours))
#
# for i in range(len(contours)):
#     print(color[i])
#     draw_img = cv2.drawContours(a.copy(), contours, i, color[i], 3)
#     print(draw_img.shape)
#     # print("contours:类型：", type(contours))
#     print("第", i , " 个contours:的个数：", len(contours))
#
#     plt.figure()  # 绘制轮廓图
#     plt.imshow(gray, cmap="gray")
#     plt.imshow(draw_img)
#     plt.show()
#
#
#
# moments = cv2.moments(contours[0])
# humoments = cv2.HuMoments(moments)
#
# print(humoments)
#
# img2 = cv2.imread('/Users/shiwakaga/Amodel_Data/train/instrument1/amodel/frame056_ins_1.png')
# a1 = cv2.copyMakeBorder(img2,0,50,0,50,cv2.BORDER_CONSTANT,value=[0,0,0])
# gray = cv2.cvtColor(a1, cv2.COLOR_BGR2GRAY)
# ret2,binary2 = cv2.threshold(a, 127, 255, 0)
# _,contours2, hierarchy2 = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# moments2 = cv2.moments(contours2[0])
# humoments2 = cv2.HuMoments(moments2)
#
# print(humoments2)
# draw_img = cv2.drawContours(a.copy(), contours2[0], 1, (0,255,0), 3)
# plt.figure()  # 绘制轮廓图
# plt.imshow(draw_img)
# plt.show()
# #
# # draw_img2 = cv2.drawContours(a1.copy(), contours2, 0, (0,255,0), 3)
# # plt.figure()  # 绘制轮廓图
# # plt.imshow(draw_img2)
# # plt.show()
# #
# sim = cv2.matchShapes(contours[0],contours2[0],1, 0)
# print(sim)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()



img = cv2.imread('/Users/shiwakaga/Amodel_Data/train/instrument1/amodel/frame001_ins_0.png')
a1 = cv2.copyMakeBorder(img,0,50,0,50,cv2.BORDER_CONSTANT,value=[0,0,0])
# print(a1.shape)
gray = cv2.cvtColor(a1, cv2.COLOR_BGR2GRAY)
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours1 = contours[0]
draw_img1 = cv2.drawContours(gray.copy(), contours1, 0, (255,0,255), 3)
plt.figure()  # 绘制轮廓图
plt.imshow(draw_img1)
plt.show()
cv2.imwrite('img1.png',draw_img1)

img = cv2.imread('/Users/shiwakaga/Amodel_instrument/output_mrcnn/instrument1/frame001_ins_0.png')
a2 = cv2.copyMakeBorder(img,0,50,0,50,cv2.BORDER_CONSTANT,value=[0,0,0])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours2 = contours[0]

draw_img2 = cv2.drawContours(gray.copy(), contours2, 0, (255,0,255), 3)
plt.figure()  # 绘制轮廓图
plt.imshow(draw_img2)
plt.show()
cv2.imwrite('img2.png',draw_img2)

sim = cv2.matchShapes(contours1,contours2,1, 0)
print(sim)