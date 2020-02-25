import cv2
import glob
import numpy as np

img_path = '/Users/shiwakaga/Amodel_Data/instrument3/split_image/*.png'
img_files = glob.glob(img_path)

out_path = '/Users/shiwakaga/Amodel_Data/instrument3/'

for img_file in img_files:
    print(img_file)
    img_name = img_file.split('/')[-1]
    img = cv2.imread(img_file)
    # print(img.shape)
    img_new_1 = img.copy()
    img_new_2 = img.copy()
    h,w = img.shape[0],img.shape[1]
    # print(int(w/2))
    #
    # # hengqie
    # for k in range(200,h,1):
    #     if [255,255,255] not in img[k,:]:
    #         n = k
    #         break
    # print(n)
    # for i in range(n,h,1):
    #     img_new_1[i,:] =  [0,0,0]
    # for i in range(0,n,1):
    #     img_new_2[i,:] = [0, 0, 0]

    # shuqie
    for k in range(30,w,1):
        if [255,255,255] not in img[:,k]:
            n = k
            break
    print(n)
    if n != 30:
        for i in range(n,w,1):
            img_new_1[:,i] =  [0,0,0]
        for i in range(0,n,1):
            img_new_2[:,i] = [0, 0, 0]


        cv2.imwrite(out_path + img_name,img_new_1)
        cv2.imwrite(out_path + img_name[:-5]+'2.png', img_new_2)
