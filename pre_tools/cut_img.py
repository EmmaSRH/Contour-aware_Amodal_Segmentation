import cv2
import glob

img_path = '/Users/shiwakaga/Amodel_Data/test/instrument7/images/*.png'

img_list = glob.glob(img_path)

for img in img_list:
    pic_o = cv2.imread(img)
    img_ins =  img.split('/')[-3]
    img_name = img.split('/')[-1]
    print(pic_o.shape)
    pic_new = pic_o[28:1052, 320:1600,:]
    # print(pic_new.shape)
    cv2.imwrite('/Users/shiwakaga/Amodel_Data/test/test/'+img_ins+'/'+img_name, pic_new)