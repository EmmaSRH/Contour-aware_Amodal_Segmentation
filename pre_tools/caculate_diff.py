import numpy as np
import cv2
import glob

def caculate_img_diff(new_masks,old_masks):
    num_new = 0
    for new_mask in new_masks:
        num_new += np.sum(new_mask == [255,255,255])
    num_o = np.sum(old_mask != [0,0,0])

    return num_new,num_o

if __name__ == '__main__':

    num_all_old_pixles = 0
    num_all_new_pixels = 0


    for i in range(1,9,1):
        new_mask_path = '/Users/shiwakaga/Amodel_Data/test/instrument'+str(i)+'/amodel/'
        old_mask_path = '/Users/shiwakaga/Amodel_Data/test/instrument'+str(i)+'/instruments_masks/'

        all_new, all_old = 0, 0

        num_overlap = 0

        if i==9:
            break

        for j in range(225,300,1):
            if j<10:
                num = '00'+str(j)
            else:
                if j<100:
                    num = '0'+str(j)
                else:
                    num = str(j)
            old_mask_img = old_mask_path + 'frame' + num + '.png'
            new_mask_img = glob.glob(new_mask_path + 'frame' + num +'_ins_*.png')
            # print(len(new_mask_img))

            new_masks = []
            for new_img in new_mask_img:
                new_masks.append(cv2.imread(new_img))

            old_mask = cv2.imread(old_mask_img)

            num_new,num_o = caculate_img_diff(new_masks,old_mask)
            # print('The new and old num of instrument' + str(i) + ' is: ', num_new, num_o)
            # print('The add pixel num of instrument' + str(i) + ' is: ', num_new - num_o)

            with open('count_test.txt','a') as f:
                f.write('instrument'+ str(i) +','+'frame'+str(num) +','+ str(len(new_mask_img)) +','+ str(num_new)+ ',' + str(num_o)+ ',' + str(num_new - num_o) + '\n')

            all_new += num_new
            all_old += num_o

        print('The new and old num of instrument' + str(i) + ' is: ', all_new, all_old)
        print('The add pixel num of instrument' + str(i) + ' is: ', all_new - all_old)
        print('The overlap rate is : ',(all_new - all_old)/all_old)
        num_all_old_pixles += all_old
        num_all_new_pixels += all_new

    print('The new and old num of instrument' + str(i) + ' is: ', num_all_new_pixels, num_all_old_pixles)
    print('The add pixel num of instrument' + str(i) + ' is: ', num_all_new_pixels-num_all_old_pixles)
    print('The overlap rate is : ', (num_all_new_pixels-num_all_old_pixles) / num_all_old_pixles)

