import glob

train_path = '/Users/shiwakaga/Amodel_Data/train/*/images/*.jpg'
test_path = '/Users/shiwakaga/Amodel_Data/test/*/images/*.png'

train_list = glob.glob(train_path)[:1350]
val_list = glob.glob(train_path)[1350:]
test_list = glob.glob(test_path)

# for img in train_list:
#     img_name = img.replace('/Users/shiwakaga/','/data/srh/')
#     label_id = img_name.split('/')[-1][:-4]
#     with open('train.txt','a') as f:
#         f.writelines(img_name + ' ' + label_id + '\n')
# for img in val_list:
#     img_name = img.replace('/Users/shiwakaga/','/data/srh/')
#     label_id = img_name.split('/')[-1][:-4]
#     with open('val.txt','a') as f:
#         f.writelines(img_name + ' ' + label_id + '\n')
# for img in test_list:
#     img_name = img.replace('/Users/shiwakaga/','/data/srh/')
#     label_id = img_name.split('/')[-1][:-4]
#     with open('test.txt','a') as f:
#         f.writelines(img_name + ' ' + label_id + '\n')

for i in range(1,7,1):
    for j in range(225):
        if j < 10:
            img_name = '/data/srh/Amodel_Data/train/instrument'+str(i)+'/images/frame00'+str(j)+'.jpg'
            label_id = 'frame00'+str(j)
        else:
            if j < 100:
                img_name = '/data/srh/Amodel_Data/train/instrument' + str(i) + '/images/frame0' + str(j) + '.jpg'
                label_id = 'frame0' + str(j)
            else:
                img_name = '/data/srh/Amodel_Data/train/instrument' + str(i) + '/images/frame' + str(j) + '.jpg'
                label_id = 'frame' + str(j)
        with open('train.txt', 'a') as f:
            f.writelines(img_name + ' ' + label_id + '\n')
for i in range(7,9,1):
    for j in range(225):
        if j < 10:
            img_name = '/data/srh/Amodel_Data/train/instrument'+str(i)+'/images/frame00'+str(j)+'.jpg'
            label_id = 'frame00'+str(j)
        else:
            if j < 100:
                img_name = '/data/srh/Amodel_Data/train/instrument' + str(i) + '/images/frame0' + str(j) + '.jpg'
                label_id = 'frame0' + str(j)
            else:
                img_name = '/data/srh/Amodel_Data/train/instrument' + str(i) + '/images/frame' + str(j) + '.jpg'
                label_id = 'frame' + str(j)
        with open('val.txt', 'a') as f:
            f.writelines(img_name + ' ' + label_id + '\n')
for i in range(1,9,1):
    for j in range(225,300,1):
        if j < 10:
            img_name = '/data/srh/Amodel_Data/train/instrument'+str(i)+'/images/frame00'+str(j)+'.png'
            label_id = 'frame00'+str(j)
        else:
            if j < 100:
                img_name = '/data/srh/Amodel_Data/train/instrument' + str(i) + '/images/frame0' + str(j) + '.png'
                label_id = 'frame0' + str(j)
            else:
                img_name = '/data/srh/Amodel_Data/train/instrument' + str(i) + '/images/frame' + str(j) + '.png'
                label_id = 'frame' + str(j)
        with open('test.txt', 'a') as f:
            f.writelines(img_name + ' ' + label_id + '\n')