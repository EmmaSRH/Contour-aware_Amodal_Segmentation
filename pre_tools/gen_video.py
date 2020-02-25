import zipfile
import glob
import cv2
import numpy as np
import os

# # unzip
# all_path = glob.glob('/Users/shiwakaga/Downloads/additional_files/*/*/*/10s_video.zip')
# for video in all_path:
#     print(video)
#     with zipfile.ZipFile(video, 'r') as z:
#         file_name = video.split('.')[0]
#         if not os.path.exists(file_name):
#             os.mkdir(file_name)
#             z.extractall(file_name)
#             os.remove(video)
# use last 5 frame to generate video
all_path = glob.glob('/Users/shiwakaga/Downloads/additional_files/*/*/*/10s_video')
for video in all_path:
    # print(video)
    fps = 25
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out_dir = '/Users/shiwakaga/Downloads/video/'
    vi_name = out_dir + video.split('/')[-4]+'_'+video.split('/')[-3]+'_'+video.split('/')[-2]+'.avi'
    print(vi_name)
    # exit()
    videoWriter = cv2.VideoWriter(vi_name, fourcc, fps, (960, 540))  # 最后一个是保存图片的尺寸(width, height)
    for i in range(235, 249):
        img = cv2.imread(video + '/' + str(i) + '.png')
        videoWriter.write(img)
    videoWriter.release()