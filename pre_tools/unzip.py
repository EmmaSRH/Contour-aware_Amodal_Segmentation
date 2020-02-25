import os
import glob
import zipfile


path = '/data/srh/Training/additional_files/*/*/*/*.zip'
zips = glob.glob(path)
print(len(zips))

def unzip_file(zip_src, dst_dir):
    os.mkdir(dst_dir)
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)
    else:
        print('This is not zip')

for zip in zips:
    print(zip[:-4])
    unzip_file(zip,zip[:-4])
