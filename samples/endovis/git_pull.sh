#!/bin/bash
#方法一
dir=$(ls -l /usr/ |awk '/^d/ {print $NF}')
for i in $dir
do
 git pull ..
done
#######
#方法二
for dir in $(ls /usr/)
do
 [ -d $dir ] && echo $dir
done
##方法三

ls -l /usr/ |awk '/^d/ {print $NF}' ##