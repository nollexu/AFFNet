#!/bin/bash
array=()
index=0
# standard="/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"
standard="colin27_MNI.nii"
function getdir(){
    for element in `ls $1`
    do  
        dir_or_file=$1"/"$element
        echo ${dir_or_file}
        array[$index]=$dir_or_file
        let index++
        echo ${index}
    done
}
function removeskull(){
    count=0
    for element in ${array[@]}
    do
        echo $element
        # 对齐方向
        echo "使用fslreorient2std对齐模板方向"
        fslreorient2std ${element} "${element%.*}_oriented.nii"
        # 裁剪颈部
        # echo "使用robustfov进行颈部裁剪"
        # robustfov -i "${element%.*}_oriented.nii.gz" -r "${element%.*}_clear_neck.nii"
        echo "裁剪颈部完成，输出后缀为clear_neck"
        # 配准到Colin27
        echo "进行配准，模板为colin27"
        # flirt -in "${element%.*}_clear_neck.nii.gz" -ref ${standard} -out "${element%.*}_colin27.nii"
        flirt -in "${element%.*}_oriented.nii.gz" -ref ${standard} -out "${element%.*}_colin27.nii"
        echo "配准完成，输出后缀为colin27"
        # 去除头骨
        echo "使用bet robust进行头骨去除"
        bet "${element%.*}_colin27.nii.gz" "${element%.*}_brain_colin27.nii" -R
        echo "去除头骨完成，输出后缀为brain_colin27"
        let count++  
        echo "第$count个MRI图像处理完成。"
    done
}
root_dir=".数据存放路径"
getdir $root_dir
removeskull
