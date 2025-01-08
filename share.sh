#!/bin/bash

# 设置目标目录，这里假设是当前目录
TARGET_DIR=$(pwd)

# 创建share文件夹，如果它不存在
mkdir -p "${TARGET_DIR}/share"

cp "${TARGET_DIR}"/*.py "${TARGET_DIR}/share/"

# 检查是否有.py文件被移动
if [ "$(ls -A "${TARGET_DIR}/share")" ]; then
    # 压缩share文件夹
    zip -r "${TARGET_DIR}/share.zip" "${TARGET_DIR}/share"
    echo "share文件夹已压缩为share.zip"
else
    echo "没有找到.py文件，无法压缩share文件夹"
fi

# 清理，删除share文件夹
rm -r "${TARGET_DIR}/share"
echo "share文件夹已被删除"