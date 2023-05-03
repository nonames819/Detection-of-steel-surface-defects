import os
import cv2 as cv
import numpy as np


'''
堆栈降噪
读取indir路径下的事件帧，将堆栈后的图像输出在outdir路径下
处理编号为index的事件流
'''
def stack(index, indir, outdir):
    index = str(index).zfill(3)  # 将数字转化为格式化数字字符串，用以进一步地形成文件名
    files = os.listdir(indir+'/'+index)  # 得到文件夹下所有文件名称

    piled_img = np.zeros((704, 704))  # 初始化图像为全黑。图像宽为704，高为704（根据toSquarePlus产生的图像宽高而设定）
    # piled_img = np.zeros((684, 684))  # 初始化图像为全黑。图像宽为704，高为704（根据toSquarePlus产生的图像宽高而设定）
    for file in files:  # 遍历该事件流所产生的所有帧
        frame = cv.imread(indir+'/'+index+'/'+file, cv.IMREAD_GRAYSCALE)  # 读取一帧
        piled_img += frame  # 叠加

    max_lightness = np.max(np.hstack(piled_img))  # 图像中最高的灰度值
    if max_lightness != 0:
        piled_img *= 255 / max_lightness  # 标准化，将图像灰度值压缩到0~255范围

        # # 标准化，将图像灰度值压缩到0~255范围
        # piled_img /= max_lightness  # 压缩到0~1范围
        # piled_img = 1 / (1 + np.exp(4.5 - (15 * piled_img)))
        # piled_img *= 255  # 还原到0~255范围
    piled_img = piled_img.astype(np.uint8)  # 将图像灰度值转为int类型

    cv.imwrite(outdir+'/'+index+'.jpg', piled_img)  # 保存该帧图像

    print("[data stage 4] event {} is done".format(index))


'''
堆栈降噪
读取indir路径下的事件帧，将堆栈后的图像输出在outdir路径下
处理编号为index的事件流
'''
def stack_frames(indir, outdir):
    # 获得需处理的事件帧文件的个数
    folder_counts = 0
    file_list = os.listdir(indir)
    for file in file_list:
        if os.path.isdir(indir+"/"+file) is True:
            folder_counts += 1
    # print(folder_counts)

    os.makedirs(outdir)

    for i in range(1, folder_counts+1):
        stack(i, indir, outdir)


if __name__ == '__main__':
    stack_frames(indir="../../dataset/data_stage3", outdir="../../dataset/data_stage4")
