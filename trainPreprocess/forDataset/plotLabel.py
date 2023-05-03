import os

import cv2 as cv


def plot_label(index):
    path = "../../dataset"
    index = str(index).zfill(3)  # 将数字转化为格式化数字字符串，用以进一步地形成文件名

    img = cv.imread(path+"/data_stage4/"+index+".jpg")

    r = 352
    # r = 342

    with open(path+"/data_raw/label/"+index+".txt", 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line != 'Perfect':
                line = line.split(',')
                x = r + int(line[1]) + 5  # x坐标
                y = r + int(line[2]) + 5  # y坐标
                w = int(line[3]) + 15  # 宽
                h = int(line[4]) + 15  # 高
                # w = int(line[3])  # 宽
                # h = int(line[4])  # 高

                xlt = int(x - w/2)
                ylt = int(y - h/2)
                xrb = xlt + w
                yrb = ylt + h

                img = cv.rectangle(img, (xlt, ylt), (xrb, yrb), (0, 0, 255), 1)

    cv.imwrite(path+"/data_stage5/visualization_train/"+index+'.jpg', img)


if __name__ == '__main__':
    for i in range(1, 121):
        plot_label(i)