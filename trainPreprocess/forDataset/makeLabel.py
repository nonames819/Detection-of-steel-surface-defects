"""
将相对圆心的坐标转为裁切-堆栈后的图像上的绝对坐标
"""
def make_label(index):
    path = "../../dataset/"
    index = str(index).zfill(3)  # 将数字转化为格式化数字字符串，用以进一步地形成文件名

    r = 352
    # r = 342
    d = r * 2

    with open(path+'data_stage5/labels/'+index+'.txt', 'w') as f2:  # 目标文件
        with open(path+"data_raw/label/"+index+".txt") as f1:  # 源文件
            lines = f1.readlines()  # 读取所有行
            for line in lines:  # 遍历每一行
                if line != 'Perfect':
                    line = line.split(',')
                    c1 = int(line[0])  # 种类
                    x1 = r + int(line[1]) + 5  # x坐标
                    y1 = r + int(line[2]) + 5  # y坐标
                    w1 = int(line[3]) + 15  # 宽
                    h1 = int(line[4]) + 15  # 高
                    # x1 = r + int(line[1]) + 5  # x坐标
                    # y1 = r + int(line[2]) + 5  # y坐标
                    # w1 = int(line[3])  # 宽
                    # h1 = int(line[4])  # 高

                    # yolo v7中：label的坐标、宽高采用的是相对于图片的百分比长度，以下保留三位小数
                    c2, x2, y2, w2, h2 = int(c1-1), round(x1/d, 5), round(y1/d, 5), round(w1/d, 5), round(h1/d, 5)  # 输出yolo格式标签
                    # c2, x2, y2, w2, h2 = int(c1-1), x1, y1, w1, h1  # 输出像素为单位的标签
                    f2.write(str(c2)+' '+str(x2)+' '+str(y2)+' '+str(w2)+' '+str(h2)+'\n')  # 存放圆心坐标


if __name__ == '__main__':
    for i in range(1, 121):
        make_label(i)
