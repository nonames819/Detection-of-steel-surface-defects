import os
from multiprocessing import Pool

from PIL import Image


'''
正方形图像保留内切圆内的部分，其余部分用黑色填充
'''
def circle(path):
    ima = Image.open(path).convert("RGBA")
    # 外圆半径
    r_outer = 352
    # r_outer = 342
    # 外圆直径
    d_outer = r_outer * 2
    # 圆心横坐标
    r = int(d_outer / 2)
    imb = Image.new('RGBA', (r_outer * 2, r_outer * 2), (255, 255, 255, 0))
    pima = ima.load()  # 像素的访问对象
    pimb = imb.load()

    for i in range(d_outer):
        for j in range(d_outer):
            lx = abs(i - r)  # 到圆心距离的横坐标
            ly = abs(j - r)  # 到圆心距离的纵坐标
            l = (pow(lx, 2) + pow(ly, 2)) ** 0.5  # 三角函数 半径
            if l < r_outer:
                pimb[i, j] = pima[i, j]
            else:
                pimb[i, j] = (0, 0, 0, 255)

    imb.save(path)
    return


'''
正方形图像保留内切圆内的部分，其余部分用黑色填充
该task所处理的事件流编号为start到end-1
'''
def to_circle_task(start, end, path):
    for i in range(start, end):
        index = str(i).zfill(3)
        for dirpath, dirnames, filenames in os.walk(path+'/'+index):
            for file in filenames:
                circle(path+'/'+index+'/'+file)
            print("[data stage 3: to circle] event {} is done".format(index))


'''
正方形图像保留内切圆内的部分，其余部分用黑色填充
使用多进程处理，以提高cpu利用率，若运算资源不足，则需减小process_num
indir下的事件帧文件总数必须为process_num的整数倍
'''
def to_circle(path):
    process_num = 10  # 进程数，path下的事件帧文件总数必须为process_num的整数倍，process_num数应小于cpu可用核数

    # 获得需处理的事件帧文件的个数
    folder_counts = 0
    file_list = os.listdir(path)
    for file in file_list:
        if os.path.isdir(path+"/"+file) is True:
            folder_counts += 1
    # print(folder_counts)

    pool = Pool(processes=process_num)
    l = int(folder_counts / process_num)
    for i in range(0, process_num):
        start = l * i + 1
        end = start + l

        pool.apply_async(to_circle_task, (start, end, path,))
    pool.close()
    pool.join()


if __name__ == '__main__':
    to_circle(path="../../TestASet/data_stage3")
