import os
from multiprocessing import Pool

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


'''
将事件帧裁切为正方形，以铝片圆心为中心
读取input路径下的帧和圆心标注，将裁切后的图像输出在output路径下
'''
def modify(input, output):
    label = input + '/label'
    # print(input)
    # print(label)
    data = input + '/data'

    # 存储中心信息
    path = label + '/1.txt'
    i=1
    list = {}
    while(os.path.exists(path)):
        file = open(label+'/'+str(i)+'.txt', 'r')
        x, y = file.read().split(',')
        list[i] = [x,y]
        # x=list[i][0] y=list[i][1]
        i += 1
        path = label + '/' + str(i) + '.txt'

    # 遍历所有样本
    path = data + '/1.png'
    i = 1
    image_name = '1.png'
    while(os.path.exists(path)):
        # 打印图片名字
        # print(str(i)+" path:"+path)
        im = Image.open(path)
        # 需要设置rgb 并且将处理过的图片存储在别的变量下
        im = im.convert('RGB')
        # 重新设置大小（可根据需求转换）
        centre_x = (int)(list[i][0])
        centre_y = (int)(list[i][1])
        radius = 352
        # radius = 342

        box = (centre_x - radius, centre_y - radius, centre_x + radius, centre_y + radius)
        rem = im.crop(box)
        # 对处理完的正方形图片进行保存
        # obj_path = os.path.join(output, image_name)
        # print(obj_path)
        rem.save(output + '/' + image_name)
        i+=1
        path = data + '/' + str(i) + '.png'
        # print(os.path.exists(path))
        image_name = str(i)+'.png'


'''
每个进程所执行的task
将事件帧裁切为正方形，以铝片圆心为中心
读取indir路径下的帧和圆心标注，将裁切后的图像输出在indir路径下
该task所处理的事件流编号为start到end-1
indir下的事件帧文件总数必须为process_num的整数倍
'''
def to_square_task(start, end, indir, outdir):
    for i in range(start, end):
        index = str(i).zfill(3)
        input = indir+'/'+index
        # print(input)
        # 输出目录 ： 比例缩小图片
        output = outdir+'/'+index
        # 创建文件夹
        os.makedirs(output)  # 创建文件夹存放该事件流产生的所有帧
        modify(input, output)
        print("[data stage 3: to square] event {} is done".format(index))


'''
将事件帧裁切为正方形，以铝片圆心为中心
读取indir路径下的帧和圆心标注，将裁切后的图像输出在indir路径下
使用多进程处理，以提高cpu利用率，若运算资源不足，则需减小process_num
'''
def to_square(indir, outdir):
    process_num = 10  # 进程数，indir下的事件帧文件总数必须为process_num的整数倍，process_num数应小于cpu可用核数

    # 获得需处理的事件帧文件的个数
    folder_counts = 0
    file_list = os.listdir(indir)
    for file in file_list:
        if os.path.isdir(indir+"/"+file) is True:
            folder_counts += 1
    # print(folder_counts)

    pool = Pool(processes=process_num)
    l = int(folder_counts / process_num)
    for i in range(0, process_num):
        start = l * i + 1
        end = start + l

        pool.apply_async(to_square_task, (start, end, indir, outdir,))
    pool.close()
    pool.join()


if __name__ == '__main__':
    to_square(indir="../../TestASet/data_stage2", outdir="../../TestASet/data_stage3")
