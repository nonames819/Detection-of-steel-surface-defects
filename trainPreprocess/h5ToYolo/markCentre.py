import os
import cv2 as cv
import numpy as np
from multiprocessing import Pool
# 不用先将图像进行模糊去噪处理，否则反而效果不好。猜测是由于事件相机产生的物体边缘信号具有足够的辨识度，容易从噪声中区分出来
# dp：边缘检测时，产生图像的缩放倍数。dp=2意为在分辨率为原图1/2倍的图像上进行边缘检测
# param1：其越大，边缘识别越宽松
# param2：边缘图上置信度高于param2的圆将被识别出来
# minDist设为2000以保证图像内只识别出一个圆


'''
从若干候选圆中选出与原图像素重叠度最高的圆
并且只选出像素重叠度大于overlap_rate的圆
'''
def draw_optimal_circle(img_blur, circles, candidates, overlap_rate):
    # 筛选最优圆
    best_score = 0  # 记录最高得分
    best_circle = None  # 记录最优预测圆
    for (x, y, r) in circles[0][:candidates]:  # 至多只取candidates个候选圆
        circle_img_pred = np.zeros(img_blur.shape)  # 黑底图像作为背景画布
        cv.circle(circle_img_pred, (x, y), r, 255, 2)  # 画预测圆
        circle_img_pred = np.uint16(circle_img_pred)

        score_map = np.uint16(img_blur) + circle_img_pred  # 实验发现将预测圈与去噪后的原图像计算重合率，判定更准
        overlaps = np.where(score_map > 255, True, False)  # 值大于255的像素即为预测圈与图像重叠的像素
        score = np.count_nonzero(overlaps) / r  # 表征像素重叠率

        if score > overlap_rate and score > best_score:  # 取得分最高的圆作为该帧的最终预测圆，且得分需高于阈值overlap_rate
            best_score = score
            best_circle = (x, y, r)
            print(best_score)
    return best_circle


'''
改进后的Hough圆检测
'''
def find_centre(img):  # 输入8位灰度图像，高宽格式(H,W)=(800,1280)。返回标注了检测出的圆及圆心的灰度图片
    im0 = img.copy()

    # 识别铝片内圆
    img_inner = cv.medianBlur(im0[300:500, :], 3)
    # img_inner = im0[300:500, :]  # 不模糊处理会导致边缘检测速度慢
    circles_inner = cv.HoughCircles(img_inner, cv.HOUGH_GRADIENT, dp=2, param1=30, param2=50,
                                    minRadius=80, maxRadius=120, minDist=1)

    '''******************************！！！以下这段代码是程序速度的瓶颈，耗时与candidates数成正比！！！******************************'''
    # 筛选最优预测圆
    find_flag = False
    centre = None
    if circles_inner is not None:
        circles_inner = np.uint16(np.around(circles_inner))  # 把小圆的圆心和半径的值转为整型

        # 筛选最优小圆
        best_inner_circle = draw_optimal_circle(img_blur=img_inner, circles=circles_inner, candidates=5, overlap_rate=12)

        '''******************************！！！以上这段代码是程序速度的瓶颈，耗时与candidates数成正比！！！******************************'''

        if best_inner_circle is not None:  # 可视化预测圆
            x1, y1, r1 = best_inner_circle
            find_flag = True
            centre = (int(x1), int(y1+300))
            cv.circle(im0, centre, r1, 255, 1)   # 画小圆
            cv.circle(im0, centre, 2, 255, 1)   # 画圆心
            # cv.imshow('output', im0)
            # cv.waitKey(0)
    return im0, centre, find_flag


'''
每个进程所执行的task
从事件帧中检测铝片圆心
读取indir路径下的帧文件，将圆心标注结果输出到outdir路径下
该task所处理的事件流编号为start到end-1
'''
def mark_centre_task(start, end, indir, outdir):
    for i in range(start, end):  # 遍历事件流
        index = str(i).zfill(3)  # 将数字转化为格式化数字字符串，用以进一步地形成文件名
        files = os.listdir(indir+'/'+index)  # 得到文件夹下所有文件名称

        os.makedirs(outdir+'/'+index+'/data')  # 创建文件夹存放该事件流中能标出圆心的帧
        os.makedirs(outdir+'/'+index+'/label')  # 创建文件夹存放该事件流产生的所有圆心标注坐标
        os.makedirs(outdir+'/'+index+'/visualization')  # 创建文件夹存放该事件流产生的所有圆心标注可视化图像

        frames = []
        for file in files:  # 遍历该事件流所产生的所有帧
            frames.append(cv.imread(indir+'/'+index+'/'+file, cv.IMREAD_GRAYSCALE))  # 读取一帧

        count_in_event = 0  # 统计某事件流内找出的优质圆心数量
        for frame in frames:
            marked_frame, centre, find_flag = find_centre(frame)
            if find_flag:
                count_in_event += 1

                cv.imwrite(outdir+'/'+index+'/data/'+str(count_in_event)+'.png', frame)  # 存放能识别出精确圆心的帧
                with open(outdir+'/'+index+'/label/'+str(count_in_event)+'.txt', 'w') as file:
                    x, y = centre
                    file.write(str(x)+','+str(y))  # 存放圆心坐标
                cv.imwrite(outdir+'/'+index+'/visualization/'+str(count_in_event)+'.png', marked_frame)  # 仅供可视化，观察圆及圆心的识别是否精准

        print("[data stage 2] event {} is done, find {} nice circle(s)".format(index, count_in_event))


'''
从事件帧中检测铝片圆心
读取indir路径下的事件帧文件，将圆心标注结果输出到outdir路径下
使用多进程处理，以提高cpu利用率，若运算资源不足，则需减小process_num
indir下的事件帧文件总数必须为process_num的整数倍
'''
def mark_centre(indir, outdir):
    process_num = 4  # 进程数，indir下的事件帧文件总数必须为process_num的整数倍，process_num数应小于cpu可用核数

    # 获得需处理的事件帧文件的个数
    folder_counts = 0
    file_list = os.listdir(indir)
    for file in file_list:
        if os.path.isdir(indir+"/"+file) is True:
            folder_counts += 1
    # print(folder_counts)

    pool = Pool(processes=process_num)
    l = int(folder_counts / process_num)
    for file in range(0, process_num):
        start = l * file + 1
        end = start + l

        pool.apply_async(mark_centre_task, (start, end, indir, outdir,))
    pool.close()
    pool.join()


if __name__ == '__main__':
    mark_centre(indir="../../dataset/data_stage1", outdir="../../dataset/data_stage2")
