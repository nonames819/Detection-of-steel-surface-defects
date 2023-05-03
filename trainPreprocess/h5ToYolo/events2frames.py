import os
import h5py
import imageio
import numpy as np
import cv2 as cv
from multiprocessing import Pool


'''
处理第index个事件流
从h5文件产生出可视化帧
读取indir路径文件夹下的h5文件，将可视化帧输出在outdir路径文件夹下
可使用固定事件数量/固定时间窗口可视化
'''
def events_to_frames(index, indir, outdir, start_at=0.0, end_at=1.0, events_per_frame=0, frame_duration=0):
    index = str(index).zfill(3)  # 将数字转化为格式化数字字符串，用以进一步地形成文件名

    f = h5py.File(indir+'/'+index+'.H5', 'r')
    os.makedirs(outdir+'/'+index)  # 创建文件夹存放该事件流产生的所有帧

    dataset = f['events']  # 这个H5文件只有‘event’这一个key，dataset的结构是: dataset[keys][事件索引]

    events = []  # 用来存储事件流
    for key in dataset.keys():
        events.append(dataset[key][:])
    events = np.array(events)
    events = events.transpose((1, 0))  # event[事件索引][属性]，属性0~3分别为: event_g, t, x, y
    event_num = events.shape[0]  # 获得事件总数量

    '''************************************以下代码是固定事件数量形成一帧************************************'''
    if events_per_frame != 0:
        i = int(event_num * start_at)
        counter = i
        frame_index = 0
        frames= []
        while i < event_num * end_at:  # 遍历一段优质事件
            frame = np.zeros((1280, 800))  # 初始化当前帧的图像为全黑。图像宽为1280，高为800（根据事件相机分辨率为1280*800而设定）
            while (i < event_num * end_at) and (i < counter + events_per_frame):  # 叠加events_per_frame个事件，形成一帧
                x = int(events[i, 2])
                y = int(799 - events[i, 3])
                frame[x, y] += events[i, 0]  # 给图像上事件对应的像素点添上灰度。采用叠加像素值而非覆盖像素值能够去噪
                i += 1
            frame_index += 1
            counter += events_per_frame

            frame = frame.transpose((1, 0))  # 宽高格式(W,H)转为高宽格式(H,W)
            max_lightness = np.max(np.hstack(frame))  # 这一帧中最高的灰度值
            if max_lightness != 0:
                frame *= 255 / max_lightness  # 标准化，将图像灰度值压缩到0~255范围

            frame = frame.astype(np.uint8)  # 将图像灰度值转为int类型

            frames.append(frame)

            cv.imwrite(outdir+'/'+index+'/'+str(frame_index)+'.png', frame)  # 保存该帧图像

        # imageio.mimsave(path+'forDataset/'+index+'.gif', frames, 'GIF', duration=0.01)  # 可视化整段事件流为gif动图
        print("[data stage 1] event {} is done".format(index))

        '''************************************以下代码是固定时间窗口形成一帧************************************'''
    else:
        i = int(event_num * start_at)
        frame_start_time = events[i, 1]
        frame_index = 0
        while i < event_num * end_at:  # 遍历一段优质事件
            frame = np.zeros((1280, 800))  # 初始化当前帧的图像为全黑。图像宽为1280，高为800（根据事件相机分辨率为1280*800而设定）
            while (i < event_num) and (
                    events[i, 1] < frame_start_time + frame_duration):  # 在frame_duration秒内，叠加所有事件，形成一帧
                x = int(events[i, 2])
                y = int(799 - events[i, 3])
                frame[x, y] += events[i, 0]  # 给图像上事件对应的像素点添上灰度。采用叠加像素值而非覆盖像素值能够去噪
                i += 1
            frame_index += 1
            frame_start_time = frame_start_time + frame_duration  # 更新下一帧的起始时间戳

            frame = frame.transpose((1, 0))  # 宽高格式(W,H)转为高宽格式(H,W)
            max_lightness = np.max(np.hstack(frame))  # 这一帧中最高的灰度值
            if max_lightness != 0:
                frame *= 256 / max_lightness  # 标准化，将图像灰度值压缩到0~255范围
            frame = frame.astype(np.uint8)  # 将图像灰度值转为int类型

            cv.imwrite(indir+'/'+index+'/'+str(frame_index)+'.png', frame)  # 保存该帧图像

        print("[data stage 1] event {} is done".format(index))


'''
每个进程所执行的task
从h5文件产生出可视化帧
读取indir路径下的h5文件，将可视化帧输出在outdir路径下
该task所处理的事件流编号为start到end-1
'''
def e2f_task(start, end, indir, outdir):
    frame_duration = 0.01  # 在frame_duration秒内，叠加所有事件，形成一帧
    events_per_frame = 100000  # 叠加events_per_frame个事件，形成一帧（当events_per_frame超过100,000时拖尾严重，小于70,000时成像不完整）

    for i in range(start, end):
        # 从事件流产生帧，并保存，采用固定事件数量可视化
        events_to_frames(i, indir=indir, outdir=outdir, events_per_frame=events_per_frame)

        # # 从事件流产生帧，并保存，采用固定时间窗口可视化。采用此方法容易导致因物体移动速度变化，图像拖尾或成像不完整
        # events_to_frames(i, start_at=start_at, end_at=end_at, frame_duration=frame_duration)


'''
从h5文件产生出可视化帧
读取indir路径下的h5文件，将可视化帧输出在outdir路径下
使用多进程处理，以提高cpu利用率，若运算资源不足，则需减小process_num
indir下的事件帧文件总数必须为process_num的整数倍
'''
def h5_to_frames(indir, outdir):
    process_num = 10  # 进程数，indir下的h5文件总数必须为process_num的整数倍，process_num数应小于cpu可用核数

    # 获得需处理的h5文件的个数
    for dirpath, dirnames, filenames in os.walk(indir):
        file_counts = len(filenames)

    pool = Pool(processes=process_num)
    l = int(file_counts / process_num)
    for i in range(0, process_num):
        start = l * i + 1
        end = start + l

        pool.apply_async(e2f_task, (start, end, indir, outdir,))
    pool.close()
    pool.join()


if __name__ == '__main__':
    h5_to_frames(indir="../../dataset/data_raw/event", outdir="../../dataset/data_stage1")
