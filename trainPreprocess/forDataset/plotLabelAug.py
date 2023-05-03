import os

import cv2 as cv


def plot_label():
    path = "C:/Users/86152/Desktop/"

    file_list = os.listdir(path+"z_images")
    for file in file_list:
        img = cv.imread(path+"z_images/"+file)
        with open(path+"z_labels/"+file.strip(".jpg")+".txt") as f:
            lines = f.readlines()
            for line in lines:
                l = line.split(' ')
                c, x, y, w, h = l[0], l[1], l[2], l[3], l[4]
                x = int(float(x) * 704)
                y = int(float(y) * 704)
                w = int(float(w) * 704)
                h = int(float(h) * 704)

                xlt = int(x - w / 2)
                ylt = int(y - h / 2)
                xrb = xlt + w
                yrb = ylt + h

                if c is "0":
                    color = (0, 0, 255)
                elif c is "1":
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)
                cv.rectangle(img, (xlt, ylt), (xrb, yrb), color, 1)

        cv.imwrite(path+"/visualization_aug/"+file, img)


if __name__ == '__main__':
    plot_label()