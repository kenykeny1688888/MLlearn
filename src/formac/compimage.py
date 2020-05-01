# -*- coding: utf-8 -*-
__author__ = 'kenydachan'
import cv2
import dlib

picture = '/Users/keny/Downloads/1.png'
detector = dlib.get_frontal_face_detector()

def add_face_from_image(image):
    imdata = cv2.imread(image)
    print(imdata)
    rgb_image = cv2.cvtColor(imdata, cv2.COLOR_BGR2RGB)
    faces = detector(rgb_image, 1)

    print(faces)
    if len(faces) == 0:
        print("没有检测到人脸")
    else:
        print('获取到了人脸数据个数：{}'.format(len(faces)))

        # 识别人脸特征点，并保存下来
if __name__ == '__main__':
    add_face_from_image(picture)
