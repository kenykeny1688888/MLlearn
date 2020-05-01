# -*- coding: utf-8 -*-
__author__ = 'kenydachan'
import dlib
import cv2


#打开摄像头，要确定运行读取摄像头
video = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
# 用终端操作
def read_camera0():
    """
    读取电脑摄像头的视频(不需要打开摄像头，cv会自动打开摄像头)
    """
    while 1:
        stream, frame = video.read()
        if stream:
            yield frame
        else:
            print("Cannot Read Camera0")
            break

def show_camera_faces():
    """
    读取摄像头数据，显示出来加上人脸检测
    """
    frames = read_camera0()
    for image in frames:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detect = detector(rgb_image, 1)

        for i, d in enumerate(detect):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            #设置红色的框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv2.imshow("camera video", image)
        key = cv2.waitKey(2)
        print(key)
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    show_camera_faces()
