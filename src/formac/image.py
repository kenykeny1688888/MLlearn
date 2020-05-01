# -*- coding: utf-8 -*-
__author__ = 'kenydachan'
import cv2

# 0 代表笔记本的摄像头,(不需要打开摄像头，cv会自动打开摄像头)
video = cv2.VideoCapture(0)
while (1):
    # read返回两个值 一个 是否读取成功  一个是每一帧的视频数据
    stream, frame = video.read()
    if not stream:
        break

    # 将读取到的视频显示出来
    cv2.imshow("video", frame)
    # waitKey 可以监听键盘输入,当你把光标点击到视频的时候按下键盘就能获取到输入了哪个键，因此我这里做了监听 如果按下 q 键 则退出
    key = cv2.waitKey(2)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
