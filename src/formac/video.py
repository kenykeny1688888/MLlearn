# -*- coding: utf-8 -*-
__author__ = 'kenydachan'

import dlib
import cv2
# 加载并初始化检测器
# 需要这个文件，运行前前先下载下面bz2的文件，并且解压到当前目录下的predata目录下
# 模型下载地址http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
detector = dlib.get_frontal_face_detector()
#需要下载90m大小
predictor = dlib.shape_predictor('./predata/shape_predictor_68_face_landmarks.dat')
# 0 代表笔记本的摄像头,(不需要打开摄像头，cv会自动打开摄像头)
camera = cv2.VideoCapture(0)
#判断是否开启
if not camera.isOpened():
    print("cannot open camear")
    exit(0)

while True:
    ret, frame = camera.read()
    #循环判断，摄像头，您可以尝试一下移动，可以动态识别。
    if not ret:
        continue
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 检测脸部
    dets = detector(rgb_image, 1)
    print("Number of faces detected: {}".format(len(dets)))
    # 查找脸部位置
    for i, face in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} ".format(
            i, face.left(), face.top(), face.right(), face.bottom()))
        # 绘制脸部位置
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 1)
        shape = predictor(rgb_image, face)
        # 绘制特征点 红色点
        for i in range(68):
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 3, (0, 0, 255), 4)
            cv2.putText(frame, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0),
                        1)

    cv2.imshow("Camera", frame)
    #记入mark点
    cv2.imwrite("mark.png",frame)
    key = cv2.waitKey(2)
    #按键用q退出应用
    if key == ord('q'):
        break

cv2.destroyAllWindows()
