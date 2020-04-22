# import cv2
# img=cv2.imread("/Users/apple/Downloads/1.jpg",0)
# cv2.imshow("before", img)
# for i in range(10,100):
#     for j in range(80,100):
#       img[i, j]=255
# cv2.imshow("after", img)
# cv2.waitKey()
# cv2.destroyAllWindows()



import numpy as np
import cv2
#-----------蓝色通道值--------------
blue=np.zeros((300,300,3), dtype=np.uint8)
blue[:, :,0]=255
print("blue=\n", blue)
cv2.imshow("blue", blue)
#-----------绿色通道值--------------
green=np.zeros((300,300,3), dtype=np.uint8)
green[:, :,1]=255
print("green=\n", green)
cv2.imshow("green", green)
#-----------红色通道值--------------
red=np.zeros((300,300,3), dtype=np.uint8)
red[:, :,2]=255
print("red=\n", red)
cv2.imshow("red", red)
#-----------释放窗口--------------
cv2.waitKey()
cv2.destroyAllWindows()
for i in range(1,1000000):
    print("i");
