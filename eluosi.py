import numpy as np
import cv2
long=600
wide=300

img=np.zeros((wide,long,3), dtype=np.uint8)

# 白、蓝、红三色
#三个图层
for i in range(0,100):
    for j in range(0,long):
        img[i,j]=255
for i in range(100, 200):
    for j in range(0, long):
        img[i, j] =[255,0,0]
for i in range(200,300):
    for j in range(0,long):
        img[i,j,2]=255



# img[:,0:100,0]=255
# img[:,100:200,1]=255
# img[:,200:600,2]=255
print("img=\n", img)
cv2.imshow("Flag of Russia", img)
cv2.waitKey()
cv2.destroyAllWindows()
for i  in range(1,100000):
    print(i)

