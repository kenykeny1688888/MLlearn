import cv2
import numpy as np
img=np.zeros((8,8), dtype=np.uint8)
print("img=\n", img)
cv2.imshow("8*8 one", img)
print("读取像素点img[0,4]=", img[0,4])

for i in range(0,8):
    img[i,4]=255
for i in range(0,8):
    img[4,i]=255

for i in range(0,8):
    img[i,3]=128
for i in range(0,8):
    img[3,i]=128



print("修改后img=\n", img)
print("读取修改后像素点img[0,3]=", img[0,3])
cv2.imshow("two", img)
cv2.waitKey()

cv2.destroyAllWindows()
for j in range(1,100000):
    print(j)

