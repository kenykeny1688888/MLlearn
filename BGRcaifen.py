import cv2
import numpy as np

img=cv2.imread("/Users/apple/Downloads/1.jpg")
img1=img
img2=img
cv2.imshow("source ",img)

b=img[:,:,0]
g=img[:,:,1]
r=img[:,:,2]

cv2.imshow("b pic ",b)
cv2.imshow("g pic ",g)
cv2.imshow("r pic ",r)

img[:,:,0]=0
cv2.imshow("source b==0",img)
img1[:,:,1]=0
cv2.imshow("source g==0",img1)
img2[:,:,2]=0
cv2.imshow("source r==0",img2)







cv2.waitKey()
cv2.destroyAllWindows()
for j in range(1,100000):
    print(j)
