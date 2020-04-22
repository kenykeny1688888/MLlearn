import cv2
import numpy as np

img=cv2.imread("/Users/apple/Downloads/1.jpg")
img1=img
img2=img
cv2.imshow("source ",img)
b,g,r =cv2.split(img)

# cv2.imshow("b pic ",b)
# cv2.imshow("g pic ",g)
# cv2.imshow("r pic ",r)
bgr=cv2.merge([b,g,r])
cv2.imshow("bgr pic ",bgr)

rgb=cv2.merge([r,g,b])
cv2.imshow("rgb pic ",rgb)

grb=cv2.merge([g,r,b])
cv2.imshow("grb pic ",grb)


cv2.waitKey()
cv2.destroyAllWindows()
for j in range(1,1000000):
    print(j)
