import cv2
import numpy as np
img=cv2.imread("/Users/apple/Downloads/1.jpg")
face= np.random.randint(20,256,(200,200,3))
img[100:300,100:300]=face  # 200*200
cv2.imshow("result",img)
cv2.waitKey()
cv2.destroyAllWindows()
for j in range(1,100000):
    print(j)
