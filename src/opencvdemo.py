import cv2
jpeg1=cv2.imread("/Users/apple/Downloads/1.jpg")
jpeg2=cv2.imread("/Users/apple/Downloads/1copy.jpg")

#cv2.imwrite("/Users/apple/Downloads/1copy.jpg",jpeg1)


a=cv2.namedWindow("kenyopencv-demo1")
# # b=cv2.namedWindow("kenyopencv-demo2")
cv2.imshow("dacha opencv first demo1", jpeg1)
cv2.imshow("dacha opencv copy  demo2", jpeg2)
#cv2.imshow(b, jpeg1)
key = cv2.waitKey()
if key!=-1:
    print("按任意键退出窗口")

# cv2.imwrite("/Users/apple/Downloads/1.bmp",jpeg1)
# cv2.destroyAllWindows()
#     print("按任意键退出窗口demo1和demo2")
# #
# if key==ord("A"):
#     print("A")
#     cv2.imshow(a,jpeg1)
# # elif key==ord("B"):
# #     print("B")
# #     cv2.imshow(a, jpeg1)
#
#
for i in range(1,10000000):
    print(i)
