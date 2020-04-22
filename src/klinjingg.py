import cv2
import numpy as np
import matplotlib.pyplot as plt
# 用于训练的数据
# train1数据位于(0,35) 训练数据0-35的数据
train1 = np.random.randint(0, 35, (50, 2)).astype(np.float32)
# train2数据位于(65,100)

train2 = np.random.randint(65, 100, (50, 2)).astype(np.float32)
#训练集叠加
trainData = np.vstack((train2, train1))


x1=np.zeros((50,1)).astype(np.float32)

y1=np.ones((50,1)).astype(np.float32)



tdLable = np.vstack((x1, y1))
# 使用绿色标注类型0
g = trainData[tdLable.ravel() == 0]
plt.scatter(g[:,0], g[:,1], 33, 'g', 'h')
# 使用hua色标注类型1)

y = trainData[tdLable.ravel() == 1]
plt.scatter(y[:,0], y[:,1], 33, 'y', 'o')
# plt.show()
# test为用于测试的随机数，该数在0到100之间
test = np.random.randint(0, 100, (1, 2)).astype(np.float32)
plt.scatter(test[:,1], test[:,0], 80, 'r', 'X')
# 调用OpenCV内的K近邻模块，并进行训练

knn = cv2.ml.KNearest_create()
print(trainData)
print(tdLable)

knn.train(trainData, cv2.ml.ROW_SAMPLE, tdLable)
# 使用K近邻算法分类
ret, results, neighbours, dist = knn.findNearest(test, 10)
# 显示处理结果
print("随机数判定为类型：", results)
print("距离当前点最近的10个邻居是：", neighbours)
print("10个最近邻居的距离： ", dist)
# 可以观察一下显示，对比上述输出
plt.show()

