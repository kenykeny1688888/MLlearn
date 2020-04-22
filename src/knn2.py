from numpy import *
import re

#从文件导入数据
def file2array(filename):
    '''
    :param filename: 文件名
    :return: 数据集(arr)、类别(list)
    '''
    label={
        'didntLike':0,  #不喜欢
        'smallDoses':1, #小喜欢
        'largeDoses':2  #喜欢
    }
    with open(filename) as fr:
        lines =fr.readlines()
    tempLine =re.split('\\s+',lines[0].strip())         #'\\s+'表示tab或多个空格  #strip()除去换行符
    returnArr = zeros((len(lines),len(tempLine)-1))     #初始化数组（存放数据集）
    classLabelVector = []                               #存放类别
    for index,line in enumerate(lines):
        listFromLine = re.split('\\s+',line.strip())    #空格或tab都行
        returnArr[index,:] = listFromLine[0:-1]
        classLabelVector.append(label[listFromLine[-1]])
    return returnArr,classLabelVector

#数据归一化
def Norm(dataSet):
    minVals = dataSet.min(0)    #0：列(特征)的最小值；1：行(样本)的最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(dataSet.shape)
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet /= tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals

#约会网站配对例子
def datingClassTest():
    hoRatio = 0.80      #80%作为测试集，20%为训练集
    datingDataMat,datingLabels = file2array('data.txt')   #加载数据
    normMat, ranges, minVals = Norm(datingDataMat)                              #数据归一化
    m = normMat.shape[0]    #数据集大小(样本的数目)
    print("m=",m)
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("预测类别: %d, 真实类别: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0     #统计预测错误的次数
    print ("平均错误率是: %f" % (errorCount/float(numTestVecs)))
    print("总测试数目:",numTestVecs,"总错误数目:",errorCount)

#kNN分类器
def classify0(inX, dataSet, labels, k):
    '''

    :param inX: 测试样本(arr)
    :param dataSet: 训练数据集(arr)
    :param labels: 类别(list)
    :param k:(int)
    :return: 类别
    '''
    #计算距离
    dataSetSize = dataSet.shape[0]  # 样本数量
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet #tile(inX{数组},(dataSetSize{倍数},1{竖向}))：将数组(inX)竖向(1)复制dataSetSize倍
    sqDiffMat = diffMat ** 2                        #先求平方
    sqDistances = sqDiffMat.sum(axis=1)             #再求平方和
    distances = sqDistances ** 0.5                  #开根号,欧式距离
    sortedDistIndicies = distances.argsort()  #距离从小到大排序的索引
    classCount = {}
    #print("k=",k)
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]  #用索引得到相应的类别
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    return max(classCount, key=lambda k: classCount[k])  # 返回频数最大的类别

if __name__ =='__main__':
    datingClassTest()
