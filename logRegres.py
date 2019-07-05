'''
Created on July 5, 2019

@author: Terry
@email：terryluohello@qq.com
'''

import random
import numpy as np

def sigmoid(inX):
    return 1.0/(1 + exp(-inX))

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """ 随机梯度上升    
    """
    m,n = np.shape(dataMatrix)
    # 初始化系数矩阵
    weights = np.ones(n)   
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001     
            # 随机产生一个0~len()之间的值
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def classifyVector(inX, weights):
    """ 分类函数

    描述：根据回归系数和特征向量计算sigmoid的值
    INPUT：
        inX：特征向量，features
        weights：根据梯度下降或随机梯度下降得到的回归系数
    OUPUT： 
        类别标签：1/0
    """
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1
    else:
        return 0


def colicTest():
    """ 打开训练集和测试集，并对数据进行格式化处理

    INPUT：
        无
    OUPUT： 
        errorRate：分类错误率
    """    
    frTrain = open('./data/horseColicTraining.txt')
    frTest = open('./data/horseColicTest.txt')
    trainingSet = []
    trainingLabels = []

    # 解析训练数据集，获取特征向量和Labels
    # trainingSet储存训练数据集特征，trainingLabels储存训练数据集的样本对应的分类标签
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    # 使用改进后的随机梯度下降算法求得最佳回归系数trainWeights
    trainWeights = stocGradAscent1(array(trainingSet),trainingLabels,1000)
    errorCount = 0;
    numTestVec = 0.0;
    # 读取测试集，进行测试，计算分类错误的样本条数和最终得错误率
    for line in frTest.readlines():
        numTestVec += 1
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print ("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    """ 调用colicTest() 10 次并且求结果得平均值

    INPUT：
        无
    OUPUT： 
        无
    """ 
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is %f" %(numTests,errorSum/float(numTests)))

if __name__ == "__main__":
    multiTest()