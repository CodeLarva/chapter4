from numpy import *


def loadDataSet():
    # 试样样本
    postingList = [['my', 'dog', 'has', 'flea', \
                    'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', \
                    'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', \
                    'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', \
                    'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # i is abusive,0 not     类别标签的集合
    return postingList, classVec


"""获得词表，词表中没有重复的词"""


def createVocabList(dataSet):
    vocabSet = set([])
    for doucument in dataSet:
        vocabSet = vocabSet | set(doucument)  # 创建两个集合的并集
    return list(vocabSet)


"""vocabList为词汇表，inputSet为某个文档"""
#文档词模型
def setOfWords2Vec(vocabList, inputSet):
    # 转化成以一维数组
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

#词袋模型
def bagOfWords2Vec(vocabList, inputSet):
    # 转化成以一维数组
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


"""计算，生成每个词对于类别上的概率"""


def trainNB0(trainMatrix, trainCategory):
    # 类别行数
    numTrainDocs = len(trainMatrix)
    # 列数
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)

    """
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0
    """
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    """
    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom
    """
    # 计算每种类型里面， 每个单词出现的概率
    # 朴素贝叶斯分类中，y=x是单调递增函数，y=ln(x)也是单调的递增的
    # 如果x1>x2 那么ln(x1)>ln(x2)
    # 在计算过程中，由于概率的值较小，所以我们就取对数进行比较，根据对数的特性
    # ln(MN) = ln(M)+ln(N)
    # ln(M/N) = ln(M)-ln(N)
    # ln(M**n)= nln(M)
    # 注：其中ln可替换为log的任意对数底
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # sum是numpy的函数，vec2Classify是一个数组向量，p1Vec是一个1的概率向量，通过矩阵之间的乘积获得p(X1|Yj)*p(X2|Yj)*...*p(Xn|Yj)*p(Yj)
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    print(p1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    print(p0)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    # 朴素贝叶斯分类, max(p0， p1)作为推断的分类
    # y=x 是单调递增的， y=ln(x)也是单调递增的。 ， 如果x1 > x2, 那么ln(x1) > ln(x2)
    # 因为概率的值太小了，所以我们可以取ln， 根据对数特性ln(ab) = lna + lnb， 可以简化计算
    # sum是numpy的函数，testVec是一个数组向量，p1Vec是一个1的概率向量，通过矩阵之间的乘机
    # 获得p(X1|Yj)*p(X2|Yj)*...*p(Xn|Yj)*p(Yj)
    # 其中pClass1即为p(Yj)
    # 此处计算出的p1是用对数表示，按照上面所说的，对数也是单调的，而贝叶斯分类主要是通过比较概率
    # 出现的大小，不需要确切的概率数据，因此下述表述完全正确
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    # 构建词向量矩阵
    # 计算listOPosts数据集中每一行每个单词出现的次数，其中返回的trainMat是一个数组的数组
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    # 测试数据集
    testEntry = ['love', 'my', 'dalmation']
    # 转换成单词向量，32个单词构成的数组，如果此单词在数组中，数组的项值置1
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    # 通过将单词向量testVec代入，根据贝叶斯公式，比较各个类别的后验概率，判断当前数据的分类情况
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

if __name__ == '__main__':
    testingNB()
