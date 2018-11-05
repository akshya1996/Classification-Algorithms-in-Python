import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import scipy as sp
'''
Output of pickdataclass function is tempfile.out
Input of splitdata2testtrain function is tempfile.out 
'''
outFile = 'tempfile.out'
SVM = svm.SVC()

'''Classification of data using SVM, centroid, knn and linear method '''


def taskA():
    '''The input string is given to letter_2_digit_convert function to get class labels
    '''
    print('Enter a input string:')
    str1 = str(input())
    '''removes duplicate characters from the input string'''
    pickDataClass('HandWrittenLetters.txt', letter_2_digit_convert(str1))

    testVector, testLabel, trainVector, trainLabel = splitData2TestTrain(outFile, 39, '30:39')
    finallinear = linear(trainVector, testVector, trainLabel, testLabel)
    '''For 5-nn it is 5'''
    finalknn = kNearestNeighbor(trainVector, trainLabel, testVector, 3)
    finalcosknn= kNearestNeighborusingcosine(trainVector, trainLabel, testVector, 3)
    finalcentroid = centroid(trainVector, trainLabel, testVector, testLabel)
    svmMatrix = svmClassifier(trainVector.transpose(), trainLabel, testVector.transpose(), testLabel)
    '''finalsvm stores the accuracy score of the svm model'''
    finalsvm = SVM.score(svmMatrix, testLabel)*100
    print("Final Accuracy for Task A:")
    print('\nSVM : %f \n' % finalsvm)
    '''comparing the accuracy  between testLabel and finalcentroid'''
    err_test_padding = testLabel - finalcentroid
    TestingAccuracy_padding = (1 - np.nonzero(err_test_padding)[0].size / float(len(err_test_padding))) * 100
    print('Centroid : %f\n' % TestingAccuracy_padding)
    centroid_accuracy=TestingAccuracy_padding
    print('Linear : %f\n' % finallinear)
    '''comparing the accuracy between testLabel and finalknn'''
    err_test_padding = testLabel - finalknn
    TestingAccuracy_padding = (1 - np.nonzero(err_test_padding)[0].size / float(len(err_test_padding))) * 100
    print('KNN (K=3) using euclidean distance: %f\n' % TestingAccuracy_padding)
    knn_accuracy=TestingAccuracy_padding
    '''for knn using cos'''
    err_test_padding = testLabel - finalcosknn
    TestingAccuracy_padding = (1 - np.nonzero(err_test_padding)[0].size / float(len(err_test_padding))) * 100
    print('KNN (K=3) using cos distance: %f\n' % TestingAccuracy_padding)
    knn_cos_accuracy = TestingAccuracy_padding

    with open('result.txt','w') as fn:
        fn.write("Input:")
        fn.write(str1)
        fn.write("\n")
        fn.write("SVM:")
        fn.write(str(finalsvm))
        fn.write("\n")
        fn.write("CENTROID:")
        fn.write(str(centroid_accuracy))
        fn.write("\n")
        fn.write("LINEAR:")
        fn.write(str(finallinear))
        fn.write("\n")
        fn.write("KNN using euclidean:")
        fn.write(str(knn_accuracy))
        fn.write("\n")
        fn.write("KNN using cos:")
        fn.write(str(knn_cos_accuracy))
        fn.write("\n")





def pickDataClass(filename, class_ids):
    nd_data1 = np.genfromtxt(filename, delimiter=',')
    classids_col = []
    for i in class_ids:
        a = np.where(nd_data1[0] == i)
        classids_col.extend(np.array(a).tolist())
    classids_col = [j for k in classids_col for j in k]
    np.savetxt(outFile, nd_data1[:, classids_col], fmt="%i", delimiter=',')





def letter_2_digit_convert(input_string):
    l2d_list = []
    d2l_list=[]
    input_string = input_string.upper()
    for i in input_string:
        if i.isalpha():
            n=ord(i)-64
            while l2d_list.__contains__(n):
                n=n+1
            #l2d_list.append(ord(i) - 64)
            l2d_list.append(n)
    for i in l2d_list:
        d2l_list.append(i+64)
    str1=""
    for i in d2l_list:
        chr(i)
        str1=str1+chr(i)
    with open('transformedinput.txt', 'w') as f:
        print(str1)
        f.write(str1)
        f.write('\n')
        f.write('Selected classes: ')
        print('Selected classes: ')
        for i in l2d_list:
            print(i)
            f.write(str(i))
            f.write('\n')
    return l2d_list



def splitData2TestTrain(filename, number_per_class, test_instances):
    first_classid, last_classid = test_instances.split(":")
    nd_data2 = np.genfromtxt(filename, delimiter=',')
    train_data = []
    test_data = []
    test_classids = list(range(int(first_classid), int(last_classid)))
    train_classids = list((set(list(range(0, number_per_class))) - set(test_classids)))
    for i in range(0, nd_data2[0].size, number_per_class):
        train_list = [id + i for id in train_classids]
        train_list.sort()
        test_list = [id + i for id in test_classids]
        test_list.sort()
        if len(train_data) == 0:
            train_data = nd_data2[:, train_list]
        else:
            train_data = np.concatenate((train_data, nd_data2[:, train_list]), axis=1)
        if len(test_data) == 0:
            test_data = nd_data2[:, test_list]
        else:
            test_data = np.concatenate((test_data, nd_data2[:, test_list]), axis=1)
            '''stores the training data in trainData.txt and testing data in testdata.txt'''
    store(train_data[0], train_data[1:, ], 'trainData.txt')
    store(test_data[0], test_data[1:, ], 'testData.txt')
    return test_data[1:, ], test_data[0], train_data[1:, ], train_data[0]


'''This function is called from splitData2testntrain function
The classids and the data of test and train are passed
It takes these data and stores them in testData.txt and trainData.txt respectively
'''


def store(class_ids, filedata, fileName):
    np.savetxt(fileName, np.vstack((class_ids, filedata)), fmt="%i")


'''
Implementing svm using scikit library
'''


def svmClassifier(train_data, trainLabel, test_data, testLabel):

    SVM.fit(train_data, trainLabel)
    SVM.predict(test_data)
    return test_data


'''
Implentation of centroid method using eucledian distane
'''


def centroid(trainVector, trainLabel, testVector, testLabel):
    result = []
    mean_list = []

    for j in range(0, len(trainVector[0]), 8):
        colavg = [trainLabel[j]]
        for i in range(len(trainVector)):
            colavg.append(np.mean(trainVector[i, j:j + 7]))
        if not len(mean_list):
            mean_list = np.vstack(colavg)
        else:
            mean_list = np.hstack((mean_list, (np.vstack(colavg))))

    for l in range(len(testVector[0])):
        linear_dist = []
        for n in range(len(mean_list[0])):
            euclid_dist = np.sqrt(np.sum(np.square(testVector[:, l] - mean_list[1:, n])))
            linear_dist.append([euclid_dist, int(mean_list[0, n])])
            linear_dist = sorted(linear_dist, key=lambda linear_dist: linear_dist[0])
        result.append(linear_dist[0][1])
    return result




'''The following is a python code for Linear regression '''

def linear(Xtrain, Xtest, Ytrain, Ytest):
    counter = 0
    N_train=len(Xtrain[0])
    N_test=len(Xtest[0])
    A_train = np.ones((1, N_train))
    A_test = np.ones((1, N_test))
    Xtrain_padding = np.row_stack((Xtrain, A_train))
    Xtest_padding = np.row_stack((Xtest, A_test))
    #Ytrain is the indicator matrix
    element, index, count = np.unique(Ytrain, return_counts=True, return_index=True)
    element = Ytrain[np.sort(index)]
    Ytrain_Indent = np.zeros((int(max(element)), count[0] * len(element)))
    for i, j in zip(count, element):
        Ytrain_Indent[int(j) - 1, counter * i:counter * i + i] = np.ones(i)
        counter += 1
        '''computing regression coefficients'''
    #Formula: (XX')^(-1) X * Y' to find beta value
    B_padding = np.dot(np.linalg.pinv(Xtrain_padding.T), Ytrain_Indent.T)
    Ytest_padding = np.dot(B_padding.T, Xtest_padding)
    Ytest_padding_argmax = np.argmax(Ytest_padding, axis=0) + 1
    '''comparing the accuracy'''
    err_test_padding = Ytest - Ytest_padding_argmax
    TestingAccuracy_padding = (1 - np.nonzero(err_test_padding)[0].size / float(len(err_test_padding))) * 100
    return TestingAccuracy_padding


'''
This function is used to implement k nearest neighbour using euclidean distance


'''


def kNearestNeighbor(trainvector, trainlabel, testvector, k):
    result = []
    N_test = len(testvector[0])
    N_train = len(trainvector[0])
    for i in range(N_test):
        points = []
        distances_array = []
        test_curr = testvector[:, i]
        for j in range(N_train):
            dist = np.sqrt(np.sum(np.square(test_curr - trainvector[:, j])))
            distances_array.append([dist, j])
            distances_array = sorted(distances_array)
        for j in range(k):
            index = distances_array[j][1]
            points.append(trainlabel[index])
        result.append(max(set(points), key=points.count))

    result = list(int(i) for i in result)
    return result

def kNearestNeighborusingcosine(trainvector, trainlabel, testvector, k):
    result = []
    N_test = len(testvector[0])
    N_train = len(trainvector[0])
    for i in range(N_test):
        points = []
        distances_array = []
        test_curr = testvector[:, i]
        for j in range(N_train):
            #dist = np.matmul(test_curr, trainvector[:, j])/(test_curr.__sizeof__())*(trainvector[:, j].__sizeof__())
            dist=sp.spatial.distance.cosine(test_curr, trainvector[:, j])
            distances_array.append([dist, j])
            distances_array = sorted(distances_array)
        for j in range(k):
            index = distances_array[j][1]
            points.append(trainlabel[index])
        result.append(max(set(points), key=points.count))

    result = list(int(i) for i in result)
    return result

taskA()


