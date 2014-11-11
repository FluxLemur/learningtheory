'''
Voted Perceptron on Digit Prediction
Leo Mehr

The voted perceptron algorithm produces a binary classifier for a pair of digits.
'''

import string
import numpy as np
from numpy.random import shuffle
from matplotlib import pyplot as plt

def displayDigit(line):
    ''' Input: length 256 array of values [0, 255]
        Output: show image as 16 x 16 matrix '''
    plt.imshow(np.transpose(np.reshape(line, (16, 16))), cmap='Greys', interpolation='gaussian')
    plt.show()

def runPercep(trainS, d1, d2, display=False, ws_i = [], T=-1):
    f = lambda x: 1 if x==d1 else -1
    f_inv = lambda x: d1 if x==1 else d2
    ws = ws_i if len(ws_i) > 0 else [np.zeros(256)]
    errors = 0
    for ([yt], xt) in trainS:
        p = predict(xt, ws, T)
        if display:
            plt.title('Prediction: ' + str(f_inv(p)))
            displayDigit(xt)
        if p == f(yt): # correct prediction
            ws.append(ws[-1])
        else:   # incorrect
            ws.append(ws[-1] + f(yt)*xt)
            errors += 1
    return (ws, errors)

def predict(xt, ws, T = -1):
    s = 0
    i = 0 if T == -1 else max(len(ws) - T, 0)
    while i < len(ws):
        wi = ws[i]
        i += 1
        s += np.sign(np.dot(wi, xt))
    if s >= 0:
        return 1
    else:
        return -1

def getData(d1, d2, trainSize=1, testSize=1):
    ''' Fetches data for two digits, d1, d2
        Returns '''
    def labelCol(size, label):
        return np.transpose(np.column_stack(np.ones(size) * int(label)))
    pathToData = './data/'
    l1 = labelCol(trainSize, int(d1))
    l1t = labelCol(testSize, int(d1))
    l2 = labelCol(trainSize, int(d2))
    l2t = labelCol(testSize, int(d2))

    data1 = np.loadtxt(pathToData+'digit'+str(d1)+'.txt')
    data2 = np.loadtxt(pathToData+'digit'+str(d2)+'.txt')
    shuffle(data1)
    shuffle(data2)

    train = np.append(zip(l1, data1[0:trainSize]), zip(l2, data2[0:trainSize]), axis=0)
    test = np.append(zip(l1t, data1[testSize:]), zip(l2t, data2[testSize:]), axis=0)
    return (train, test)

def displayDigits(data):
    for ([l], x) in dat:
        displayDigit(x)
        print l

def plots():
    trainSize = 900
    testSize  = 200
    Tvals = [1, 10, 100]

    for Tval in Tvals:
        err_percents = []
        for i in xrange(0, 10):
            err_i = [0] * (i+1)
            for j in xrange(i+1, 10):
                print (i, j)
                (train, test) = getData(i, j, trainSize, testSize)
                shuffle(train)
                shuffle(test)
                (ws, _) = runPercep(train, i, j, T=Tval)
                (_, error) = runPercep(test, i, j, False, ws_i=ws, T=Tval)
                err_i.append(error / (testSize * 2.))
            err_percents.append(err_i)

        err_percents = err_percents + np.transpose(err_percents)
        plt.title('Percent Error for Digit Pairs T=' + str(Tval))
        plt.imshow(err_percents, interpolation='none', cmap='Greys')
        plt.colorbar()
        plt.savefig('images/T'+str(Tval)+'.png')
        plt.clf()

def errors():
    trainSize = 800
    testSize  = 300
    Tvals = [10, 25, 50, 75, 100, 200]
    err_percents = []
    i = 3
    j = 2
    for Tval in Tvals:
        (train, test) = getData(i, j, trainSize, testSize)
        shuffle(train)
        shuffle(test)
        (ws, _) = runPercep(train, i, j, T=Tval)
        (_, error) = runPercep(test, i, j, True, ws_i=ws, T=Tval)
        err_percents.append((Tval, error / (testSize * 2.)))
    plt.title('Percent Error for 1, 2 T=' + str(Tval))
    err_arr = np.array(err_percents)
    plt.plot(err_arr[:,:1], err_arr[:,1:], 'bo-')
    plt.savefig('images/err12.png')
    plt.clf()

errors()
