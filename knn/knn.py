'''
K-Nearest Neighbors
Leo Mehr
'''

from pylab import *
from heapq import *
import numpy as np

def plot_data(S):
    ''' Plots the data in s = (x,y,l) in S, where l is the labeling of point (x,y) '''
    for s in S:
        if s[2] > 0:
            plot(s[0], s[1], 'bo')
        else:
            plot(s[0], s[1], 'ro')

    xlabel('x')
    ylabel('y')

def knn(S,x,y,k):
    ''' return majority labelling of k-nearest neighbors to (x,y) in S '''
    nn = [] # maxheap for nearest neighbors
    for s in S:
        d = -dist(s[0], s[1], x, y) # must be (-), heapq uses minheaps
        if len(nn) < k or d > nn[0][0]:
            if len(nn) >= k:
                heappop(nn)
            heappush(nn, (d, s))
    
    # get the majority labeling
    maj = 0
    for n in nn:
        maj += n[1][2]
    
    if maj >= 0:
        return 1
    else:
        return -1

def dist(x1, y1, x2, y2):
    return sqrt((x1-x2)**2 + (y1-y2)**2)

def test_data(D, S, k, plot_on):
    ''' Using training set S, determine the number of points from labeled set D
    misclasified using k-nn '''
    num_err = 0
    for d in D:
        l_pred = knn(S, d[0], d[1], k)
        if l_pred!= d[2]:
            if plot_on:
                plot(d[0], d[1], 'y^')
            num_err += 1
    return num_err

def sweep_k(D, S):
    ''' Get k-nn errors for various k values '''
    k = range(1, 10, 1) + range(10, 100, 10) + range(100, 501, 100)
    err = []
    for ki in k:
        print ki
        e = test_data(D,S,ki, False)
        #print e
        err.append(e / 500.) 
    plot(k, err)

def decision_regions(S, k):
    ''' Color an image, representing the regions for point labeling
    given training set S for k-nn '''
    d = 0.02    # detail of grid spacing
    grid = []
    x = -1
    while x < 1:
        y = -1
        ys = []
        while y < 1:
            l = knn(S, x, y, k)
            if l > 0:
                ys.append(0.1)
            else:
                ys.append(0.7)
            y += d
        x += d
        grid.append(ys)
    imshow(transpose(grid)[::-1], interpolation='none', extent=(-1, 1, -1, 1))

trainS = np.loadtxt('data/trainxy.txt')
trainL = np.loadtxt('data/trainc.txt')
trainS = np.append(trainS, transpose([trainL]), axis=1)

testS = np.loadtxt('data/testxy.txt')
testL = np.loadtxt("data/testc.txt")
testS = np.append(testS, transpose([testL]), axis=1)

'''
# Plot training data
#plot_data(trainS)
#print test_data(testS, trainS, 50, True)
#show()
'''

'''
# Generate a plot of percent error vs. log(k) for various k values
sweep_k(testS, trainS)
xscale('log')
title('Performance of K-NN')
xlabel('K value')
ylabel('Percent Error')
show()

'''

# Display points and boundary region for given k value
k = 75
plot_data(trainS)
decision_regions(trainS, k)
title(str(k) + '-NN Boundary Regions')
show()
