# Implementation of Voted Perceptron
Again, using python and matplotlib.

## Overview
The algorithm runs on a set of training data that consists of some labelled examples
of digits d1 and d2. The test set another labelled set of the same digit pair, d1 and d2.
The algorithm performs online learning to create a linear threshold function in `[0,255]^256`,
the space of digit representation. However, because the concept target may not be in the
hypothesis class of linear threshold functions, we introduce voting. That is to say,
we may not have a realisable case for the regular perceptron algorithm. Voting
involves storing the previous classification vector `w_i` for all `i=1...t` such that
at round `t+1`, to classify `x` we take the majority vote of all `w_i dot x`.

This approach on a high level smooths out the instability in regular perceptron if there
is no concept to converge to. Also, because taking every classification vector from
the very start of the algorithm can be very costly, we cap voting to the last `T`
classification vectors.

## Predicting
The following examples are extracted from the algorithm in the instance of a training set
of 1600 digits, either 2 or 3, for `T = 10`.
![Predict 2](https://github.com/FluxLemur/learningtheory/blob/master/vpercep/images/predict2_2.png)
![Predict 3](https://github.com/FluxLemur/learningtheory/blob/master/vpercep/images/predict2_3.png)

## Visualizing Error
Below is a heat map of the percent error for when the algorithm ran on different pairs of digits.
The training set was 1800 digits and the test set 400, taking T=10.
![Heat Map](https://github.com/FluxLemur/learningtheory/blob/master/vpercep/images/T10.png)
