# Implementation of [K Nearest Neighbors](http://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
Using python! (Images with [matplotlib](http://matplotlib.org/))

## Visualizing Performance
Error is calculated as the number of incorrect labelings, using k-nn conditioned on
the training set, compared to the actual labelings of the test set.
![Percent Error vs. K Value](https://github.com/FluxLemur/learningtheory/blob/master/knn/images/error_log.png)
Very small k (< 10) classification is prone to error due to noise in the training set.
![1 Closest Neighbor](https://github.com/FluxLemur/learningtheory/blob/master/knn/images/region_1.png)
![5 Closest Neighbors](https://github.com/FluxLemur/learningtheory/blob/master/knn/images/region_5.png)

Because the noise and number of total points are small, a small increase in k smooths classification quickly.
![10 Closest Neighbors](https://github.com/FluxLemur/learningtheory/blob/master/knn/images/region_10.png)

The classification regions continue to smooth, and reach an optimum for k ~ 30.
![25 Closest Neighbors](https://github.com/FluxLemur/learningtheory/blob/master/knn/images/region_25.png)

Whether more or less definition is useful depends on distribution of the
concept target, however, it is clear that using very large n (1/5 of the
training set, > 100) does not improve results here.
![100 Closest Neighbors](https://github.com/FluxLemur/learningtheory/blob/master/knn/images/region_100.png)
![250 Closest Neighbors](https://github.com/FluxLemur/learningtheory/blob/master/knn/images/region_250.png)

Very large k values quickly converge to the majority labeling.
![400 Closest Neighbors](https://github.com/FluxLemur/learningtheory/blob/master/knn/images/region_400.png)
![483 Closest Neighbors](https://github.com/FluxLemur/learningtheory/blob/master/knn/images/region_483.png)

The expected error converges to not more than 0.5 taking the global majority, with k = |Sample Set|.
![500 Closest Neighbors](https://github.com/FluxLemur/learningtheory/blob/master/knn/images/region_500.png)
