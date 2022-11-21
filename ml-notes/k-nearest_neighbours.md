# K- Nearest Neigbours

Points to Note:

1. No training phase,called lazy learning Method.
2. Inference time proportional to the size of training data.
3. Finds the most similar training examples to predict its class.
4. Voronoi Cells
   1. Partition the spaces into regions
   2. boundary: points at same distance from two different examples.
5. Non linear Decision Boundary
6. Ways to improve the Sensitivity to Outliers
   1. Use more than one nearest neighbour to make decision
   2. count class labels in k most similar training examples
7. Algorithm
   1. Compute Distance between every pair of training examples.
   2. Select k closest instances and their labels.
   3. Output the class which is most frequent in case of classification task or output the mean of the k nearest instance target label.
8. Predicting on interpolated examples are comparitively easier than prediciting on extrapolated examples.
9. Value of k has strong effect on knn performance
   1. large value - everything classified as the most probable class.
   2. Small value - highly variable,unstable decision boundary,small changes to training set,large changes in classification.
   3. Use Validation set for selecting the value of k
10. Parameters
    1. Distance Measure
11.  Resolving Ties
    1. For binary case, use odd k
    2. For multi class ,break ties by any of the following strategies:
       1. Random: flip a coin to decide positive/negative
       2. prior - pick class with greater priors.
       3. Nearest: use 1nn classifier to decider
12. Handling missing value is a must otherwise distance cannot be calculated.
    1. Use missing value strategy which should affect distance as little as possible.
13. Parzen Windows Approach for Binary Classification
    1. Instead of looking at a fixed k nearest neighbour ,use region with fixed area or volume in space.
    2. Assign label categories as +1 or -1.
    3. The prediction is based on the sign of the summation of the labels in the fixed radius.
14. Pros of KNN
    1. Non parametric approach.No assumptions about the data. 
15. Cons of KNN
    1. Need to handle missing data
    2. Sensitive to class outliers(mislabeled training instances)
    3. Sensitive to lots of irrelevant attributes (affect distance)
    4. Computationally expensive
       1. Space- need to store all training examples O(n)
       2. time - as number of training example grows,system will become slower.
       3. n= numbe of examples,d - dimensionality
          1. Training time Complexity - O(d)
          2. Testing time complexity O(nd) 
       4. expense is at testing time and not training time
16. Making knn fast
    1. Reduce d: - Dimensionality reduction
    2. Reduce n : don't compare to all training examples
       1. use only m instances where m<<n .Hence TestTime Complexity = O(md)
       2. K-D trees used for low dimensional(12-30), continuous real valued data(not sparse)
          1. O(d log2 n) only works when d <<n 	, can miss neighbors
       3. Inverted Lists: high dimensional ,discrete(sparse) data 
          1. O(n'd) where d'<<d , n'<<n only for sparse data(eg:text)
       4. Locality-sensitive hashing : high d, real valued or discrete 
          1. O(n'd) , n'<<n bits in fingerprint, can miss neighbors.
17. K-D Tree
    1. Pick random dimension,find median,split data, repeat
    2. Builds tree with binary split.
    3. Test Time Complexity - O(dlogn)
18. Inverted List
19. LSH
    1. Random hyper-planes h1...hk
       1. space sliced into 2^k regions
    2. At testing time , compare the x only to training points in the same region.
    3. Test Time Complexity - O(kd+dn/2^k) ~~ O(dlogn)
       1. O(kd) - to find region R, k<<n 
          1. Dot-product x with h1..hk and the point x is d dimensional.
       2. O(dn/2^k) - Compare to n/2^k points in R
          1. dot product x with each point in the region .
          2. Number of points in any given region - n/2^k(as there are 2^k regions containing total n points.)
    4. Can miss neigbours.
       1. Handle it by repeating the whole procedure L different times with different set of hyperplanes. 
20. Add
    1. 