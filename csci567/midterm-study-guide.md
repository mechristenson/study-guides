# CSCI 567: Machine Learning
## Midterm Study Guide

### Details:
  - Wednesday February 27th, 2019
  - 17:00 - 20:00
  - THH 101 & 201

### Major Topics:
  - KNN
  - Decision Tree
  - Naive Bayes
  - Linear Regression
  - Perceptron
  - Logistic Regression
  - Multiclass Classification
  - Neural Networks
  - Convolutional Neural Networks
  - Kernel Methods
  - Clustering

---
## Machine Learning Overview

### Definitions
1. Machine Learning
  - A set of methods that can automatically detect patterns in data, and then
  use the uncovered patterns to predict future data, or to perform other kinds
  of decision making under uncertainty - Murphy
  - A computer program learns if its performance at tasks in T, as measured by
  P, improves with experience E - Mitchell

### Key Ingredients of Machine Learning
1. Data
  - Collected from past observation
  - Often called training data
2. Models
  - Devised to capture patterns in the data
  - Does not have to be true as long as it provides useful insights
3. Predictions
  - Devised by applying a model to forecast what is going to happen

### Types of Machine Learning
1. Supervised Learning
  - Aim to predict
  - Trained on labeled set of inputs
2. Unsupervised Learning
  - Aim to learn patterns
  - No feedback from environment to indicate if outputs are correct
3. Reinforcement Learning
  - Aim to act optimally under uncertainty

### General Setup for Multiclass Classification
  - Training Data
    - N samples/instances: D^TRAIN = {(x1, y1), (x2, y2),...,(xN, yN)}
    - Each xn in R^D is called a feature vector
    - Each yn in [C] = {1, 2,..., C} is called a label/class/category
    - Our goal is to learn a function f: that maps the values of R^D to C for
    future prediction. Given an unseen observation, x, f(x), can confidently
    predict the corresponding output y.
  - Special Case: Binary Classification
    - Number of classes, C = 2
    - Conventional Labels, {0,1} or {-1,1}

### Measuring Performance
  - We can measure performance by computing:
    - Accuracy (A) - percentage of data points being correctly classified
    - Error Rate (ε) - percentage of data points being incorrectly classified

### Hyperparameters
  - Most algorithms have hyperparameters
  - Tuning hyperparameters is usually the most time-consuming part of applying
  an algorithm
  - Process for Tuning Hyperparameters:
    - Split data into three separate sets:
      - Training Data:
        - N samples/instances, DTrain = {(x1,y1),(x2,y2)...(xn,yn)}
        - Used for Learning f(.)
      - Test Data:
        - M samples/instances, DTest = {(x1,y1),(x2,y2)...(xm,ym)}
        - Used for assessing how well our model f(.) will do
      - Development/Validation Data
        - L samples/instances, DDev = {(x1,y1),(x2,y2)...(xl,yl)}
        - Used for tuning our hyperparameters
    - For each possible value of our hyperparameter:
      - Train a model using DTrain
      - Evaluate the performance of the model on DDev
    - Choose the model with the best performance on DDev
    - Evaluate the model on DTest
  - If we do not have a development set, we can use S-Fold Cross Validation:
    - Split the training data into S equal parts
    - Run the algorithm S times
    - For each run take the "folded" block as a development dataset and use the
    others as a training dataset
    - Choose the hyperparameter with the best average performance
    - Special case: S=N, called leave-one-out

### Typical Steps of Developing a Machine Learning System
  - Collect data, split into training, development and test sets
  - Train a model with a machine learning algorithm. Most often, we apply
  cross-validation to tune hyper parameters
  - Evaluate using the test data and report performance
  - Use the model to predict future/make decisions

### Classification vs Regression
  - Classification is a problem of finding a mapping function f from training
  data (x in R^D) to discrete output variables (y in C)
    - The output variables are called labels or classes or categories
    - The mapping function predicts the class for a given observation
    - The classification accuracy is computed as the percentage of correctly
    classified examples out of all examples
  - Regression is a problem of finding a mapping function f from training data
  (x in R^D) to a continuous output variable (y in R)
    - The output variable is a continuous quantity
    - Regression predictions can be evaluated using the mean squared error

#### Problems
  - Discussion 1 - Problem 4
  - Discussion 1 - Problem 5
  - Discussion 1 - Problem 6

---

## Models

### Nearest Neighbor Classifier

#### Overview
  - The label of a datapoint equals the label of the nearest training point
  - We calculate the label by finding the index of the point with the minimum
  distance from our test data, then we check the label at that instance

#### Key Points
  - Non-parametric
  - Supervised
  - For every point in the space, we can determine its label using the NNC rule.
  This gives rise to a decision boundary that partitions the space into
  different regions
  - Simple, easily implemented, widely used
  - Computationally intensive for large-scale problems O(ND) for each prediction
  naively
  - Need to carry the training data around
  - Choosing the right hyperparameters can be laborious
  - When calculating label for a data point, don't consider other test points

#### Measuring Performance
  - We can measure performance by calculating the number of points correctly (or
  incorrectly) classified divided by the total number of points
  - One point to note, if we measure performance on the training set, we will
  always hit 100% accuracy. This is because for every training point, its
  nearest neighbor is itself.
  - To measure performance on the training set, we can use a Leave-One-Out
  algorithm.
    - For each training instance, xn, take it out of the training set and then
    label it.
    - For NNC, xn's nearest neighbor will not be itself, so the error rate will
    not necessarily be zero.

#### Hyper Parameters (Variants)
  - Distance Functions
    - Most commonly, we use Euclidean or L2 distance
    - We can also use Manhattan or L1 distance or many other distance functions
  - KNN
    - We can also increase the number of nearest neighbors that we use
    - Typically we will take the majority of the K neighbors to determine the
    label, if there is a tie, we can also have a tie-breaker hyperparameter
    - Typically, we will use an odd number for K, to help avoid ties
    - We may also use weights for our K's
    - As K increases, our decision boundary becomes smoother
    - When K = N, we let the majority decide the value of all test points
  - Preprocessing Data
    - One issue we run into with NNC is that distances depend on the units of
    our features
    - We can preprocess our data so it looks more normalized
    - One example is to scale our feature by subtracting the average value of
    each feature from each feature and then dividing by the standard deviation.
    This shapes our data into a Gaussian Distribution

#### Practice Problems
  - Discussion 1 - Problem 1
  - Discussion 1 - Problem 2
  - Discussion 1 - Problem 3

### Decision Tree

#### Overview
  - Simple tree-shaped if-then model
  - We build our tree where each node of the tree tests a specific attribute,
  each branch from a node corresponds to one of the possible values for that
  attribute.
  - We calculate the label by traversing our tree model with our data point
  until we reach a leaf. Then we assign the label (or a probability of the label
  to our data point.)

#### Key Points
  - Parametric
  - Supervised
  - Decision trees represent disjunctions of conjunctions
  - The decision tree corresponds to a classifier with boundaries
  - It is too computationally expensive to calculate the optimal tree structure,
  (if we have Z nodes, number_of_features^Z is the number of all possible
  configurations). Instead we can use a greedy top-down approach to learning
  the parameters

#### Learning a Decision Tree
  - Parameters:
    - Structure of the tree
      - depth, number of branches, number of nodes, etc.
    - Test at each node
      - Which features to test?
      - What threshold for continuous features?
    - The value or prediction of the leaves
  - Methods:
    - As we mentioned, a greedy approach is best.
    - At each node we should choose the attribute with the most "pure" children,
    i.e. the children with highest probabilities of a label
    - Entropy is a non-negative measure of disorder that we can use to calculate
    purity. Note, we want to minimize Entropy.

#### ID3 Algorithm
```
DecisionTreeLearning(Examples, Features):
  if Examples have the same class
    return a leaf with this class
  else if Features is empty
    return a leaf with the majority class
  else if Examples is empty
    return a leaf with the majority class of parent
  else
    find the best feature, A, to split based on conditional entropy
    Tree <- a root with test on A
    For each value a of A:
      Child <- DecisionTreeLearning(Examples with A=a, Features-{A})
      add Child to Tree as a new branch
  return Tree
```

#### Hyperparameters (Variants)
  - Replace entropy by the Gini Impurity

#### Overfitting
  - We can run into overfitting here for a number of reasons
    - large numbers of attributes
    - too little training data
    - Many kinds of "noise" (same feature but different classes, values of
    attributes are incorrect, classes are incorrect)
  - We can avoid overfitting in the following ways
    - stop growing when you reach some depth or number of nodes
    - stop growing when the data split is not statistically significant
    - Acquire more training data
    - remove irrelevant attributes
    - grow a full tree and then prune it

#### Pruning
  - To avoid overfitting we can try to use pruning.
  - Pruning is done by replacing a whole subtree by a leaf node and assigning
  the most common class to that node.
    - Split data into training and validation sets
    - Grow a full tree based on training set
    - Do pruning until its harmful
      - Evaluate the impact on the validations set of pruning each possible node
      - Greedily remove the node that most improves validation set accuracy
    - Do not consider leaves when pruning

#### Practice Problems
  - Discussion 2 Problem 1
  - Discussion 2 Problem 2
  - Discussion 2 Problem 3

### Naive Bayes

#### Overview
  - Calculating the Bayes Optimal Classification for a dataset is, as its name
  suggests, the optimal classification for a dataset. However, it is
  exponentially hard to learn the optimal classifier. If we "naively" assume
  that all xd are conditionally independent, given y, we can reduce the
  complexity to linear.
  - This model is unrealistic and naive, but Naive Bayes still works very
  well, even if the assumption is wrong.
  - To make a prediction, we calculate the probability of each xd given y, and
  then select the maximum value of those probabilities.

#### Practice Problems
  - Discussion 2 Problem 4
  - Discussion 2 Problem 5
  - Discussion 2 Problem 6

### Linear Regression

#### Overview
  - Regression with Linear Models
  - Aim to predict a value, y from a set of values, x, using a linear function
  - Our goal is to pick the model (unknown parameters) that minimizes the
  average/total prediction error

#### Measuring Error
  - Classification Error (0-1 loss) is inappropriate for continuous outcomes
  - We can look at absolute error, |prediction - actual|, or squared error
  (prediction - actual)^2

#### Goal of Linear Regression
  - Our goal is to minimize the total squared error
  - To do this we can minimize the euclidean distance with our Residual Sum of
  Squares (RSS) function, or find the least squares solution
  - Either way, we reduce our machine learning problem to a optimization problem

#### Case: D=0
  - When we set D=0, we only have one variable to calculate, w0, our intercept
  - Thus, f ends up as a horizontal line
  - if we use RSS, f is the average, if we use absolute error, its the median

#### Case: D=1
  - When we set D=1, our goal becomes to find stationary points, i.e. points
  with zero gradient.
  - In general, stationary points are not necessarily minimizers, but for convex
  objects, it is true.

#### General Case
  - Generally, we leverage multivariate calculus to find the stationary points
  from the RSS function
  - This has a closed form solution where the gradient of the RSS function is
  is equal to (X^TX)w-X^Ty=0, or w*=(X^TX)^-1-X^Ty, where X^TX is called the
  covariance matrix
  - Note the closed form solution applies for D=0 and D=1 as well.

#### Practice Problems
---
## Linear Algebra
