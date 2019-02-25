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

#### Problems
---
## Linear Algebra
