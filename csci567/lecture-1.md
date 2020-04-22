# CSCI 567: Machine Learning
## Midterm Study Guide
## Lecture 1

### Major Topics:
  - Machine Learning Overview
  - Nearest Neighbor Classifier
  - Core Machine Learning Concepts
  - Steps of Developing a Machine Learning System
  - Linear Regression
  - Regression with Non-Linear Basis
  - Regularized Regression
  - Linear Discriminant Analysis
  - Perceptron
  - Logistic Regression
  - Multi-Class Classification
  - Neural Networks
  - Convolutional Neural Networks
  - Kernel Methods
  - Lagrangian Duality
  - Support Vector Machines

### Sources of Information
  - CSCI 567 Lecture Notes, Haipeng Luo [Lecture]
  - CSCI 567 Discussion Notes, Victor Adamchik [Discussion]
  - Machine Learning a Probabilistic Perspective, Kevin Murphy [MLaPP]
  - The Elements of Statistical Learning, Hastie, Tibshirani, Friedman [ESL]

---
### Lecture 1 Notes

#### Topics:
  - Machine Learning Overview
  - Nearest Neighbor Classifier
  - Core Machine Learning Concepts
  - Steps of Developing a Machine Learning System

#### Sources:
  - Lecture 1 (Week 1)
  - Discussion 1 (Week 2)
  - MLaPP 1.4.5, 1.4.7-8, 7.1-3, 7.5.1-2, 7.5.4, 7.6
  - ESL 7.1-3, 7.10

#### Definitions
  1. Machine Learning
    - A set of methods that can automatically detect patterns in data, and then use the uncovered patterns to predict future data, or to perform other kinds of decision making under uncertainty.
  2. Nearest Neighbor Classifier
    - Label each unknown data point according to the nearest known data point
  3. Sample
    - data point
  4. Feature Vector
    - n-dimensional input vector representing a data point
  5. Class
    - an output value representing the label of a data point
  6. Decision Boundary
    - NNC algorithms will naturally partition the space into different regions based on classes. The partitioning line, plane or hyperplane is called the decision boundary
  7. Accuracy
    - A measure of how correct an algorithm is
  8. Error
    - A measure of how incorrect an algorithm is

#### Concepts

##### I. Key Ingredients of Machine Learning:
  1. Data
    - Collected from past observation (often called training data)  
  2. Modeling
    - Devised to capture the patterns in the data
    - model does not have to be an absolute truth, but it should be useful
  3. Prediction
    - Apply the model to forecast what is going to happen in the future

##### II . Types of Machine Learning Problems
  1. Supervised Learning
    - Aim to predict
  2. Unsupervised Learning
    - Aim to discover hidden and latent patterns and explore data
  3. Reinforcement Learning
    - Aim to act optimally under uncertainty

##### III. Tuning an Algorithm via a Development Dataset
We should partition our data into three sets for development:
  1. Training Data:
    - N samples/instances: D<sup>TRAIN</sup> = {(x<sub>1</sub>, y<sub>1</sub>), (x<sub>2</sub>, y<sub>2</sub>),...,(x<sub>N</sub>, y<sub>N</sub>)}
    - They are used for learning f(.)
  2. Test Data:
    - N samples/instances: D<sup>TEST</sup> = {(x<sub>1</sub>, y<sub>1</sub>), (x<sub>2</sub>, y<sub>2</sub>),...,(x<sub>M</sub>, y<sub>M</sub>)}
    - They are used for assessing how well f(.) will do
  3. Validation Data:
    - N samples/instances: D<sup>VALID</sup> = {(x<sub>1</sub>, y<sub>1</sub>), (x<sub>2</sub>, y<sub>2</sub>),...,(x<sub>L</sub>, y<sub>L</sub>)}
    - They are used to optimize hyper-parameters
Note: These three sets should not overlap

For each possible value of a hyper-parameter (e.g. K = 1, 3,...):
  - Train a model using D<sup>TRAIN</sup>
  - Evaluate the performance of the model on D<sup>VALID</sup>
  - Choose the model with the best performance on D<sup>VALID</sup>
  - Evaluate the model on D<sup>TEST</sup>

##### IV. S-Fold Cross-Validation
If we don't have a development set, we can perform cross-validation to compensate for the lack of data:
  - Split the training data into S equal parts, denote each part as D<sub>s</sub><sup>TRAIN</sup>
  - For each possible value of the hyper parameter (e.g. K = 1, 2, 3...)
    - For every s ∈ [S]:
      - Train a model using D<sub>\s</sub><sup>TRAIN</sup> = D<sup>TRAIN</sup> - D<sub>s</sub><sup>TRAIN</sup>
      - Evaluate the performance of the model on D<sub>s</sub><sup>TRAIN</sup>
    - Average the S performance metrics
  - Choose the hyper parameter with the best average performance
  - Use the best hyper-parameter to train a model using all of D<sup>TRAIN</sup>
  - Evaluate the model on D<sup>TEST</sup>
Special Case:
  - S=N, called leave-one-out

##### V. Developing a Machine Learning System
  1. Collect data, split into training, development and test sets
  2. Train a model with a machine learning algorithm. Most often, apply cross-validation to tune hyper-parameters
  3. Evaluate using the test data and report performance
  4. Use the model to predict future/make decisions

##### VI. Expected Risk
Test error only makes sense when the training set and test set are correlated. Therefore, we assume that:
  - Every data point (x, y) from D<sup>TRAIN</sup>, D<sup>DEV</sup>, and D<sup>TEST</sup> is an independent and identically distributed (i.i.d) sample of an unknown joint distribution P.
    - (x,y) <sup>i.i.d</sup>~ P

Therefore, test error of a fixed classifier is a random variable, so it does not work as a certain measure of performance. Instead we can use expected error:
  - 𝔼[ε<sup>TEST</sup>] = 1/M Σ<sup>M</sup><sub>m=1</sub> 𝔼<sub>(x<sub>m</sub>, y<sub>m</sub>)~P</sub> 𝕀[f(x<sub>m</sub>) != y<sub>m</sub>] = 𝔼<sub>(x, y)~P</sub> 𝕀[f(x) != y], i.e. the expected error/mistake of f
  - Test error is a good proxy of expected error. The larger the test set, the better the approximation
  - Further note that we cannot use this formula to find training error here, because the function f, is dependent on the training data, therefore you are eliminating any randomness to the data.

###### Loss Function
More generally, expected risk can be formulated with a loss function, L(y', y):
  - O-1 Loss: L(y', y) = 𝕀[y' != y]

Expected Risk of f is:
  - R(f) = 𝔼<sub>(x, y)~P</sub> L(f(x), y)

##### VII. Bayes Optimal Classifier
To judge the usefulness of our models, we can compare our prediction to the prediction of an ideal classifier. One such classifier is the Bayes Optimal Classifier

If we know P(y|x), we can predict x using the Bayes Optimal Classifier as such:
  - f*(x) = argmax<sub>c∈[C]</sub>P(c|x)
The optimal risk is:
  - R(f*) = 𝔼<sub>x~P<sub>x</sub></sub>[1 - max<sub>c∈[C]</sub>P(c|x)], where P<sub>x</sub> is the marginal distribution of x.
  - It is easy to show R(f*) ≤ R(f) for any f.
  - For the special case where C = 2, let η(x) = P(0|x), then:
    - R(f*) = 𝔼<sub>x~P<sub>x</sub></sub>[min{η(x), 1-η(x)}]

#### Algorithms

##### I. General Setup for Multi-Class Classification

###### Training Data (Set)
  - N samples/instances: D<sup>TRAIN</sup> = {(x<sub>1</sub>, y<sub>1</sub>), (x<sub>2</sub>, y<sub>2</sub>),...,(x<sub>N</sub>, y<sub>N</sub>)}
  - Each x<sub>n</sub> ∈ ℝ<sup>D</sup> is called a feature vector
  - Each y<sub>n</sub> ∈ [C] = {1,2,...,C} is called a label/class/category
  - They are used for learning f: ℝ<sup>D</sup> → [C] for future prediction

###### Special Case: Binary Classification
  - Number of Classes: C = 2
  - Conventional labels: {0,1}, or {-1,+1}

#### II. Nearest Neighbor Classification (NNC)

##### Nearest Neighbor Overview
  - x(1)=x<sub>nn(x)</sub>, where nn(x) ∈ [N] = {1,2,...,N}, i.e. the index to one of the training instances
  - nn(x) = argmin<sub>n∈[N]</sub> ||x - x<sub>n</sub>||<sub>2</sub> = argmin<sub>n∈[N]</sub>(Σ<sup>D</sup><sub>d=1</sub> (x<sub>d</sub> - x<sub>nd</sub>)<sup>2</sup>)<sup>1/2</sup>, where ||.||<sub>2</sub> is the L2/Euclidean Distance.

##### Classification Rule
  - y = f(x) = y<sub>nn(x)</sub>, i.e. the classification of x, y, is the class of the nearest x value to x.

#### V. NNC Test Performance

##### Overview
  To test the performance of our model or algorithm, we can check the accuracy, the percentage of data points being correctly classified, or the error rate, the percentage of data points being classified.
  Note: Accuracy + Error_Rate = 1

##### Defined on the Training Set
  - Accuracy: A<sup>TRAIN</sup> = 1/N Σ<sub>n</sub> 𝕀[f(x) == y<sub>n</sub>]
  - Error: ε<sup>TRAIN</sup> = 1/N Σ<sub>n</sub> 𝕀[f(x) != y<sub>n</sub>]
  - Note: 𝕀[.] is the indicator function
  - Note: For every training data point, its nearest neighbor is itself, therefore, Training Error is not a good indicator of whether NNC is a good algorithm.

##### Test Error
  - For NNC, we need to test accuracy when predicting unseen data, therefore we need fresh test data that is not used for training. Training Algorithm cannot see the test data while its training.

##### Variants for NNC

###### Variant 1: Measure Nearness with Other Distances
Previously, we have used L2 Distance (Euclidean), we can use one of many other alternative distances such as L1 Distance (Manhattan Distance), Lp Distance, etc.

###### Variant 2: K-Nearest Neighbor (KNN)
We can increase the number of nearest neighbors to use:
  - 1-nearest neighbor: nn<sub>1</sub>(x) = argmin<sub>n∈[N]</sub>||x-x<sub>n</sub>||<sub>2</sub>
  - 2-nearest neighbor: nn<sub>2</sub>(x) = argmin<sub>n∈[N]-nn<sub>1</sub>(x)</sub>||x-x<sub>n</sub>||<sub>2</sub>
  - 3-nearest neighbor: nn<sub>3</sub>(x) = argmin<sub>n∈[N]-nn<sub>1</sub>(x)-nn<sub>2</sub>(x)</sub>||x-x<sub>n</sub>||<sub>2</sub>

The set of K-nearest neighbor:
  - knn(x) = {nn<sub>1</sub>(x),nn<sub>2</sub>(x),...,nn<sub>K</sub>(x)}

Note, for x(k) = x<sub>nn<sub>k</sub>(x)</sub>, we have:
  - ||x-x(1)||<sub>2</sub> ≤ ||x-x(2)||<sub>2</sub> ... ≤ ||x-x(K)||<sub>2</sub>

Classification Rule for KNN:
  - Every neighbor votes: naturally, x<sub>n</sub> votes for its label y<sub>n</sub>
  - Aggregate everyone's vote on a class label c:
    - v<sub>c</sub> = Σ<sub>n∈knn(x)</sub> 𝕀(y<sub>n</sub>==c), ∀ c∈[C]
  - Predict with the majority
    - y = f(x) = argmax<sub>c∈[C]</sub>v<sub>c</sub>

Note:
  - As K increases, the decision boundary becomes smoother
  - If K = N, we have a special case where the majority always wins

###### Variant 3: Preprocessing Data
One of the issues of the NNC is that distances between vectors depend on the units of features.

One Solution: preprocess data so that it looks more normalized. One of many ways to normalize your data is as follows:
  - We can compute the means and standard deviations of each feature:
    - mean: x<sub>d</sub> = 1/N Σ<sub>n</sub>x<sub>nd</sub>
    - standard deviation: s<sup>2</sup><sub>d</sub> = 1/(N-1) Σ<sub>n</sub>(x<sub>nd</sub>-x<sub>d</sub>)<sup>2</sup>
  - Then we can scale the feature accordingly:
    - x<sub>nd</sub> ← (x<sub>nd</sub> - x<sub>d</sub>) / s<sub>d</sub>
    - i.e. shift it by the mean and then divide by standard deviation, so every point has zero mean. This will cause the data to look more like a gaussian distribution

###### Choosing Variants
The different variants of an algorithm are considered "Hyper-parameters" of an algorithm. Most algorithms have hyper-parameters and tuning these hyper-parameters is a significant part of applying that algorithm.

In NNC our hyper-parameters are:
  - The distance measure (e.g. the parameter p for L<sub>p</sub> norm)
  - K (how many nearest neighbors)
  - The different ways of preprocessing

##### Summary
- Simple, easy to implement (widely used in practice)
- Computationally intensive for large-scale problems: O(ND) for each prediction naively. We need to compute distance to every training point you have. If we want to optimizing this, we make the algorithm complicated
- Need to carry the training data around, i.e. this algorithm is non-parametric, meaning memory increases as training data increases.
- NNC has lots of hyper-parameters that can be involved in optimizing this algorithm

##### Comparing NNC to Bayes Optimal Classifier
Theorem:
Let f<sub>N</sub> be the 1-nearest neighbor binary classifier using N training data points, we have:
  -  R(f*) ≤ lim<sub>n→∞</sub> 𝔼[R(f<sub>N</sub>)] ≤ 2R(f*), expected risk of NNC as our training data goes to infinity is at most twice of the best possible risk

---

### Math Review

#### Concepts

##### Vector Distances
###### L1 Norm
  - The Manhattan distance of a vector from another vector
  - Syntax: ||.||<sub>1</sub>
  - Formally, ||x - x<sub>n</sub>||<sub>1</sub> = Σ<sup>D</sup><sub>d=1</sub> |x<sub>d</sub> - x<sub>nd</sub>|

###### L2 Norm
  - The Euclidean distance of a vector from another vector
  - Syntax: ||.||<sub>2</sub>
  - Formally, ||x - x<sub>n</sub>||<sub>2</sub> = (Σ<sup>D</sup><sub>s=1</sub> (x<sub>d</sub> - x<sub>nd</sub>)<sup>2</sup>)<sup>1/2</sup>

###### Lp Norm
  - The general distance form for (p ≥ 1)
  - Syntax: ||.||<sub>P</sub>
  - Formally, ||x - x<sub>n</sub>||<sub>p</sub> = (Σ<sup>D</sup><sub>d=1</sub> (x<sub>d</sub> - x<sub>nd</sub>)<sup>p</sup>)<sup>1/p</sup>
  - Note: p has to be greater than or equal to 1 in order to satisfy the triangle inequality and thereby acting as a true measure of distance.

##### Indicator Function
  - The indicator function, 𝕀, denotes whether an expression is true or not
  - If an expression, x, is true, then 𝕀(x) = 1
  - If an expression, x, is false, then 𝕀(x) = 0

##### Taylor Series
##### Gradient
##### Hessian
