# ML intro learning textbook 
## @ http://ciml.info/ as recommended as a pre-requisite for https://web.stanford.edu/class/cs224n/

## Ch1: Decision Trees

What does it mean to learn? 
To be able to generalize, ie. use what you learned and be able to apply it to something similar

Training data: algorithm learns from
Test set: a final exam for the algorithm

Decision Tree: 
The goal in learning is figuring out what questions to ask, in what order to ask them, and what answer to predict once you've asked them. 
Feature: questions you can ask
Feature values: responses to the question

Variations of Decision Trees: 
Shallow decision tree: have a pre-defined maximum depth d, and once it queried on d many features, we must make a guess

Loss function: measure how off a system's prediction is in comparison to the truth
    Regression: Squared loss or Absolute loss
    Binary Classification: 0/1 loss
    Multiclass classification: 0/1 loss
    0/1 loss function: counts how many mistakes an hyppothesis function h makes on the training set 

Data generating distribution D: distribution over (x, y) pairs

Induction machine learning: given a loss function l and a sample D from some unknown distribution D, compute a function f that has low expected error e over D with respect to l. 

## Ch2: Limits to learning
Bayes optimal classifier: a classifier that returns the y that maximizes the distribution D for any test input x. Achieves minimal zero/one error of any determinisitic classifier 
![alt text](image-1.png)
Bayes error rate: error rate of the Bayes optimal classifier, and the best error rate you can ever hope to achieve on this classification problem. 
Parity function: inspect every feature to make a prediction

Sources of error
Inductive bias: a set of assumptions made by an algorithm. 
    Ex. Shallow decision tree would be good at learning a function like 'students only like AI courses', but not so good at 'if this studnet has liked an odd nuber of their past coureses, they will like the next one". 
Noise in data - feature level and label level
    Ex. Student writes a scathingly negative review for a course, but accidently click 5* (label level). Typo in the review (feature level)
Features available for learning is insufficient
    Ex. Made mistake when downloading data, only dl first 5 reviews of each course
May not have a single correct answer
    Ex. label what may be considered offensive (as ppl find different content offensive)

Underfitting: had the opportunity to learning something but didn't
    Ex. empty decision tree, just guesses one answer
Overfitting: pay too much attention of idiosyncracies of training data, can't generalize. 
    Ex. a decision tree with a node for each example

Seperate training & test data (usually 90/10 if you have a lot of data)
Development data: seperate from training & test data, for tuning hyperparameters
General appoarch: 
1. split data into training data, development data, and test data
2. for each possible setting of the hyperparameter, train a model using that setting of hyperparameters and then compute the model's error rate on the development data
3. choose the one that achieved the lowest error rate on development data
4. evaluate that model on the test data to estimate future test performance

Model: tells us what sort of things we can learn and its inductive bias
Parameters: we use the data to decide on, what the algorithm has to figure out. 
    Ex. decision tree learning algorithm needs to take the data and figure out the parameters, i.e. specific questions to ask, classification decisions at the leaves
Hyperparamters: additional knobs that you can adjust, using to tune the inductive bias of algorithms
    Ex. maximum depth for the decision trees 

IRL applications of ML: 
![alt text](image-3.png)
Oracle experiment: assume everything below some line can be solved perfectly, and measure how much impact that will have on a higher line
    Ex. if classifier is perfect, how much money would we make? Use this to decide whether this problem is worth tackling

# Ch3: Geometry and nearest neighbors 
Mapping data set to a feature vector
 - real-valued features get copied directly
 - binary features -> 0/1
 - categorical features -> v-many binary indicator features (i.e. isYellow vector, 0/1 value)

Methods to compute distance between feature vectors
Euclidean distance: ![alt text](image-4.png)

Nearest neighbours: prone to overfitting as it only looks at the neighbour closest to it
![alt text](image-5.png)
K-nearest neighbours: solves the problem above
k: hyperparameter

Inductive bias of k-neighbours
 - assumes nearby points should have the same label
 - assumes all features are eequally important 
 - feature scaling (i.e. distingushing between ski and snowboard based on width and height. If width is given in mm and height is in cm, then it will almost purely based on height)

Steps to the algorithm
1. compute distance from test point to all training points
2. points are sorted according to distance
3. sum the class labels for each of the K nearest neighbours and using the sign of this sum as our predictions

K-means clustering: represent each cluster by it's cluster center
Steps to find this center: 
1. guess the clsuter centers
2. assign data point to closest center
3. recomputer cluster center
4. repeat until clusters stop moving
Hyperparameters: number of clusters

Curse of dimensionality
Computational: slow for a very large data set, as you must look at every training example every time you want to make a prediction. You can split the plane into a grid, and only train points in the grid cell of the test point. However it is not realistic in higher dimensions as the number of cells is too large. 
Mathematical: math gets weird when working in higher dimensions, as many of your intuitions don't carry over. Most notably, in moderately high dimensions, all distances become equal and it makes it hard for knn to accurately distinguish clusters.

Solutions
Dimension reduction

# Ch4: The perceptron
Perceptron algorithm: can learn weights for features. Based on how neurons work. The sign of the sum of weights (w) & input vectors (x), which is called the activation, determine if the neuron 'fires'. 
![alt text](image-6.png)
To have a non-zero threshold, introduce a bias term b. 
![alt text](image-7.png)
Steps of the algorithm: 
![alt text](image-8.png)
notes: 
1. since y is -1 if the neuron doesn't fire and +1 if it does, ya will be > 0 if they have the same sign and < 0 if they don't. Thus the algorithm only updates the weight if the algorithm predicts wrong. 
2. algorithm looks at one example at a time

Hyperparameter: 
MaxIter: how many times it passes through the data. Too many -> overfitting, too little -> underfitting

Permuting: permute the order of examples at the beginning of each iteration to yield ~20% savings in # of iterations (in practice). 

Geometric interpretation: 
If we think of the weights as a vector w, then the decision boundary is simply the plane perpendicular to w. Note that the scale of the weight vector is irrelevant as only the sign matters, so its common to work with normalized weight vectors. 

Scaling features: if you want to find how sensitive the final classification is to a feature, you can sort all the weights from largest to smallest and select the ones at the top and bottom. However if w1 == w2 yet x1 can take the values 0, 1, and x2 can take on values between 0 and 100, then in practice, x2 (100*w2) can over-dominate x1 (1*w1). To fix this, scale the features appropriately before using perceptron algorithm. 

Does this algorithm converge?
It converges only if the data is linearly separable, i.e. if there exists a hyperplane that separates all the + examples and - examples. 
Margin: the distance between the hyperplane that separates the data and nearest point. 
![alt text](image-9.png)
Margin of a data set: ![alt text](image-10.png)

Perceptron convergence theorem: 
![alt text](image-11.png)
Do understand the proof but don't need to know it 

Voted perceptron: hyperplanes get votes based on how long they survived, and the prediction on a test point would be ![alt text](image-12.png), where c is the survival times for each weight vector. Impractical as you need to store weight vectors with their counts for every update, and super slow
Average perceptron: similar to voted perceptron except you maintain a running rum of hte averaged weight vector and average bias. ![alt text](image-13.png)

Limitations: decision boundaries has to be linear. Can't do stuff like the XOR problem. 

# Ch5: Practical issues

