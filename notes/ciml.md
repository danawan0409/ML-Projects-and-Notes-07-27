ML intro learning textbook @ http://ciml.info/ as recommended as a pre-requisite for https://web.stanford.edu/class/cs224n/

What does it mean to learn? 
To be able to generalize, ie. use what you learned and be able to apply it to something similar

Training data - algorithm learns from
Test set - a final exam for the algorithm

Decision Tree: 
The goal in learning is figuring out what questions to ask, in what order to ask them, and what answer to predict once you've asked them. 
Feature: questions you can ask
Feature values: responses to the question

Loss function: measure how off a system's prediction is in comparison to the truth
    Regression: Squared loss or Absolute loss
    Binary Classification: 0/1 loss
    Multiclass classification: 0/1 loss

Data generating distribution D: distribution over (x, y) pairs

Induction machine learning: given a loss function l and a sample D from some unknown distribution D, compute a function f that has low expected error e over D with respect to l. 
