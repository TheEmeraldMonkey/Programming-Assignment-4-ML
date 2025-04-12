# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 1 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import random

#Import Sklearn Libraries for comparison


def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    # INSERT YOUR CODE HERE
    z = dict()
    #iterate over all of x
    for i in range(len(x)):
        if x[i] in z.keys():     #if key already in dictionary add new index
            cArray = z[x[i]]
            cArray.append(i)
            z.update({x[i] : cArray})
        else:                       #else create new key with only the current index
            cArray = [i]
            z.update({x[i] : cArray})
        

    return z
    raise Exception('Function not yet implemented!')



def entropy(labels, wts=None):
    from math import log2
    if len(labels) == 0:
        return 0
    total_weight = np.sum(wts)
    unique_labels = np.unique(labels)
    ent = 0
    for label in unique_labels:
        p = np.sum(wts[labels == label]) / total_weight
        if p > 0:
            ent -= p * log2(p)
    return ent
    raise Exception('Function not yet implemented!')


def mutual_information(split, y, weights=None):
    """Computes weighted mutual information (information gain)."""

    total_entropy = entropy(y, weights)

    # Split on True and False branches
    mask_true = split
    mask_false = ~split

    w_true = weights[mask_true]
    w_false = weights[mask_false]
    y_true = y[mask_true]
    y_false = y[mask_false]

    p_true = np.sum(w_true) / np.sum(weights)
    p_false = np.sum(w_false) / np.sum(weights)

    weighted_entropy = p_true * entropy(y_true, w_true) + p_false * entropy(y_false, w_false)

    return total_entropy - weighted_entropy
    raise Exception('Function not yet implemented!')


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5, weights=None):
    import numpy as np

    if attribute_value_pairs is None:
        attribute_value_pairs = {(j, val) for j in range(x.shape[1]) for val in np.unique(x[:, j])}

    if weights is None:
        weights = np.ones(len(y)) / len(y)

    # Base case: all labels are the same
    if len(set(y)) == 1:
        return y[0]

    # Base case: max depth or no attributes left
    if not attribute_value_pairs or depth == max_depth:
        return weighted_majority(y, weights)

    # Best attribute-value pair using weighted mutual information
    best_pair = max(
        attribute_value_pairs,
        key=lambda pair: mutual_information(x[:, pair[0]] == pair[1], y, weights)
    )

    attribute_value_pairs = attribute_value_pairs - {best_pair}
    mask_true = x[:, best_pair[0]] == best_pair[1]
    mask_false = ~mask_true

    # Subsets
    x_true, y_true, w_true = x[mask_true], y[mask_true], weights[mask_true]
    x_false, y_false, w_false = x[mask_false], y[mask_false], weights[mask_false]

    tree = {
        (best_pair[0], best_pair[1], True): id3(x_true, y_true, attribute_value_pairs.copy(), depth + 1, max_depth, w_true),
        (best_pair[0], best_pair[1], False): id3(x_false, y_false, attribute_value_pairs.copy(), depth + 1, max_depth, w_false)
    }

    return tree
    raise Exception('Function not yet implemented!')
def weighted_majority(y, weights):
    """Returns the weighted majority class in y."""
    from collections import defaultdict
    weight_sum = defaultdict(float)
    for label, w in zip(y, weights):
        weight_sum[label] += w
    return max(weight_sum, key=weight_sum.get)

def bootstrap_sampler(x, y, num_samples):

    arr = np.zeros((num_samples, 7))
    print(arr)
    
    # Blank array to fill
    blank_array = np.empty((num_samples, 7), dtype=int)

    source_array = np.concatenate((x, y.reshape(-1, 1)), axis=1)

    print(f'Shape: {source_array.shape}')  # (10, 7)
    print(f'Source Array:\n{source_array}')

    for i in range(num_samples):
        random_row_index = np.random.choice(k)
        blank_array[i] = source_array[random_row_index]
        
    return blank_array
    raise Exception('Function not yet implemented!')


def bagging(x, y, max_depth, num_trees):
    """
    Implements bagging of multiple id3 trees where each tree trains on a boostrap sample of the original dataset
    """
    raise Exception('Bagging not yet implemented!')

def boosting(x, y, max_depth, num_stumps):

    """
    Implements an adaboost algorithm using the id3 algorithm as a base decision tree
    """
    '''
    TASKS:
        * what do we return (probably something to do with a completed decision tree)
        * what algorithm are we using
        * when create bootstrap
        * what is num_stumps (num_trees, but very short trees)
        * how to loop through
        * report confusion matrix in this method, or in main?
    '''
    n = len(y)
    # Step 1: Initialize weights
    weights = np.ones(n) / n
    classifiers = []
    alphas = []

    for t in range(num_stumps):
        # Step 2: Train a weighted decision stump
        stump = id3(x, y, max_depth=max_depth, weights=weights)

        # Step 3: Make predictions on training data
        predictions = np.array([predict_example(x[i], stump) for i in range(n)])

        # Step 4: Calculate weighted error
        incorrect = predictions != y
        error = np.sum(weights * incorrect)

        # Avoid divide-by-zero errors
        if error == 0:
            alpha = 1
        else:
            alpha = 0.5 * np.log((1 - error) / error)

        # Step 5: Update weights
        weights = weights * np.exp(-alpha * y * (2 * predictions - 1))
        weights /= np.sum(weights)  # Normalize

        # Store the classifier and its alpha
        classifiers.append(stump)
        alphas.append(alpha)

    return classifiers, alphas

    
    raise Exception('Boosting not yet implemented!')

def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using a combination of weighted trees
    Returns the predicted label of x according to tree
    """
    if not isinstance(tree, dict):
        return tree  # it's a leaf

    for (feature, value, decision), subtree in tree.items():
        if (x[feature] == value) == decision:
            return predict_example(x, subtree)
    raise Exception('Function not yet implemented!')


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """
    #variable to see how many y-values don't match
    unequal = 0
    
    #iterate over all the y values
    for i in range(len(y_true)):
        #count up if y-values do not match
        if (y_true[i] != y_pred[i]):
            unequal += 1

    return (1/len(y_true) * unequal)
    # INSERT YOUR CODE HERE
    raise Exception('Function not yet implemented!')


def visualize(tree, depth=0):
    """
    Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
    print the raw nested dictionary representation.
    DO NOT MODIFY THIS FUNCTION!
    """

    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))

def convert_to_builtin(obj):
    if isinstance(obj, dict):
        return {convert_to_builtin(k): convert_to_builtin(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return type(obj)(convert_to_builtin(item) for item in obj)
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj
def predict_boosted(x, classifiers, alphas):
    preds = np.array([[predict_example(sample, clf) for clf in classifiers] for sample in x])
    weighted_preds = np.dot(preds * 2 - 1, alphas)  # Convert labels 0/1 to -1/1
    return (weighted_preds > 0).astype(int)

if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    #loop through bootstrap function to create sets to train model on
    # k = 50 
    # boot = bootstrap_sampler(Xtrn, ytrn, k)
    # print(f'bootstrap:\n{boot}')
    
    #boosting models
    for d in range(1, 3):
        for k in range(1, 3):
            boosted = boosting(Xtrn, ytrn, d, k * 20)
            clean_boosted = convert_to_builtin(boosted)
            
            y_pred = predict_boosted(Xtrn, boosted[0], boosted[1])
            TP = np.sum((y_pred == 1) & (ytrn == 1))
            TN = np.sum((y_pred == 0) & (ytrn == 0))
            FP = np.sum((y_pred == 1) & (ytrn == 0))
            FN = np.sum((y_pred == 0) & (ytrn == 1))
            
            print("Confusion Matrix:")
            print("              Predicted")
            print("              Pos     Neg")
            print(f"Actual  Pos     {TP:<5} {FN:<5}")
            print(f"        Neg     {FP:<5} {TN:<5}")
            #print(f'{d}: {k * 20}\nBoosting: {clean_boosted}')
           

    '''#figure out boosting

    # Learn a decision tree of depth 3
    decision_tree = id3(Xtrn, ytrn, max_depth=3)
    visualize(decision_tree)

    # Compute the test error
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)

    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
'''