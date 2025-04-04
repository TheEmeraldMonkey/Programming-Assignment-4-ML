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
import math


def partition(x):       #tested
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

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
    # INSERT YOUR CODE HERE
    raise Exception('Function not yet implemented!')

#uncertainty of the value of x (how much suprise there will be)
def entropy(y):    #finished not tested
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    #create a dictionary of the counts of  0 and 1
    unique, counts = np.unique(y, return_counts=True)
    dictionary = dict(zip(unique, counts))
    
    #if there is only 1's or only 0's return 0 (there is not entropy/surprise)
    if len(unique) == 1:
        return 0
    
    #find the individual entropies
    zeros = dictionary[0]       #find the amount of zeros
    zeros = zeros/np.size(y)    #calculate the probability of zero
    zeros = -zeros * math.log2(zeros)    #math for the entropy
    
    ones = dictionary[1]
    ones = ones/np.size(y)
    ones = -ones * math.log2(ones)
    
    #calculate the final entropy
    answer = zeros + ones


    return answer
    
    return #p(y=1) long2(p(y=1)) + p(y=0) log2(p(y=0))
    # INSERT YOUR CODE HERE
    raise Exception('Function not yet implemented!')


def mutual_information(x, y):       #done and tested
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    oldEntropy = entropy(y)     #calculate current level entropy
    
    #go through all the possible splits
    partitions = partition(x)
    
    for key in partitions.values():  #loop through all the partitions(keys)
        zeros = 0
        ones = 0
        for val in key:
            if (y[val] == 0):
                zeros += 1
            else:
                ones += 1
                
        total = zeros + ones
        
        
        if zeros != 0 and ones != 0:
            H = -(zeros/total)*math.log2(zeros/total) -(ones/total)*math.log2(ones/total)
        elif zeros == 0:
            H = -(ones/total)*math.log2(ones/total)
        elif (ones == 0):
            H = -(zeros/total)*math.log2(zeros/total)
        oldEntropy -= (total/len(y) * H)
    
    return oldEntropy
    

    # INSERT YOUR CODE HERE
    raise Exception('Function not yet implemented!')


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    #create a list of all the attribute-value pairs
    if attribute_value_pairs is None:
        attribute_value_pairs = {(j, val) for j in range(x.shape[1]) for val in np.unique(x[:, j])}
        
    #base cases
    #if all the items of y are uniform
    if len(set(y)) == 1:  # Pure labels
        return y[0]

    #if there are no more attribute value pairs or if the max depth has been reached
    if not attribute_value_pairs or depth == max_depth:  # Stopping condition
        return max(set(y), key=list(y).count)
    
    #find the best pair by finding the attribute-value pair with the highest information gain
    #max finds the biggest value of the parameters given
    #attribute_value_pairs is the list which we are iterating over
    #key=lambda pair makes pair the current pair that we are checking the information gained from
    #mutual_information is the method that we use for information gain
    #x[:, pair[0]] == pair[1] chooses only the values that are included in the chosen partition
    best_pair = max(attribute_value_pairs, key=lambda pair: mutual_information(x[:, pair[0]] == pair[1], y))
    
    #remove the chose best pair from the list of attibute_value_pairs that can be chose as partitions
    attribute_value_pairs = attribute_value_pairs - {best_pair}
    
    #prepare the x's and y's that are going to continue the tree
    x_true = x[x[:, best_pair[0]] == best_pair[1]]
    y_true = y[x[:, best_pair[0]] == best_pair[1]]
    x_false = x[x[:, best_pair[0]] != best_pair[1]]
    y_false = y[x[:, best_pair[0]] != best_pair[1]]
    
    #recursive call for what is included in true partition and false partition
    tree = {
        (best_pair[0], best_pair[1], True): id3(x_true, y_true, attribute_value_pairs.copy(), depth + 1, max_depth),
        (best_pair[0], best_pair[1], False): id3(x_false, y_false, attribute_value_pairs.copy(), depth + 1, max_depth)
    }

    return tree
    
    
    raise Exception('Function not yet implemented!')


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """
    #base case
    if(type(tree) is np.int64):
        return tree
    
    #find the current branch posibility
    key = list(tree.keys())[0]
    
    #find which branch to go down
    if x[key[0]] == key[1]:  # True branch
        return predict_example(x, tree[key])
    else:  # False branch
        return predict_example(x, tree[(key[0], key[1], False)])    

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    raise Exception('Function not yet implemented!')


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_trues) and the predicted labels (y_pred)

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


if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    #M = np.genfromtxt('./chatgpt.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    #M = np.genfromtxt('./chatgpt.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    # Learn a decision tree of depth 3
    decision_tree = id3(Xtrn, ytrn, max_depth=3)
    visualize(decision_tree)

    # Compute the test error
    y_pred = [predict_example(x, decision_tree) for x in Xtst]      #currently predicting all 1's
    tst_err = compute_error(ytst, y_pred)

    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
