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
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import graphviz

#Ameya Sansguiri and Juan Arce

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
    #Place indices of vals that match with vi into dictionary
    #v0 = indices of x where x[index] = 0
    #v1 = indices of x where x[index] = 1

    j = 0   
    dictVector = {}

    #Get all unique values in x, store as key in dictionary
    key = np.unique(x)
    for i in range(len(key)): 
        dictVector.setdefault(key[i], [])         #Sets default value of each key in dictionary to empty list
    dictKeys = list(dictVector.keys())          #Stores keys in list
    
    #Traverses through the x vals to find which vals match the dictionary key, stores indices as values for the key
    for i in range(len(x)): #For each x val
        for j in range(len(dictKeys)):      #For each key in dictionary
            if x[i] == dictKeys[j]:     #If x val matches the key and the x val's index is not already in the dictionary, add the index to dictionary
                if i not in dictVector[x[i]]:
                    dictVector[x[i]].append(i)

    return dictVector
    raise Exception('Function not yet implemented!')


def entropy(y, weights=None):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z. 
    Include the weights of the boosted examples if present

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    # INSERT YOUR CODE HERE
    if weights == None:
        v = partition(y)    #Gets the unique values of y and the indices with the unique values by calling the partition function
        ent = 0             #Stores entropy
        key = v.keys()      #Stores the unique values/the keys 
        for i in key:  #For length of dictionary, calculate each vi's probability and entropy
                examp = (len(v[i])) / (y.size)      #Calculate p(z = vi) = count(indices of unique values) / total size of v
                if examp > 0:
                    ent += (-(examp * np.log2(examp)))       #Calculate entropy and add to existing entropy count => P (z=vi) * log(P(z=vi))
    else:
        total_weight = np.sum(weights)
        unique_labels = np.unique(y)
        ent = 0
        for label in unique_labels:
            p = np.sum(weights[y == label]) / total_weight
        if p > 0:
            ent -= p * np.log2(p)
    return ent
    
    raise Exception('Function not yet implemented!')


def mutual_information(x, y, weights=None):
    """
    
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)

    Compute the weighted mutual information for Boosted learners
    """

    # INSERT YOUR CODE HERE
    if weights == None:
        yEnt = entropy(y)       #Stores H(y)
        xUniqueVals = partition(x)  #Gets dictionary with unique vals of x
        xKeys = list(xUniqueVals.keys()) #Gets keys of x-vals --> unique values
        totalConditEnt = 0      #Stores the conditional entropy H (y|x)
        probX = 0
        probXY = 0

        #For each unique x val
        for i in range(len(xKeys)):
            probX = len(xUniqueVals[xKeys[i]])/ (x.size)       #P(X = x) = length of list with unique val of x / total x size
            probXY = 0
            yGivenX = y[(x == xKeys[i])]   #Get Y val when X = x
            probXY = entropy(yGivenX)       #Calculates entropy of when Y=y and X=x
            totalConditEnt += ((probX * probXY))    #Calculates total conditional entropy and adds it to corresponding variable
            
        mutualInfo = yEnt - totalConditEnt  #Mutual info calculated

    else:
        total_entropy = entropy(y, weights)

        # Split on True and False branches
        mask_true = x
        mask_false = ~x

        w_true = weights[mask_true]
        w_false = weights[mask_false]
        y_true = y[mask_true]
        y_false = y[mask_false]

        p_true = np.sum(w_true) / np.sum(weights)
        p_false = np.sum(w_false) / np.sum(weights)

        weighted_entropy = p_true * entropy(y_true, w_true) + p_false * entropy(y_false, w_false)

        mutualInfo =  total_entropy - weighted_entropy
    return mutualInfo
    raise Exception('Function not yet implemented!')

 #Mutual information helper - calculates mutual information of a split within a column
def mutual_info_col(x, val, y):
    yEnt = entropy(y)       #Stores H(y)
    xUniqueVals = partition(x)  #Gets dictionary with unique vals of x
    yUniqueVals = partition(y)  #Gets dictionary with unique vals of y
    xKeys = list(xUniqueVals.keys()) #Gets keys of x-vals --> unique values
    yKeys = list(yUniqueVals.keys())   #Gets keys of y-vals --> unique values
    probX = len(xUniqueVals[val])/ (x.size)       #P(X = x) = length of list with unique val of x / total x size
    probNotX = 1 - probX        #P(X != x) = 1 - P(X = x)
    yGivenX = y[(x == val)]   #Get Y vals when X = x
    yNotGivenX = y[x != val]    #Get Y vals when X != x

    probXY = entropy(yGivenX)       #Calculates entropy of when Y=y and X=x
    probNotXY = entropy(yNotGivenX) #Calculates entropy of when Y=y and X!=x
    totalConditEnt = ((probX * probXY) + (probNotX * probNotXY))   #Calculates total conditional entropy for the split on the specific value
    mutualInfo = yEnt - totalConditEnt      #Calculates the mutual information of the specific split on the value, returns the mutual information
    return mutualInfo


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5, weights=None):
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

    #  #If first time coming in to ID3, initialize attribute_value_pairs
    if attribute_value_pairs == None:
        #Traverse through set, add attribute col num and val pairs that have not already been added
        #For each col, place the val into the set with the col num 
        
        attribute_value_pairs = list() 
        numCols = x.shape[1]

        #For each col, place the val into the attribute_value_pairs list with the col num 
        for i in range(numCols):
            xCol = x[: , i]     #Takes the current column
            for j in range(xCol.size):      #For each value in the column
                 val = xCol[j]
                 attValPair = (i, val)      #Store the col num and val as a tuple
                 if attValPair not in attribute_value_pairs:        #If not already added, add the pair 
                    attribute_value_pairs.append(attValPair)

#Condition checking
    zeroY = 0       #Stores num of y vals with 0
    oneY = 0        #Stores num of y vals with 1
    total = 0       #Total y vals

    #Condition 1: Counts the y vals with 0 and y vals with 1 by looping thorugh the y vals
    #If either the zeroY count or oneY count equal the total num of vals --> pure --> return the corresponding label
    for i in range(len(y)):
        if y[i] == 0:
            zeroY += 1
        elif y[i] == 1:
            oneY += 1
    total = zeroY + oneY
    if zeroY == total:
            return 0
    elif oneY == total:
            return 1
    
    #Condition 2 and 3 - if attribute_value_pairs has no elements or the max depth has been reached, return the most common label
    if (len(attribute_value_pairs) == 0) or (depth == max_depth):
        if zeroY > oneY:
            return 0
        else:
            return 1
        
    #If there are no examples, return the most common label
    if len(x) == 0:
        if zeroY > oneY:
            return 0
        else:
            return 1

    #Chooses best attribute-value pair by looping through each column, calculating information gain - pick x col with highest mutual information -> call mutual information function
    #Then loop through each attribute-value pair and check what specific value of the best chosen x col will give the highest mutual information --> call mutual information helper function
    #Mutual info takes x and y as params - x = col  
    mutInfo = 0     #Stores mutual info
    xColumn = 0      #Stores the xCol with the highest mutual info
    bestPair = attribute_value_pairs[0]       #Takes val of ith col from pairs list
    if isinstance(x, (list)):       #If x is a list, make it an np array
                x = np.array(x)
    cols = x.shape[1]   #Gets the number of columns in x
    for i in range(cols):       #For each x column
           
            xCol = x[:, i]    #Get the current col of x
            
            temp = mutual_information(xCol, y)  #Sends x col and corresponding y vals to mutual info

            #If the mutual info calculated is greater than the one already stored, replace it so mutInfo has the highest mutual info
            if temp > mutInfo:
                mutInfo = temp
                xColumn = i       #Best column is now the current col
            
    mutInfo = 0     #Reset mutInfo to use now
    #Found the col with the best mutual information, now want to find the col-val pair in attribute-val pairs that gives best mutual information
    for i in range(len(attribute_value_pairs)):
        column, value = attribute_value_pairs[i]
        if column == xColumn:       #If the column of the current attribute-value pair matches the best x column found,
            xCol = x[:, column]     #Get the x values of the column
            if np.isin(value, xCol):    #If the value of the current attribute-value pair is in the column, send the column, value, and y labels to the mutual info helper function
                temp = mutual_info_col(xCol, value, y)      #Calculates the mutual information between the specific value of the attribute and y
                if temp > mutInfo:      #If the mutual information found is greater than the one already stored, replace it so mutInfo has the highest mutual info
                    mutInfo = temp
                    bestPair = attribute_value_pairs[i]     #Assign the bestPair with the current attribute-value pair

    #Creates an empty tree
    tree = {}

    #Check through each example, check if Xi of best pair = specified value (EX: check if X2 = 1, X2 = 2, etc.)
    #Loop through each possible Xi val
    bestAttribute, bestValue = bestPair     #Get the best attribute and best value from the best pair calculated
    bestPairTrue = (bestAttribute, bestValue, True)     #Format to be stored as in tree
    bestPairFalse = (bestAttribute, bestValue, False)   #Format to be stored as in tree
    yTrueVals = []              #Stores the corresponding y vals of the examples that are true
    yFalseVals = []             #Stores the corresponding y vals of the examples that are false
    trueExamples = []           #Stores the examples that are true
    falseExamples = []          #Stores the examples that are false
    attribute_value_pairs.remove(bestPair)  #Removes the best attribute value pair from attribute_value_pairs before splitting

   #For each value in x
   #If they are equal, add the example and the corresponding y val to trueExamples and yTrueVals 
    for i in range(len(x)):      
         if x[i][(bestAttribute)] == bestValue:      #Check if the x val in the column = the best value from the best pair
            trueExamples.append(x[i])        #If they are equal (meaning Xi = true), add the example to trueExamples
            yTrueVals.append(y[i])          #Add the corresponding y val to yTrueVals
         else:
             falseExamples.append(x[i])     #Else, Xi = false, so add the example to falseExamples
             yFalseVals.append(y[i])        #Add the corresponding y val to yFalseVals

    yTrueVals = np.array(yTrueVals)     #Make yTrueVals an np array
    yFalseVals = np.array(yFalseVals)   #Make yFalseVals an np array

    #Call ID3 recursively for the true and the false examples, making the children/other nodes of the tree
    tree[bestPairTrue]= id3(trueExamples, yTrueVals, attribute_value_pairs, depth + 1, max_depth)
    tree[bestPairFalse]= id3(falseExamples, yFalseVals, attribute_value_pairs, depth + 1, max_depth)

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
    xtrnBt = np.empty(6)
    ytrnBt = np.empty(1)
    for i in range(num_samples):
        index = random.randint(0, len(x) - 1)
        xtrnBt = np.vstack((xtrnBt, x[index]))
        ytrnBt = np.append(ytrnBt, y[index])
    xtrnBt = np.delete(xtrnBt, 0, axis = 0)
    ytrnBt = np.delete(ytrnBt, 0, axis = 0)
    return xtrnBt, ytrnBt  
    raise Exception('Function not yet implemented!')

def findMajority(y):
    countZeros = 0
    countOnes = 1
    for i in range(len(y)):
        if y[i] == 0:
            countZeros += 1
        else:
            countOnes += 1
    if countZeros > countOnes:
        return 0
    else:
        return 1

def bagging(x, y, max_depth, num_trees):
    """
    Implements bagging of multiple id3 trees where each tree trains on a boostrap sample of the original dataset
    """
    pred = []
    majority = []
    y_pred = [0] * len(y)
    tempPred = []
    for i in range(num_trees):
    # Learn a decision tree with bootstrap set with depth max_depth
        decision_tree = id3(x, y, max_depth=max_depth)
        #Predict y set
        for j in range(len(x)):
            y_pred[j] = predict_example(x[j], decision_tree)
        pred.append(y_pred)     #Add set of predicitons to list
    for k in range(len(x)):
        tempPred = []
        for p in pred:
            tempPred.append(p[k])       #Add y predictions of the kth column
        majority.append(findMajority(tempPred)) #Find a majority vote of the y predictions
    return majority
    raise Exception('Bagging not yet implemented!')

def boosting(x, y, max_depth, num_stumps):

    """
    Implements an adaboost algorithm using the id3 algorithm as a base decision tree
    """
    n = len(y)
    # Step 1: Initialize weights
    weights = np.ones(n) / n
    classifiers = []
    alphas = []
    ensemble = []

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
        pair = (alpha, stump)
        ensemble.append(pair)
        classifiers.append(stump)
        alphas.append(alpha)

    return ensemble

    
    raise Exception('Boosting not yet implemented!')

def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    
    #If the tree is not a dictionary, but is a tuple, get the label and return the label if it is 1 or 0
    #If the tree is not a tuple, that means the tree is an integer, return the tree
    if type(tree) is not dict:
        if type(tree) is tuple:
            pair, label = tree
            if label == 1 or label == 0:
                return label
        else:
            return tree
    
    treeIterator = iter(tree.items())       #Iterator to get the left and right subtrees
    leftvalPair, leftsubtree =  next(treeIterator)      #Gets the left subtree and stores the pair and remaining subtree in corresponding variables
    rightvalPair, rightsubtree = next(treeIterator)     #Gets the right subtree and stores the pair and remaining subtree in corresponding variables
    leftindex, leftval, labelPred = leftvalPair         #Gets the index, val, and label of the left pair
    rightindex, rightval, labelPred = rightvalPair      #Gets the index, val, and label of the right pair

    
    if x[leftindex] == leftval:                   #If the val of the best attribute in the example equals the best val in the left subtree, traverse down the left subtree
        return predict_example(x, leftsubtree)
    elif x[rightindex] == rightval:                #If the val of the best attribute in the example equals the best val in the right subtree, traverse down the right subtree
        return predict_example(x, rightsubtree)            
    elif False in rightvalPair:                         #If the pair has false, traverse down the right subtree (to the right = false)
        return predict_example(x, rightsubtree)
    else:
         return predict_example(x, leftsubtree)        #Else, traverse down the left subtree

    #raise Exception('Function not yet implemented!')

#h_ens = ensemble of weighted hypotheses
#h_ens = array of pairs (hypothesis, weight)
def predict_example_ens(x, h_ens):
    """
    Predicts the classification label for a single example x using a combination of weighted trees
    Returns the predicted label of x according to tree
    """
    alphas = [i[0] for i in h_ens]
    classifiers = [i[1] for i in h_ens]
    preds = np.array([[predict_example(sample, clf) for clf in classifiers] for sample in x])
    weighted_preds = np.dot(preds * 2 - 1, alphas)  # Convert labels 0/1 to -1/1
    return (weighted_preds > 0).astype(int)

def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    # INSERT YOUR CODE HERE
    matchSum = 0        #Counter for the vals that do not match
    for i in range(len(y_true)):        #Traverses through both true and predicted y val arrays, if the vals don't match, add 1 to counter
        if (y_true[i] != y_pred[i]):
            matchSum += 1
    error = (1/len(y_true)) * matchSum      #Error = 1/size of true y val array * the num of vals that don't match
    return error
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

#Calculates the confusion matrix, takes the predicted and actual y values as arguments
def confMatrix(yPred, yActual):
    #Counters for true positive, false negative, false positive, and true negative
    truePositive = 0        
    falseNegative = 0
    falsePositive = 0
    trueNegative = 0

    #Loops through actual y vals
    for i in range(len(yActual)):
        if yPred[i] == yActual[i] == 1:     #If both actual and predicted y values are 1, add 1 to true positive count
            truePositive += 1
        elif yPred[i] == yActual[i] == 0:   #If both actual and predicted y values are 0, add 1 to true negative count
            trueNegative += 1
        else:
            if yPred[i] == 1 and yActual[i] == 0:   #If actual y value is 0 and predicted y val is 1, add 1 to false positive count
                falsePositive += 1
            else:
                falseNegative += 1                  #Else, add 1 to false negative count
    
    #Builds table to display on screen
    print("                              Predicted Value")
    print("                     Positive                    Negative")
    print("                 --------------------------------------------")
    print("                 |  True Positive      |  False Negative     |")
    print("  Positive       | " +  str(truePositive) + "                   | " + str(falseNegative) + "                   |")
    print("                 |                     |                     |")
    print("Actual           |---------------------| --------------------|")
    print("Value            |  False Positive     |  True Negative      |")
    print("                 | " +  str(falsePositive) + "                   | " + str(trueNegative) + "                   |")
    print("   Negative      |                     |                     |")
    print("                 |                     |                     |")
    print("                 --------------------------------------------")

        

if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    monks1set = [Xtrn, ytrn, Xtst, ytst]  

    M = np.genfromtxt('monks-2.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    #Xtst = 6 labels from cols 2-7
    #ytst = label - 1st col
    M = np.genfromtxt('monks-2.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    monks2set = [Xtrn, ytrn, Xtst, ytst] 

    M = np.genfromtxt('monks-3.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    #Xtst = 6 labels from cols 2-7
    #ytst = label - 1st col
    M = np.genfromtxt('monks-3.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    monks3set = [Xtrn, ytrn, Xtst, ytst]

    setNum = 1
    for set in [monks1set, monks2set, monks3set]:
        if setNum == 1:
            print("====================================================================================================")
            print("MONKS-1")
        elif setNum == 2:
            print("====================================================================================================")
            print("MONKS-2")
        else:
            print("====================================================================================================")
            print("MONKS-3")

        Xtrn, ytrn, Xtst, ytst = set
        print("BAGGING:")
        depth = [3,5]
        bagSize = [10, 20]
        #Loops through each combination of depth of 3, 5 and bag size of 10, 20 for bagging
        for i in depth:
            for j in bagSize:
                print("-------------------------------------------------------------------------------------------------")
                print("Depth: " + str(i) +"    Bag Size: " + str(j))

                #Training bootstrap set
                print("\nTraining Set:")
                bootStrapTrn = bootstrap_sampler(Xtrn, ytrn, j)
                xtrnBootStrap, ytrnBootStrap = bootStrapTrn
                y_predTrn = bagging(xtrnBootStrap, ytrnBootStrap, i, j)
                errTrn = compute_error(ytrnBootStrap, y_predTrn)
                print('Train Error = {0:4.2f}%.'.format(errTrn * 100))
                cMatrix = confMatrix(y_predTrn, ytrnBootStrap)      #Confusion matrix for training bootstrap set

                #Test bootstrap set     Changed
                print("\nTest Set:")
                #bootStrapTst = bootstrap_sampler(Xtst, ytst, j)
                #xtstBootStrap, ytstBootStrap = bootStrapTst
                y_predTst = bagging(Xtst, ytst, i, j)       #changed
                errTst = compute_error(ytst, y_predTst)
                print('Test Error = {0:4.2f}%.'.format(errTst * 100))
                cMatrix = confMatrix(y_predTst, ytst)      #Confusion matrix for testing bootstrap set

                #Scikit-Learn training bootstrap set
                print("\n\nScikit-Learn:")
                clf = BaggingClassifier()
                clf = clf.fit(Xtrn, ytrn)
                yPredSK = clf.predict(xtrnBootStrap)
                errTrnSK = compute_error(ytrnBootStrap, yPredSK)
                print('\nTrain Scikit-Learn Error = {0:4.2f}%.'.format(errTrnSK * 100))
                confusionMatrix = confusion_matrix(ytrnBootStrap, yPredSK)
                matrix = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=['0', '1'])
                matrix.plot()
                matrix.ax_.set_title("Bagging - Depth " + str(i) + " and Bag Size " + str(j) + " - SciKit-Learn Train Bootstrap Confusion Matrix")
                plt.show()

                #Scikit-Learn test bootstrap set
                clf = BaggingClassifier()
                clf = clf.fit(Xtst, ytst)       #changed
                yPredSK = clf.predict(Xtst) #xtstBootStrap                  Changed
                errTstSK = compute_error(ytst, yPredSK) #ytstBootstrap      Changed
                print('\nTest Scikit-Learn Error = {0:4.2f}%.'.format(errTstSK * 100))
                confusionMatrix = confusion_matrix(ytst, yPredSK)  #Changed
                matrix = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=['0', '1'])
                matrix.plot()
                matrix.ax_.set_title("Bagging - Depth " + str(i) + " and Bag Size " + str(j) + " - SciKit-Learn Test Bootstrap Set Confusion Matrix")
                plt.show()
    
        print("-----------------------------------------------------------------------------------------------")
        print("BOOSTING")
        for d in range(1, 3):
            for k in range(1, 3):
                #Loops for each combination of depth of 1, 2 and bag size of 20, 40
                print("-------------------------------------------------------------------------------------------------")
                print("Depth: " + str(d) +"    Bag Size: " + str(k*20))

                #Training bootstrap set
                print("\nTraining Set: ")
                bootStrapTrn = bootstrap_sampler(Xtrn, ytrn, k*20)
                xtrnBootStrap, ytrnBootStrap = bootStrapTrn
                boosted = boosting(xtrnBootStrap, ytrnBootStrap, d, k * 20)
                y_predTrn = predict_example_ens(xtrnBootStrap, boosted)
                errorTrn = compute_error(ytrnBootStrap, y_predTrn)
                print('Train Error = {0:4.2f}%.'.format(errorTrn * 100))
                cMatrix = confMatrix(y_predTrn, ytrnBootStrap)
                
                #Test bootstrap set
                print("\nTest Set:")
                #bootStrapTst = bootstrap_sampler(Xtst, ytst, k*20)     Changed
                #xtstBootStrap, ytstBootStrap = bootStrapTst
                boosted = boosting(Xtst, ytst, d, k * 20)
                y_predTst = predict_example_ens(Xtst, boosted)  #changed
                errorTst = compute_error(ytst, y_predTst)       #changed
                print('Test Error = {0:4.2f}%.'.format(errorTst * 100))
                cMatrix = confMatrix(y_predTst, ytst)           #changed

                #Adaboost training bootstrap set
                print("\n\nAdaboost:")
                clf = AdaBoostClassifier()
                clf = clf.fit(Xtrn, ytrn)
                yPredAda = clf.predict(xtrnBootStrap)
                errorTrn = compute_error(ytrnBootStrap, yPredAda)
                print('\nTrain Adaboost Error = {0:4.2f}%.'.format(errorTrn * 100))
                confusionMatrix = confusion_matrix(ytrnBootStrap, yPredAda)
                matrix = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=['0', '1'])
                matrix.plot()
                matrix.ax_.set_title("Adaboost - Depth " + str(d) + " and Bag Size " + str(k*20) + " - Train Bootstrap Set Confusion Matrix")
                plt.show()

                #Adaboost test bootstrap set
                clf = AdaBoostClassifier()
                clf = clf.fit(Xtst, ytst)       #changed
                yPredAda = clf.predict(Xtst)    #changed
                errorTst = compute_error(ytst, yPredAda)    #changed
                print('\nTest Adaboost Error = {0:4.2f}%.'.format(errorTst * 100))
                confusionMatrix = confusion_matrix(ytst, yPredAda)  #changed
                matrix = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=['0', '1'])
                matrix.plot()
                matrix.ax_.set_title("Adaboost - Depth " + str(d) + " and Bag Size " + str(k*20) + " - Test Bootstrap Set Confusion Matrix")
                plt.show()

        setNum += 1
