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

#Ameya Sansguiri
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import graphviz

#Takes col of x-vals as input, returns dictionary with key as vi (unique val) and value as indices of x whose values match each other
#Top-down induction
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

#Uses partition function
#Determines certainty
#For each vi, calculate entropy and add to existing entropy count
def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    # INSERT YOUR CODE HERE
    v = partition(y)    #Gets the unique values of y and the indices with the unique values by calling the partition function
    ent = 0             #Stores entropy
    key = v.keys()      #Stores the unique values/the keys 
    for i in key:  #For length of dictionary, calculate each vi's probability and entropy
                examp = (len(v[i])) / (y.size)      #Calculate p(z = vi) = count(indices of unique values) / total size of v
                if examp > 0:
                    ent += (-(examp * np.log2(examp)))       #Calculate entropy and add to existing entropy count => P (z=vi) * log(P(z=vi))
    return ent
    raise Exception('Function not yet implemented!')

#x is an attribute over all ex. - 1st col of 1st feature, 2nd col of 2nd feature, etc.
#Y = all labels
#Calls entropy function before split set
#Calls entropy function for each possible split, take avg of it
#H(y | x) = - sum(P(X = x)) sum(P(Y = y | X = x)) log P(Y=y | X =x)
#For each xi, calculate P(Y| Xi = x)

def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """

    # INSERT YOUR CODE HERE
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

#Input to algorithm = training data and max depth of tree to be learned
#First, should iterate through y, check if y-val = 0 or 1, maintain 2 counts: one for y = 0 and one for y = 1, compare to size of y ==> 1st condition conpleted
#Condition 2: from above iteration, should compare y = 0 and y = 1 counts, return majority
#Condition 3: if depth == max_depth -> same as condition 2
#Otherwise, calculates information gain - pick x with highest mutual information -> call mutual_information function
#Call id3 recursively
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
        
   #If first time coming in to ID3, initialize attribute_value_pairs
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
    print("  Positive       | " +  str(truePositive) + "                 |   " + str(falseNegative) + "               |")
    print("                 |                     |                     |")
    print("Actual           |---------------------| --------------------|")
    print("Value            |  False Positive     |  True Negative      |")
    print("                 | " +  str(falsePositive) +   "                   |   " + str(trueNegative) + "               |")
    print("   Negative      |                     |                     |")
    print("                 |                     |                     |")
    print("                 --------------------------------------------")

        

def convert(value):
   # try:
        # Try to convert to float
        if(isinstance(value, int)):
            return int(value)
        else:
        # If it fails, encode as hex
            return value.encode('utf-8').hex()
def decode_hex(value):
   # try:
        if(isinstance(value, int)):
            return int(value)
   # except ValueError:
        # Try to decode assuming it's hex
        else:
            return bytes.fromhex(value).decode('utf-8')
   # except ValueError:
        # If it fails (it's just a number), leave it as-is
    #    return value

if __name__ == '__main__':
    # Load the training data
    filename = f'./wcmatches.csv'
    #M = np.genfromtxt('./wcmatches.csv', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    M = np.genfromtxt('./wcmatches.csv', skip_header=0, delimiter=',', dtype=str)


    # Step 3: Apply it across the entire array
    vectorized_convert = np.vectorize(convert)  # Applies function to every element
    converted_data = vectorized_convert(M)

    # Step 2: Vectorize it to apply over the array
    vectorized_decode = np.vectorize(decode_hex, otypes=[str])
    decoded_data = vectorized_decode(converted_data)
    print(decoded_data)
    
    #create filter
    country_list = ['Brazil', 'Germany', 'Italy', 'Argentina', 'France', 'Spain', 'England', 'Uruguay']
    hex_vals = vectorized_convert(country_list)
    print(f'hex vals: {hex_vals}')
    #target_value = 'Uruguay'.encode('utf-8').hex()
    #print(f'target value: {target_value}')

    # Find which rows match
    mask = np.isin(converted_data[:, 4], hex_vals)

    # Apply the mask to get the filtered rows
    filteredHomeTeam_rows = converted_data[mask]

    mask = np.isin(filteredHomeTeam_rows[:, 5], hex_vals)

    # Apply the mask to get the filtered rows
    filteredTeam_rows = filteredHomeTeam_rows[mask]
    
    #Years of 1998 to 2018
    year_list = ['1998', '2002', '2006', '2010', '2014','2018']
    hex_vals = vectorized_convert(year_list)
    print(f'hex vals: {hex_vals}')
    mask = np.isin(filteredTeam_rows[:, 0], hex_vals)

    #Extracts the matches between the years of 1998 to 2018
    filteredTeamYear_rows = filteredTeam_rows[mask]
    decoded_data = vectorized_decode(filteredTeamYear_rows)

    print(decoded_data)

    #Had to remove , from Hong Kong, China - code was thinking Hong Kong and China were separate columns, causing error
    M = np.genfromtxt('./fifa_mens_rank.csv', skip_header=0, delimiter=',', dtype=str)

    mask = np.isin(M[:, 3], country_list)
    filtered_RankTeam = M[mask]
    mask = np.isin(filtered_RankTeam[:, 0], year_list)
    filtered_RankYearTeam = filtered_RankTeam[mask]
    print(filtered_RankYearTeam)
    for country in country_list:
         mask = np.isin(filtered_RankYearTeam[:, 3], country)
         rankTemp = filtered_RankYearTeam[mask]
         mask = np.isin(filtered_RankYearTeam[:, 3], country)
         rankTemp = filtered_RankYearTeam[mask]
         for year in year_list:
             mask = np.isin(rankTemp[:, 0], year)
             rank = rankTemp[mask]
             rankSemester = rank[:, 2]
             avgRank = int((int(rankSemester[0]) + int(rankSemester[1])) / 2)
            # for row in decoded_data:
               #   indices = np.where(country)[3] and np.where(year)[0]
            # for i in range(decoded_data.shape[0]):
            #      for j in range(decoded_data.shape[1]):
            #        if decoded_data[i][4] == country or decoded_data[i][5] == country and decoded_data[i][0] == year:
                            
    #M = np.genfromtxt('./FIFA - 2014.csv', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = decoded_data[:, 10]
    Xtrn = decoded_data[:, 4:]
    
    print(f'ytest: {ytrn}')
    print(f'Xtst: {Xtrn}')