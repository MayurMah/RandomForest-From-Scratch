from scipy import stats
import numpy as np


def entropy(class_y):
    """Compute entropy for the given list of classes (to be used in the calculation of information gain)

    Args:
        class_y(list): list of class labels (0's and 1's)

    Returns:
        Entropy (float)
    """

    # Example:
    #    entropy([0,0,0,1,1,1,1,1,1]) = 0.92

    entropy = 0

    class_y = list(map(int, class_y))

    if len(class_y) == 0:
        return entropy

    p = (class_y.count(1)) / len(class_y)
    q = class_y.count(0) / len(class_y)

    if p != 0 and q != 0:
        entropy = -p * np.log2(p) - q * np.log2(q)
    elif p == 0:
        entropy = -q * np.log2(q)
    else:
        entropy = -p * np.log2(p)

    return entropy


def partition_classes(X, y, split_attribute, split_val):
    """Partition the data(X) and labels(y) based on the split value - BINARY SPLIT.

    Args:
        X(list of list): data containing all attributes/features
        y(list) : labels
        split_attribute(int): column index of the attribute to split on
        split_val: either a numerical or categorical value to divide the split_attribute

    Returns:

    """

    # We will have to first check if the split attribute is numerical or categorical
    # If the split attribute is numeric, split_val should be a numerical value
    # For example, your split_val could be the mean of the values of split_attribute
    # If the split attribute is categorical, split_val should be one of the categories.
    #
    # We can perform the partition in the following way
    # Numeric Split Attribute:
    #   Split the data X into two lists(X_left and X_right) where the first list has all
    #   the rows where the split attribute is less than or equal to the split value, and the
    #   second list has all the rows where the split attribute is greater than the split
    #   value. Also create two lists(y_left and y_right) with the corresponding y labels.
    #
    # Categorical Split Attribute:
    #   Split the data X into two lists(X_left and X_right) where the first list has all
    #   the rows where the split attribute is equal to the split value, and the second list
    #   has all the rows where the split attribute is not equal to the split value.
    #   Also create two lists(y_left and y_right) with the corresponding y labels.

    '''
    Example:

    X = [[3, 'aa', 10],                 y = [1,
         [1, 'bb', 22],                      1,
         [2, 'cc', 28],                      0,
         [5, 'bb', 32],                      0,
         [4, 'cc', 32]]                      1]

    Here, columns 0 and 2 represent numeric attributes, while column 1 is a categorical attribute.

    Consider the case where we call the function with split_attribute = 0 and split_val = 3 (mean of column 0)
    Then we divide X into two lists - X_left, where column 0 is <= 3  and X_right, where column 0 is > 3.

    X_left = [[3, 'aa', 10],                 y_left = [1,
              [1, 'bb', 22],                           1,
              [2, 'cc', 28]]                           0]

    X_right = [[5, 'bb', 32],                y_right = [0,
               [4, 'cc', 32]]                           1]

    Consider another case where we call the function with split_attribute = 1 and split_val = 'bb'
    Then we divide X into two lists, one where column 1 is 'bb', and the other where it is not 'bb'.

    X_left = [[1, 'bb', 22],                 y_left = [1,
              [5, 'bb', 32]]                           0]

    X_right = [[3, 'aa', 10],                y_right = [1,
               [2, 'cc', 28],                           0,
               [4, 'cc', 32]]                           1]

    '''

    X_np = np.array(X)
    y_np = np.array([y])

    Xy_np = np.concatenate((X_np, y_np.T), axis=1)

    X_left = []
    X_right = []

    y_left = []
    y_right = []

    isnumeric = True

    try:
        float(Xy_np[0, split_attribute])
    except ValueError:
        isnumeric = False

    if isnumeric:  # continuous case
        X_left = Xy_np[Xy_np[:, split_attribute].astype(float) <= float(split_val), :]
        X_right = Xy_np[Xy_np[:, split_attribute].astype(float) > float(split_val), :]
        y_left = X_left[:, -1].tolist()
        y_right = X_right[:, -1].tolist()
        X_left = X_left[:, :-1].tolist()
        X_right = X_right[:, :-1].tolist()
    else:  # categorical case
        X_left = Xy_np[Xy_np[:, split_attribute] == split_val, :]
        X_right = Xy_np[Xy_np[:, split_attribute] != split_val, :]
        y_left = X_left[:, -1].tolist()
        y_right = X_right[:, -1].tolist()
        X_left = X_left[:, :-1].tolist()
        X_right = X_right[:, :-1].tolist()

    for i in range(len(X_left)):
        for j in range(len(X_left[i])):
            if X_left[i][j].isnumeric():
                X_left[i][j] = float(X_left[i][j])

    for i in range(len(X_right)):
        for j in range(len(X_right[i])):
            if X_right[i][j].isnumeric():
                X_right[i][j] = float(X_right[i][j])

    y_left = list(map(int, y_left))
    y_right = list(map(int, y_right))

    return (X_left, X_right, y_left, y_right)


def information_gain(previous_y, current_y):
    """Compute and return the information gain from partitioning the previous_y labels into the current_y labels.

    Args:
        previous_y(list): the distribution of original labels (0's and 1's)
        current_y: the distribution of labels after splitting based on a particular
                   split attribute and split value

    Returns:
        Information gain from the partitioning (float)

    """

    """
    Example:

    previous_y = [0,0,0,1,1,1]
    current_y = [[0,0], [1,1,1,0]]

    info_gain = 0.45915
    """

    info_gain = 0

    H = entropy(previous_y)
    HL = entropy(current_y[0])
    HR = entropy(current_y[1])

    info_gain = H - (HL * len(current_y[0]) / len(previous_y) + HR * len(current_y[1]) / len(previous_y))

    return info_gain
