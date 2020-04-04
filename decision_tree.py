from util import entropy, information_gain, partition_classes
import numpy as np
import ast


class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list
        # self.tree = []
        self.tree = {}
        # pass

    def learn(self, X, y):
        """Train the decision tree (self.tree) using the the sample X and labels y

        Args:
            X(list of list): features/attributes
            y(list): labels

        Returns:
            Learned RandomForest Model (tree)

        """

        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a
        #    'right' key. We can add more keys which might help in classification
        #    (eg. split attribute and split value)

        max_depth = 10
        min_size = max(int(0.01 * len(y)), 1)  # 1% of total data

        self.tree = self.split(X, y, max_depth, min_size, 1)

    def split(self, X, y, max_depth, min_size, depth):
        """Split the tree on the attribute that provides the maximum info gain

        Args:
            X(list of list): data containing all attributes/features
            y(list): labels
            max_depth(int): max allowed depth
            min_size(int): minimum number of data points needed to be in a node in order to split
            depth(int): current depth

        Returns:
            Updated tree with the split performed

        """

        if len(X) <= min_size:
            node = {'index': depth, 'split_attribute': -1, 'split_val': self.to_terminal(y), 'left': {}, 'right': {}}
            return node

        if depth >= max_depth:
            node = {'index': depth, 'split_attribute': -1, 'split_val': self.to_terminal(y), 'left': {}, 'right': {}}
            return node

        # check all columns to find the column to split on
        max_info_gain = -1
        max_info_gain_col = -1
        max_info_gain_col_val = -1
        split_type = ''

        for i in range(len(X[0])):
            # partition X & y based on i
            cur_split_type = ''
            currentcol = [row[i] for row in X]

            isnumeric = True

            try:
                float(currentcol[0])
            except ValueError:
                isnumeric = False

            if isnumeric:

                split_val = np.mean([float(x) for x in currentcol])
                cur_split_type = 'continuous'
            else:
                split_val = max(set(currentcol), key=currentcol.count)
                cur_split_type = 'categorical'

            # split_attr, split_val decided

            X_left, X_right, y_left, y_right = partition_classes(X, y, i, split_val)

            # calculate info gain
            current_info_gain = information_gain(y, [y_left, y_right])

            if max_info_gain < current_info_gain:
                max_info_gain = current_info_gain
                max_info_gain_col = i
                max_info_gain_col_val = split_val
                split_type = cur_split_type

        left_node = DecisionTree()
        right_node = DecisionTree()
        node = dict({'index': depth, 'split_attribute': max_info_gain_col, 'split_val': max_info_gain_col_val,
                     'split_type': split_type, 'left': left_node, 'right': right_node})
        self.tree.update(node)
        self.tree['left'] = left_node.split(X_left, y_left, max_depth, min_size, depth + 1)
        self.tree['right'] = right_node.split(X_right, y_right, max_depth, min_size, depth + 1)
        return self.tree

    def to_terminal(self, y):
        if y.count(0) >= y.count(1):
            return 0
        else:
            return 1

    def classify(self, record):
        """Classify the record using self.tree and return the predicted label

        Args:
            record(list): new data point to be classified

        Returns:
            Predicted label for the given record (0 or 1)

        """

        pred = self.parse_tree(self.tree, record)
        return pred

    def parse_tree(self, node, X):
        """Recursively parse the tree until the label for the given record is found

        Args:
            node(tree): current node
            X(list): given record

        Returns:
            Split value of the node

        """

        if node['split_attribute'] == -1:
            return node['split_val']

        if node['split_type'] == 'continuous':
            if X[node['split_attribute']] <= node['split_val']:
                return self.parse_tree(node['left'], X)
            else:
                return self.parse_tree(node['right'], X)
        if node['split_type'] == 'categorical':
            if X[node['split_attribute']] == node['split_val']:
                return self.parse_tree(node['left'], X)
            else:
                return self.parse_tree(node['right'], X)

# Testing code
# d = DecisionTree()
# X = [[3, 'aa', 10],
#         [1, 'bb', 22],
#         [2, 'cc', 28],
#         [5, 'bb', 32],
#         [4, 'cc', 32]]

# y = [1,1,0,0,1]

# d.learn(X,y)
# pred = d.classify([5, 'bb', 32])
# print("predicted:")
# print(pred)
