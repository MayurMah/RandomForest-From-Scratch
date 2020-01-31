from util import entropy, information_gain, partition_classes
import numpy as np
import ast

class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        #self.tree = []
        self.tree = {}
        #pass

    def learn(self, X, y):
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in utils.py to train the tree

        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)
        #print("learning...")
        max_depth = 10
        min_size = max(int(0.01*len(y)),1) #1% of total data

        self.tree = self.split(X,y,max_depth,min_size,1)

        #print("tree built...")
        #print(self.tree)

    def split(self,X,y,max_depth, min_size, depth):

        if(len(X)<=min_size):
            node = {'index':depth,'split_attribute':-1,'split_val':self.to_terminal(y),'left':{},'right':{}}
            return node

        if(depth>=max_depth):
            node = {'index':depth,'split_attribute':-1,'split_val':self.to_terminal(y),'left':{},'right':{}}
            return node

        #check all columns to find the column to split on
        max_info_gain = -1
        max_info_gain_col = -1
        max_info_gain_col_val =  -1
        split_type = ''

        for i in range(len(X[0])):
            #partition X & y based on i
            cur_split_type = ''
            currentcol = [row[i] for row in X]
            #print("here..",currentcol,"i:",i)

            isnumeric = True

            try:
                float(currentcol[0])
            except ValueError:
                isnumeric=False

            if(isnumeric):
                #print(currentcol)
                split_val = np.mean([float(x) for x in currentcol])
                cur_split_type = 'continuous'
                #print(cur_split_type)
            else:
                split_val = max(set(currentcol),key=currentcol.count)
                cur_split_type = 'categorical'
                #print("inside split")
                #print(cur_split_type)
                #print(currentcol[0:5])
                #print(split_val)

            #split_attr, split_val decided
            #print("before partition...i, split_val :",i, split_val)
            X_left, X_right, y_left, y_right  = partition_classes(X, y, i, split_val)

            #print("partitioned..")
            #print(X_left, X_right, y_left, y_right)

            #calculate info gain
            current_info_gain = information_gain(y, [y_left,y_right])

            if(max_info_gain<current_info_gain):
                max_info_gain = current_info_gain
                max_info_gain_col = i
                max_info_gain_col_val = split_val
                split_type = cur_split_type

        #print("node created..",max_info_gain,max_info_gain_col,max_info_gain_col_val,split_type)

        left_node = DecisionTree()
        right_node = DecisionTree()
        node = dict({'index':depth,'split_attribute':max_info_gain_col,'split_val':max_info_gain_col_val,'split_type':split_type,'left':left_node,'right':right_node})
        self.tree.update(node)
        #print("node created at ",depth)
        #print(self.tree)
        self.tree['left'] =  left_node.split(X_left,y_left,max_depth, min_size, depth+1)
        self.tree['right'] = right_node.split(X_right,y_right,max_depth, min_size, depth+1)
        return self.tree

    def to_terminal(self,y):
        if(y.count(0)>=y.count(1)):
            return 0
        else:
            return 1

    def classify(self, record):
        # TODO: classify the record using self.tree and return the predicted label
        #print("inside classify")
        #print("X:",record)
        #print(self.tree)
        pred = self.parse_tree(self.tree,record)
        #print("predicted...")
        #print(pred)
        return pred

    def parse_tree(self,node,X):
        #print("inside parse_tree, X:",X)
        #print("inside parse_tree, node:",node)
        #print("inside parse_tree, split_attribute, split_val:",node['split_attribute'],node['split_val'])

        if(node['split_attribute']==-1):
            return node['split_val']

        if(node['split_type']=='continuous'):
            if(X[node['split_attribute']]<=node['split_val']):
                return self.parse_tree(node['left'],X)
            else:
                return self.parse_tree(node['right'],X)
        if(node['split_type']=='categorical'):
            if(X[node['split_attribute']]==node['split_val']):
                return self.parse_tree(node['left'],X)
            else:
                return self.parse_tree(node['right'],X)

        #pass

#d = DecisionTree()
#X = [[3, 'aa', 10],
#         [1, 'bb', 22],
#         [2, 'cc', 28],
#         [5, 'bb', 32],
#         [4, 'cc', 32]]

#y = [1,1,0,0,1]

#d.learn(X,y)
#pred = d.classify([5, 'bb', 32])
#print("predicted:")
#print(pred)
