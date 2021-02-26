#%% Imports
import pandas as pd
import os
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#%% distance functions
def euclidean_dist(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2, axis=-1))

def manhattan_dist(x1, x2):
    return np.sum(np.abs(x1 - x2), axis=-1)

def minkowski_dist(x1, x2, power=3):
    return np.power((np.sum(np.abs(x1 - x2)**power, axis=-1)), 1/power)


#%% KNN
class KNN:
    
    def __init__(self, k=1, distance_fct=euclidean_dist):
        self.k = k
        self.dist_fct = distance_fct
        self.x = None
        self.y = None
    
    # MAKE SURE TO PASS IN MATRICES/VECTORS, NOT DATAFRAMES
    def fit(self, x_matrix, y_vector):
        self.x = x_matrix
        self.y = y_vector
    
    
    def predict(self, x_test):
        
        num_test = x_test.shape[0]

        distances = self.dist_fct(self.x[None,:,:], x_test[:,None,:])

        knns = np.zeros((num_test, self.k), dtype=int)

        predictions = np.zeros((num_test, 2))
        for i in range(num_test):
            knns[i,:] = np.argsort(distances[i])[:self.k]

            predictions[i, 0] = np.bincount(self.y[np.argsort(distances[i])[:self.k]]).argmax()
            predictions[i, 1] = np.bincount(self.y[np.argsort(distances[i])[:self.k]]).max() / self.k
        
        return predictions
    
    def get_x(self):
        return self.x
    
    def get_y(self):
        return self.y
        
    

#%% Cost functions for evaluating partitions
def classification_cost(y_vector):
    counts = np.bincount(y_vector) 
    class_probs = counts / np.sum(counts)
    return 1 - np.max(class_probs)

def cost_entropy(labels):
    class_probs = np.bincount(labels) / len(labels)
    class_probs = class_probs[class_probs > 0]              
    return -np.sum(class_probs * np.log(class_probs))

def cost_gini_index(labels):
    class_probs = np.bincount(labels) / len(labels)
    return 1 - np.sum(np.square(class_probs)) 


#%% Decision tree
# Helper function that performs the partition given the feature number and the line to split at
def perform_partition(x_matrix, feature, critical_value):
    feature_vector = x_matrix[:,feature]
    
    all_indices = np.arange(0, feature_vector.shape[0])
    left_indices = np.argwhere(feature_vector <= critical_value).flatten()
    right_indices = np.setdiff1d(all_indices, left_indices)

    
    return left_indices, right_indices

# Helper function to fin the best partition
def find_best_partition(x_matrix, y_vector, cost_fct):
    observation_amount, predictor_amount = x_matrix.shape
    
    best_cost = np.inf
    best_feature = None
    best_value = None
    best_left = None
    best_right = None
        
    for predictor in range(predictor_amount):
        for value in range(observation_amount):
            left_indices, right_indices = perform_partition(x_matrix, predictor, value)
            
            # Avoiding partitions that do nothing
            if left_indices.shape[0] == 0 or right_indices.shape[0] == 0:
                continue
            
            total_nodes = x_matrix.shape[0]
            left_cost = left_indices.shape[0]/total_nodes*cost_fct(y_vector[left_indices])
            right_cost = right_indices.shape[0]/total_nodes*cost_fct(y_vector[right_indices])
            
            if left_cost + right_cost <= best_cost:
                best_cost = left_cost + right_cost
                best_feature = predictor
                best_value = value
                best_left = left_indices
                best_right = right_indices
    
    return best_feature, best_value, best_cost, best_left, best_right
    
testx =np.array([[1],[2],[3],[4]])
testy = np.array([1,2,1,2])

print(find_best_partition(testx, testy, classification_cost))
class Node:
    
    def __init__(self, x_matrix, y_vector, cost_fct, depth, max_depth, parent, improvement_cutoff):
        
        assert depth <= max_depth
        
        self.depth=depth
        
        self.left_child = None
        self.right_child = None
        self.parent = parent
        self.tested_feature = None
        self.critical_value = None
        self.x = x_matrix
        self.y = y_vector
        self.cost = cost_fct(y_vector)
        
        # i.e. this node is a leaf
        if depth == max_depth:
            return
        
        best_feature, best_value, partition_cost, left_indices, right_indices  = find_best_partition(x_matrix, y_vector, cost_fct)
        
        # if we couldn't find a legit partition, we make this a leaf
        if best_feature is None:
            return
        
        # again, this is a leaf
        if self.cost - partition_cost <= improvement_cutoff:
            return
        
        # if we make it here, this means we have an internal node
        self.left_child = Node(x_matrix[left_indices,:], y_vector[left_indices], cost_fct, self.depth+1, max_depth, self, improvement_cutoff)
        self.right_child = Node(x_matrix[right_indices,:], y_vector[right_indices], cost_fct, self.depth+1, max_depth, self, improvement_cutoff)
        self.tested_feature = best_feature
        self.critical_value = best_value
    
    
    # predicts a single label for some x-vector
    def predict_point(self, x_vector):
        # If this node is a leaf, we need to predict
        if self.left_child is None and self.right_child is None:
            counts = np.bincount(self.y) 
            prediction = np.argmax(counts)
            certainty = np.max(counts / np.sum(counts))
            return prediction, certainty

        # if not, delegate the call to the children
        else:
            if x_vector[self.tested_feature] <= self.critical_value:
                return self.left_child.predict_point(x_vector)
            else:
                return self.right_child.predict_point(x_vector)
    
    def predict(self, x_matrix):
        predictions = np.apply_along_axis(self.predict_point, 1, x_matrix)
        
        return predictions
    
    
    def get_height(self):
        if self.left_child is None and self.right_child is None:
            return 0
        else:
            return 1 + max(self.left_child.get_height(), self.right_child.get_height())
    
    
    def get_leaf_list(self):
         if self.left_child is None and self.right_child is None:
            return [self]
         else:
            return [] + self.left_child.get_leaf_list() + self.right_child.get_leaf_list()
    
    def get_internal_list(self):
        # If the node is a leaf
        if self.left_child is None and self.right_child is None:
            return []
        else:
            return [self] + self.left_child.get_internal_list() + self.right_child.get_internal_list()
    
    def get_all_nodes(self):
        return self.get_leaf_list() + self.get_internal_list()

    def convert_to_leaf(self):
        self.left_child = None
        self.right_child = None
        self.tested_feature = None
        self.critical_value = None


class DecisionTree:
    def __init__(self, cost_fct, max_depth, improvement_cutoff = 0):
        self.cost_fct = cost_fct
        self.max_depth = max_depth
        self.improvement_cutoff = improvement_cutoff
        self.root = None
    
    def fit(self, x_matrix, y_vector):
        self.root = Node(x_matrix, y_vector, self.cost_fct, 0, self.max_depth, None, self.improvement_cutoff)
    
    def predict(self, x_matrix):
        return self.root.predict(x_matrix)
    
    # returns a copy of the tree, with the least costly node converted into a leaf
    def prune_best_node(self):
        tree_copy = copy.deepcopy(self)
        int_nodes = tree_copy.root.get_internal_list()
        n = tree_copy.root.y.shape[0]
        
        # Sort the nodes by the smallest increase in cost
        int_nodes.sort(key= lambda node: node.cost*node.y.shape[0]/n)
        
        
        int_nodes[0].convert_to_leaf()
        return tree_copy

    #returns a list of trees, each one being a pruned version of the last
    # ALSO RETURNS THE INITIAL TREE
    def prune(self):
        if len(self.root.get_internal_list()) == 0:
            return [self]
        else:
            pruned_once = self.prune_best_node()
            return [self] + pruned_once.prune()
    
    # gets the cost of the whole 
    def get_cost(self):
        leaves = self.root.get_leaf_list()
        total_cost = 0
        n = self.root.y.shape[0]
    
        for leaf in leaves:
            total_cost += leaf.cost*leaf.y.shape[0]/n
        
        return total_cost
    
    def get_x(self):
        return self.root.x
    
    def get_y(self):
        return self.root.y

#%% Testing code
def evaluate_acc(predicted_ys, true_ys):
    success_vector = np.equal(predicted_ys, true_ys)
    
    return np.count_nonzero(success_vector)/success_vector.shape[0]

          
def test_KNN_diff_k(training_x, training_y, testing_x, testing_y, k_list, distance_fct):
    output = []
    for k in k_list:
        temp_model = KNN(k=k, distance_fct=distance_fct)
        temp_model.fit(training_x, training_y)
        temp_preds = temp_model.predict(testing_x)[:,0]
        accuracy = evaluate_acc(temp_preds, testing_y)
        output.append([k, accuracy])
    
    return np.array(output)

# NO PRUNING HAPPENS HERE
def test_DTree_diff_depth(training_x, training_y, testing_x, testing_y, depth_list, cost_fct, improvement_cutoff=0):
    output = []
    for d in depth_list:
        temp_model = DecisionTree(cost_fct, d, improvement_cutoff=improvement_cutoff)
        temp_model.fit(training_x, training_y)
        temp_preds = temp_model.predict(testing_x)[:,0]
        accuracy = evaluate_acc(temp_preds, testing_y)
        output.append([d, accuracy])
        
    return np.array(output)

# Produces the data points to create the graph on the pruning slide
def test_DTree_pruning(training_x, training_y, testing_x, testing_y, d, cost_fct, improvement_cutoff=0):
    dtree = DecisionTree(cost_fct, d, improvement_cutoff=improvement_cutoff)
    dtree.fit(training_x, training_y)
    
    pruned_trees = dtree.prune()
    output = []
    for tree in pruned_trees:
        cost = evaluate_acc(tree.predict(testing_x)[:,0], testing_y)
        output.append([len(tree.root.get_internal_list()), 1 -cost])
    
    return np.array(output)



#%% Decision boundary plotting function


def plot_decision_boundaries(model, xlabel, ylabel):
    x = model.get_x()
    y = model.get_y()
    
    x0v = np.linspace(np.min(x[:,0]), np.max(x[:,0]), 200)
    x1v = np.linspace(np.min(x[:,1]), np.max(x[:,1]), 200)
    
    #to features values as a mesh  
    x0, x1 = np.meshgrid(x0v, x1v)
    x_all = np.vstack((x0.ravel(),x1.ravel())).T
    
    plane_preds = model.predict(x_all)

    plt.scatter(x=x[:,0], y=x[:,1], c=y, marker='o')
    plt.scatter(x_all[:,0], x_all[:,1], c=plane_preds[:,0], marker='.', alpha=0.01)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()



