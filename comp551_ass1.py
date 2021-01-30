#%% Imports
import pandas as pd
import os
import numpy as np

#%%
hepatitis_filepath = os.getcwd() + "\\hepatitis.csv"
hepatitis_df = pd.read_csv(hepatitis_filepath)
clean_hep_df = hepatitis_df[-hepatitis_df.eq('?').any(1)]

cancer_filepath = os.getcwd() + "\\breast_cancer_wisconsin.csv"
cancer_df = pd.read_csv(cancer_filepath)
cancer_df.drop("id", axis=1, inplace=True)
clean_cancer_df = cancer_df[-cancer_df.eq('?').any(1)]

clean_cancer_df = clean_cancer_df.astype('int32')

#%% Data handling functions

## Takes a data frame and the a
def partition_dataset(dataframe, train_prop):
    data_size = dataframe.shape[0]
    training_amount = int(train_prop*data_size)
    
    # Before sampling, shuffle around the data:
    shuffled_df = dataframe.sample(frac=1)
    
    training_df = shuffled_df.iloc[0:training_amount]
    testing_df = shuffled_df.iloc[training_amount:data_size]
    
    return(training_df, testing_df)

#%% distance functions
def euclidean_dist(arr1, arr2):
    
    diff_vector = np.subtract(arr1, arr2)
    
    return np.sum(np.square(diff_vector))**0.5

#%%
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
    
    def predict_point(self, x):
        
        # Function that will be applied to all the points in the training data
        distance_to_x_fct = lambda a: self.dist_fct(a, x)
        
        # Distances between x and all the points in the training data
        distances = np.apply_along_axis(distance_to_x_fct, 1, self.x)
        
        prediction_indices = np.argsort(distances)[0:self.k]
        k_nearest_labels = self.y[prediction_indices]
        
        prediction_label = np.bincount(k_nearest_labels).argmax()
        certainty = np.count_nonzero(k_nearest_labels == prediction_label) / self.k
        
        return prediction_label, certainty
    
    def predict(self, x_matrix):
        
        predictions = np.apply_along_axis(self.predict_point, 0, x_matrix)
        
        return predictions
    
        
#%% Example using a signle point
training, testing = partition_dataset(clean_cancer_df, 0.75)

training_predictors = clean_cancer_df.drop('Class', axis=1).to_numpy()
training_labels = clean_cancer_df['Class'].to_numpy()

yeet = KNN(k=10)

yeet.fit(training_predictors, training_labels)

test_point = training_predictors[69]

print(yeet.predict_point(test_point))

#%%
def classification_cost(y_vector):
    counts = np.bincount(y_vector) 
    class_probs = counts / np.sum(counts)
    return 1 - np.max(class_probs)

def cost_entropy(labels):
    class_probs = np.bincount(labels) / len(labels)
    class_probs = class_probs[class_probs > 0]              g
    return -np.sum(class_probs * np.log(class_probs))



#%%
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
            
            if left_cost + right_cost < best_cost:
                best_cost = left_cost + right_cost
                best_feature = predictor
                best_value = value
                best_left = left_indices
                best_right = right_indices
    
    return best_feature, best_value, best_cost, best_left, best_right
    

class Node:
    
    def __init__(self, x_matrix, y_vector, cost_fct, depth, max_depth, parent, improvement_cutoff=0.03):
        
        assert depth <= max_depth
        
        self.depth=depth
        
        self.left_child = None
        self.right_child = None
        self.parent = None
        self.tested_feature = None
        self.critical_value = None
        self.x = x_matrix
        self.y = y_vector
        self.cost = cost_fct(y_vector)
        
        # i.e. this node is a leaf
        if depth == max_depth:
            self.x = x_matrix
            self.y = y_vector
            self.parent = parent
            return
        
        best_feature, best_value, partition_cost, left_indices, right_indices  = find_best_partition(x_matrix, y_vector, cost_fct)
        
        # if we couldn't find a legit partition, we make this a leaf
        if best_feature is None:
            self.x = x_matrix
            self.y = y_vector
            self.parent = parent
            return
        
        # again, this is a leaf
        if self.cost - partition_cost < improvement_cutoff:
            self.x = x_matrix
            self.y = y_vector
            self.parent = parent
            return
        
        # if we make it here, this means we have an internal node
        self.left_child = Node(x_matrix[left_indices,:], y_vector[left_indices], cost_fct, self.depth+1, max_depth, self)
        self.right_child = Node(x_matrix[right_indices,:], y_vector[right_indices], cost_fct, self.depth+1, max_depth, self)
        self.tested_feature = best_feature
        self.critical_value = best_value
        self.parent = parent
    
    
    def predict(self, x_vector):
        
        # If this node is a leaf, we need to predict
        if self.left_child is None and self.right_child is None:
            counts = np.bincount(self.y) 
            prediction = np.argmax(counts)
            certainty = np.max(counts / np.sum(counts))
            return prediction, certainty

        # if not, delegate the call to the children
        else:
            if x_vector[self.tested_feature] <= self.critical_value:
                return self.left_child.predict(x_vector)
            else:
                return self.right_child.predict(x_vector)
    
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
        
    


#%%
noot = Node(training_predictors, training_labels, classification_cost, 0, 10, None)

test_point = np.array([1,1,1,1,1,1,1,1,1,1])

leaves = noot.get_leaf_list()

