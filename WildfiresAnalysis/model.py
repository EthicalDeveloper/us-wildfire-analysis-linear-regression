import pandas as pd
import numpy as np
import copy
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Read the Excel file
df = pd.read_csv('Data/US_Lightning_Forest_Fires_Train.csv')

# Assuming the columns you want to use are named 'feature1', 'feature2', and 'feature3'
# Replace these names with the actual column names from your Excel file
features = ['FIRE_SIZE', 'FIRE_SIZE_CLASS', 'STATE']
target = 'Days_to_extinguish_fire'


# Handle categorical data by encoding
# Example: If 'FIRE_SIZE_CLASS' is categorical
# A: 0 B: 1 C: 2 D: 3 E: 4 F: 5 G: 6
if df['FIRE_SIZE_CLASS'].dtype == 'object':
    le_fire_size = LabelEncoder()
    df['FIRE_SIZE_CLASS'] = le_fire_size.fit_transform(df['FIRE_SIZE_CLASS'])

# Handle categorical data by encoding
# Example: If 'STATE' is categorical
if df['STATE'].dtype == 'object':
    le_state = LabelEncoder()
    df['STATE'] = le_state.fit_transform(df['STATE'])


# # Print the mapping of labels to integers
# for category, label in zip(le_state.classes_, le_state.transform(le_state.classes_)):
#     print(f'{category}: {label}')

# Select the columns and convert to NumPy array
X_train = df[features].to_numpy()
y_train = df[target].to_numpy()

# Apply StandardScaler to input
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X_train)


# Apply normalization to output
# Normalization
arr_min = np.min(y_train)
arr_max = np.max(y_train)

y_standardized = (y_train - arr_min) / (arr_max - arr_min)


b_init = 235
w_init = np.array([0.3,18,-53])

def plot_data():

    fire_size = X_standardized[:,0]
    fire_class = X_standardized[:,1]
    fire_state = X_standardized[:,2]

    # Set axis limits if needed
    plt.xlim(min(fire_size), max(fire_size))
    plt.ylim(min(y_standardized), max(y_standardized))
    # Create the scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(fire_size, y_standardized, color='blue', label='# Of Days')


    # Add labels and title
    plt.xlabel('Fire Sizes')
    plt.ylabel('# Of Days')
    plt.title(f'Relationship between Feature {2} and Target Values')
    plt.legend()


    # Show the plot
    plt.grid(True)
    # Show plot
    plt.show()




def compute_cost_linear(X,y,w,b):
    # Define number of training examples
    m = X.shape[0]
    cost=0.0

    for i in range(m):
        f_wb_i = np.dot(X[i],w) + b
        cost = cost + (f_wb_i - y[i])**2
    
    cost = cost / (2 * m)
    return cost




def compute_gradient_linear(X,y,w,b):
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = (np.dot(X[i],w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i][j]
        dj_db = dj_db + err
    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_db, dj_dw



def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    # Array of all costs that was obtained through iterations
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        # Calculate the gradient
        dj_db, dj_dw = compute_gradient_linear(X, y, w, b)

        # Updated weight and bias simultaneously
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration

        if i < 100000:    # this is just to prevent resource exaustion
            J_history.append(compute_cost_linear(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print (f"Iteration {i:4d}: Cost {J_history[-1]} | Gradient: w = {w} b = {b} ")


    return w, b, J_history     # return final w, b, J history




# w_tmp = np.zeros_like(X_standardized[0])
# b_tmp = 0.
# alph = 0.01
# iters = 10000

# w_out, b_out, _ = gradient_descent(X_standardized, y_standardized, w_tmp, b_tmp, alph, iters)

# print (f"\nUpdated Parameters: w:{w_out}, b:{b_out}")


# # Updated Parameters: w:[ 0.00667911  0.01330238 -0.00132519], b:0.010515863738268145

def predict_normalized(X, w, b):
    m = X.shape[0]  # Number of training examples
    y_pred_norm = np.zeros(m)
    for i in range(m):
        y_pred_norm[i] = np.dot(X[i], w) + b  # Compute the linear combination
        

    return y_pred_norm

def prediction_unnormalize(X):
    m = X.shape[0]  # Number of training examples
    pred_unnormalize = np.zeros(m)

    for i in range(m):
        pred_unnormalize[i] = round(X[i] * (arr_max - arr_min) + arr_min)        


    return pred_unnormalize

def accuracy_normalized(y_true, y_pred):
    # Ensure that y_true and y_pred are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # round to 2 decimal places
    y_true = np.round(y_true, 2)
    y_pred = np.round(y_pred, 2)

    # Calculate the number of correct predictions
    correct_predictions = np.sum(y_true == y_pred)
    
    # Calculate accuracy as a percentage
    accuracy = (correct_predictions / len(y_true)) * 100
    
    return accuracy

def accuracy(y_true, y_pred):
    # Ensure that y_true and y_pred are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    y_pred = np.round(y_pred)

    # Calculate the number of correct predictions
    correct_predictions = np.sum(y_true == y_pred)
    
    # Calculate accuracy as a percentage
    accuracy = (correct_predictions / len(y_true)) * 100
    
    return accuracy

w = np.array([0.00667911,0.01330238,-0.00132519])
b = 0.010515863738268145
m = X_train.shape[0]

pred_array = predict_normalized(X_standardized, w, b)

unnormalized_prediction = prediction_unnormalize(pred_array)

#!!! Calculate the normalized prediction accuracy

#!!! Calculate the normalized prediction accuracy

for i in range(m):
    if (round(unnormalized_prediction[i]) == y_train[i]):
        print(f"Match: Predicted: {round(unnormalized_prediction[i])} Actual: {y_train[i]}")
    else:
        print(f"No match: Predicted: {round(unnormalized_prediction[i])} Actual: {y_train[i]}")


print (f"Accuracy: {accuracy(y_train,unnormalized_prediction)}")


# for i in range(m):
#     if (round(pred_array[i],2) == round(y_standardized[i],2)):
#         print(f"Match: Predicted: {pred_array[i]} Actual: {y_standardized[i]}")
#     else:
#         print(f"No match: Predicted: {pred_array[i]} Actual: {y_standardized[i]}")


# print (f"Accuracy: {accuracy(y_standardized,pred_array)}")