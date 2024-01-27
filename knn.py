from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import GridSearchCV
# save the model
import pickle
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import matplotlib.pyplot as plt

# Custom function to convert stringified lists to numpy arrays
def convert_to_array(str_list):
    return np.array(json.loads(str_list))

def compute_tdoas(arrival_times):
    # just do 3 tdoas for now
    tdoas = []
    tdoas.append(arrival_times[0] - arrival_times[1])
    tdoas.append(arrival_times[0] - arrival_times[2])
    tdoas.append(arrival_times[0] - arrival_times[3])
    return tdoas


# Load data from CSV and preprocess
def load_data_from_csv(file_path):
    data = pd.read_csv(file_path)

    # Converting string lists to numpy arrays
    for column in data.columns:
        data[column] = data[column].apply(convert_to_array)

    # Compute TDOAs
    data['arrival_times'] = data['arrival_times'].apply(compute_tdoas)

    # Split the data into features and target
    X = data[['stations', 'arrival_times']]
    y = data['source']

    # Flatten the data
    X = np.array([np.concatenate([x['stations'], x['arrival_times']]) for _, x in X.iterrows()])
    y = np.array(y.tolist())

    return X, y

# X_train, y_train = load_data_from_csv('test6.csv')
# X_test, y_test = load_data_from_csv('test3.csv')

# # Split the data into training and testing sets
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Scale the features
# scaler = MinMaxScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Initialize KNN regressor
# knn = KNeighborsRegressor()



# param_grid = {
#     'n_neighbors': [3, 5, 7, 10, 15],
#     'weights': ['uniform', 'distance'],
#     'metric': ['euclidean', 'manhattan'],
#     'leaf_size': [10, 30, 50, 100],
#     'p': [1, 2, 3]
# }

# # Initialize GridSearchCV with a higher verbose level
# grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='r2', verbose=1, n_jobs=-1)

# grid_search.fit(X_train_scaled, y_train)

# # Best parameters and best score
# print("Best Parameters:", grid_search.best_params_)
# print("Best R^2 Score:", grid_search.best_score_)

# # Evaluate the best model on the test set
# best_knn = grid_search.best_estimator_
# test_score = best_knn.score(X_test_scaled, y_test)
# print(f"Test Set R^2 Score: {test_score}")


# filename = 'knn_tdoa_model.pkl'
# pickle.dump(best_knn, open(filename, 'wb'))

# # save the scaler
# filename = 'scaler.pkl'
# pickle.dump(scaler, open(filename, 'wb'))

# import the model and scaler and test on a few examples and plot the results
knn_tdoa = pickle.load(open('knn_tdoa_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Load data from CSV and preprocess
x, y = load_data_from_csv('test4.csv')

# Scale the features
x_scaled = scaler.transform(x)

# Make predictions
predictions = knn_tdoa.predict(x_scaled)

# the difference between the predictions and the actual values
diff = predictions - y

plt.figure(figsize=(15, 5))
for i, label in enumerate(['x', 'y', 'z']):
    plt.subplot(1, 3, i+1)
    plt.scatter(y[:, i], predictions[:, i], alpha=0.5)
    plt.plot([y[:, i].min(), y[:, i].max()], [y[:, i].min(), y[:, i].max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs. Predicted {label}')
plt.tight_layout()
plt.show()

# Histogram of Differences
plt.figure(figsize=(15, 5))
for i, label in enumerate(['x', 'y', 'z']):
    plt.subplot(1, 3, i+1)
    plt.hist(predictions[:, i] - y[:, i], bins=30, alpha=0.7)
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    plt.title(f'Error Distribution in {label}')
plt.tight_layout()
plt.show()
