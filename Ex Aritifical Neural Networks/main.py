To implement regression using an Artificial Neural Network (ANN) for a given dataset, we can use the Keras library (which is part of TensorFlow). The steps are as follows:

Import Required Libraries: We will need libraries like pandas, numpy, tensorflow, and matplotlib.
Prepare the Dataset: Load the dataset and preprocess it (e.g., normalization, splitting into training and test sets).
Build the ANN Model: Define the structure of the ANN with input layers, hidden layers, and an output layer.
Compile the Model: Choose an appropriate loss function and optimizer.
Train the Model: Fit the model on the training data.
Evaluate and Predict: Test the model's performance on the test data and make predictions.
Visualize the Results: Optionally, plot the results for better interpretation.
Here’s a Python implementation of regression using ANN for a sample dataset.


# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 2: Load the Dataset
# Example dataset: Let's use a dataset where 'X' is the input and 'y' is the output for regression
# Here we use a simple synthetic dataset for illustration purposes

# For a real-world scenario, replace the dataset with your own (e.g., pd.read_csv("data.csv"))
data = {'X': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'y': [1.2, 2.3, 3.1, 4.5, 5.6, 6.8, 7.4, 8.1, 9.0, 10.2]}
df = pd.DataFrame(data)

X = df[['X']].values  # Features
y = df['y'].values    # Target

# Step 3: Preprocess the Data
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data for better ANN performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Build the ANN Model
model = Sequential()

# Input layer (1 input feature) and 1 hidden layer with 8 neurons
model.add(Dense(8, input_dim=1, activation='relu'))

# Output layer (regression problem, so 1 neuron)
model.add(Dense(1))

# Step 5: Compile the Model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 6: Train the Model
history = model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)

# Step 7: Evaluate and Predict
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Predict the results for the test set
y_pred = model.predict(X_test)

# Step 8: Visualize the Results
# Plot the training loss curve
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Plot actual vs predicted values
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.title('Actual vs Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
