Question:
You are working on a project where you need to analyze the correlation between the amount of time students spend studying and their exam scores. 
Discuss how you would use Python and linear regression to model this relationship and make predictions.

//main.py//
# Step 1: Import Required Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Define the dataset (Study Time vs Exam Scores)
study_time = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)  # Features (Study time in hours)
exam_scores = np.array([50, 55, 60, 65, 70, 75])  # Target (Exam scores)

# Step 3: Visualize the data with a scatter plot
plt.scatter(study_time, exam_scores, color='blue', label='Data Points')
plt.title('Study Time vs Exam Scores')
plt.xlabel('Study Time (hours)')
plt.ylabel('Exam Scores')
plt.show()

# Step 4: Split the data into training and test sets (Optional)
X_train, X_test, y_train, y_test = train_test_split(study_time, exam_scores, test_size=0.2, random_state=42)

# Step 5: Create the Linear Regression model
model = LinearRegression()

# Step 6: Train the model on the training data
model.fit(X_train, y_train)

# Step 7: Get the model parameters (slope and intercept)
slope = model.coef_[0]
intercept = model.intercept_
print(f"Slope (β1): {slope}")
print(f"Intercept (β0): {intercept}")

# Step 8: Make predictions using the model
y_pred = model.predict(X_test)

# Step 9: Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Step 10: Plot the regression line along with the data points
plt.scatter(study_time, exam_scores, color='blue', label='Data Points')
plt.plot(study_time, model.predict(study_time), color='red', label='Regression Line')
plt.title('Study Time vs Exam Scores with Regression Line')
plt.xlabel('Study Time (hours)')
plt.ylabel('Exam Scores')
plt.legend()
plt.show()

# Step 11: Make Predictions (e.g., Predict the exam score for 7 hours of study)
predicted_score = model.predict([[7]])
print(f"Predicted Exam Score for 7 hours of study: {predicted_score[0]}")



Explanation of the Code:
Import Libraries:

numpy for handling numerical operations.
matplotlib.pyplot for plotting the data.
LinearRegression from sklearn.linear_model for building the model.
train_test_split for splitting the data into training and testing sets (optional but recommended).
mean_squared_error and r2_score for evaluating the model's performance.
Dataset:

study_time is the feature (independent variable), and exam_scores is the target (dependent variable).
The data is reshaped to ensure it fits the model (required by sklearn).
Visualize Data:

The scatter plot shows the relationship between study time and exam scores.
Train-Test Split:

The data is split into training (80%) and test (20%) sets for model evaluation.
Model Creation:

The LinearRegression model is created and trained using the training data (X_train, y_train).
Get Model Parameters:

The model's slope (β1) and intercept (β0) are printed out.
Make Predictions:

Predictions are made using the test set (X_test) to evaluate how well the model performs.
Model Evaluation:

Mean Squared Error (MSE) gives an indication of the error magnitude, and R-squared shows how well the model fits the data.
Plot the Regression Line:

The regression line (best-fit line) is plotted alongside the data points to visually show the relationship.
Make Future Predictions:

The model is used to predict the exam score for a given study time (e.g., 7 hours).
Example Output:
Slope (β1): The rate of change in exam scores for each additional hour of study.
Intercept (β0): The expected exam score when the study time is zero.
R-squared: How well the regression line fits the data. A value close to 1 indicates a good fit.
Prediction for 7 hours of study: This is the exam score the model predicts for a new input (7 hours of study).
Example Plot:
The scatter plot shows the data points.
The red line shows the best-fit line calculated by the linear regression model.
Model Evaluation Metrics:
R-squared (R²): Indicates the proportion of variance in the target variable (exam scores) that can be explained by the input feature (study time).
A higher R² means the model is a good fit for the data.
Mean Squared Error (MSE): Measures the average squared difference between actual and predicted values. Lower MSE indicates better model performance.
