# Step 1: Import Required Libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn import tree

# Step 2: Load the Iris Dataset
# The Iris dataset is a built-in dataset in scikit-learn
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target labels (species)

# Step 3: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and Train the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Step 5: Make Predictions on the Test Set
y_pred = dt_classifier.predict(X_test)

# Step 6: Evaluate the Model's Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification report for more detailed performance metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 7: Visualize the Decision Tree (optional)
plt.figure(figsize=(12, 8))
tree.plot_tree(dt_classifier, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree Visualization")
plt.show()










Explanation:
Load the Dataset:

We are using the Iris dataset from scikit-learn, which is a multi-class classification dataset with 150 instances of iris flowers.
It contains 4 features: sepal length, sepal width, petal length, and petal width.
Train-Test Split:

The dataset is split into 80% training and 20% testing using train_test_split.
Create the Decision Tree Classifier:

A DecisionTreeClassifier model is created and trained using the training data (X_train, y_train).
Make Predictions:

After the model is trained, predictions are made on the test data (X_test), and the accuracy is calculated using accuracy_score.
Evaluate the Model:

We print the accuracy of the model and use classification report to show detailed metrics like precision, recall, and F1-score.
Visualize the Decision Tree (Optional):

The Decision Tree can be visualized using the plot_tree function, which provides a graphical representation of the decision rules
