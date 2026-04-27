# Random Forest Classification with Encoding and Improvements

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv(r"C:\Users\niles\Downloads\Churn_Modelling.csv")

# Features: exclude RowNumber, CustomerId, Surname, and the target column
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encode categorical data (Geography, Gender)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(drop='first'), [1, 2])],  # Geography=col1, Gender=col2
    remainder='passthrough'
)
X = ct.fit_transform(X)

# Split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)

# Train Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(
    max_depth=6, n_estimators=100, random_state=0, criterion='entropy'
)
classifier.fit(X_train, y_train)

# Predictions
y_pred = classifier.predict(X_test)

# Evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Bias vs Variance
print("Bias (Train Accuracy):", classifier.score(X_train, y_train))
print("Variance (Test Accuracy):", classifier.score(X_test, y_test))

# Feature Importance
importances = classifier.feature_importances_
feature_names = ct.get_feature_names_out()

plt.figure(figsize=(12,6))
plt.bar(range(len(importances)), importances, tick_label=feature_names)
plt.xticks(rotation=90)
plt.title("Feature Importance in Random Forest")
plt.show()

# Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [4, 6, 8, None],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=0),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)


# applying k-fold cross validation

from sklearn.model_selection import cross_val_score
acc=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=5)
print('accuracy: {:.2f} %'.format(acc.mean()*100))