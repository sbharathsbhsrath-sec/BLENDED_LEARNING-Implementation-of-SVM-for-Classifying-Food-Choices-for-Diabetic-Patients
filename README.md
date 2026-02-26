# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. - Data Preparation and Splitting
The dataset is loaded from food_items_binary.csv. Nutritional features such as calories, fats, sugars, fiber, and protein are chosen as predictors, while the class column is the target. The data is split into training and testing sets (80/20 split). Standardization is applied using StandardScaler so that all features are on the same scale, which is important for algorithms like SVM.

2. - Model Selection with Grid Search
A Support Vector Machine (SVC) classifier is used. To find the best hyperparameters, GridSearchCV is applied with a parameter grid that includes different values for C, kernel types (linear, rbf), and gamma settings (scale, auto). Cross‑validation ensures the model is tested on multiple folds of the training data to identify the most accurate configuration

3. - Training and Evaluation
The best model found by grid search is trained on the scaled training data. Predictions are made on the test set. The model’s performance is measured using accuracy, a classification report (precision, recall, F1‑score), and a confusion matrix to show how well the model distinguishes between the two classes.

4.- Visualization
The confusion matrix is plotted as a heatmap using Seaborn. This provides a visual summary of correct versus incorrect predictions, making it easier to interpret the model’s strengths and weaknesses in classifying the food items.
 

## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


data=pd.read_csv('food_items_binary.csv')


print(data.head())
print(data.columns)

features=['Calories','Total Fat','Saturated Fat','Sugars','Dietary Fiber','Protein']
target='class'

x=data[features]
y=data[target]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

svm=SVC()

param_grid={
    'C':[0.1,1,10,100],
    'kernel':['linear','rbf'],
    'gamma': ['scale','auto']
}

grid_search=GridSearchCV(svm,param_grid,cv=5,scoring='accuracy')
grid_search.fit(x_train,y_train)

best_model=grid_search.best_estimator_
print('\nName: Bharath S')
print("Reg no: 212225230031")
print("Best Parameters",grid_search.best_params_,'\n')

y_pred=best_model.predict(x_test)

accuracy=accuracy_score(y_test,y_pred)
print("Accuracy",accuracy)

print('classification report:\n',classification_report(y_test,y_pred))
conf_matrix=confusion_matrix(y_test,y_pred)
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='Blues')
plt.xlabel('preedicted')
plt.ylabel('Actual')
plt.title('confusion Matrix')
plt.show() 

```

## Output:
<img width="719" height="419" alt="image" src="https://github.com/user-attachments/assets/2a79123e-e193-4a5e-9ba3-9351858efd17" />

<img width="686" height="585" alt="image" src="https://github.com/user-attachments/assets/f8c4c4cb-2e7b-4748-acaf-6744f738325b" />


## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
