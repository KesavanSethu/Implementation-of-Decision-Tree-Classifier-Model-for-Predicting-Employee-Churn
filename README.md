# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Elavarasan M
RegisterNumber:  212224040083
*/
```
```
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
data=pd.read_csv("Employee.csv")
data.head()
```
```
data.tail()
```
```
data.isnull().sum()
```
```
data.info()
```
```
data["left"].value_counts()
```
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
```
```
x=data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", "time_spend_company", "Work_accident", "promotion_last_5years","salary"]]
x
```
```
y=data["left"]
y
```
```
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=100)
```
```
dt = DecisionTreeClassifier(criterion='entropy', random_state=100)
dt.fit(x_train, y_train)
```

```
y_pred = dt.predict(x_test)
```
```
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy=metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100}%")
```

```
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree  
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```
## Output:

**Head Values**

![Screenshot 2025-05-12 174428](https://github.com/user-attachments/assets/85b44b4a-6df7-4add-97e4-fa5c918b1408)

**Tail Values**

![Screenshot 2025-05-12 174436](https://github.com/user-attachments/assets/d9587ea6-b468-49fc-b809-5e4bbcd65213)

**Sum - Null Values**

![Screenshot 2025-05-12 174447](https://github.com/user-attachments/assets/ffaf6e25-132a-48af-b16b-5770b72b6413)

**Data Info**

![Screenshot 2025-05-12 174453](https://github.com/user-attachments/assets/0192e982-ec67-4875-b89e-4384bc697cbb)

**Values count in left column**

![Screenshot 2025-05-12 174500](https://github.com/user-attachments/assets/a9dc0f28-b5bf-498c-b8f1-9e1a50ed3426)

**X values**

![Screenshot 2025-05-12 174513](https://github.com/user-attachments/assets/e7f75a59-7cc9-4c98-8576-d034a0869c97)

**Y Values**

![Screenshot 2025-05-12 174519](https://github.com/user-attachments/assets/093bcf45-a5f2-4bc0-a43c-0f34c714e062)

**Training the model**

![Screenshot 2025-05-12 174524](https://github.com/user-attachments/assets/4e9c5f7a-9f69-46f0-b21a-c50402738455)

**Accuracy**

![Screenshot 2025-05-12 174531](https://github.com/user-attachments/assets/446f17d0-fe1e-448a-8f9c-16f9d2bbab7a)

**Data Prediction**

![Screenshot 2025-05-12 174544](https://github.com/user-attachments/assets/7ff2f61e-b3cc-480a-bc1e-d6f5a425dd70)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
