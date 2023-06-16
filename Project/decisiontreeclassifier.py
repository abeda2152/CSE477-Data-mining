# Decision Tree Classification

# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Importing the dataset
dataset = pd.read_csv('chess_king_rook_weka_dataset.csv')

#preprocessing categorical 
dwkf = pd.get_dummies(dataset['white_king_file'], prefix = 'wkf')
dwkr = pd.get_dummies(dataset['white_king_rank'], prefix = 'wkr')
dwrf = pd.get_dummies(dataset['white_rook_file'], prefix = 'wrf')
dwrr = pd.get_dummies(dataset['white_rook_rank'], prefix = 'wrr')
dbkf = pd.get_dummies(dataset['black_king_file'], prefix = 'bkf')
dbkr = pd.get_dummies(dataset['black_king_rank'], prefix = 'bkr')

#concat new columns to original dataframe
dataset_concat = pd.concat([dwkf, dwkr, dwrf, dwrr, dbkf, dbkr, dataset['result']], axis = 1)

#encoding class label 
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
dataset_concat['result'] = labelencoder.fit_transform(dataset_concat['result'])
'''labelencoder.fit(dataset_concat['result'])
dataset_concat['result'] = dataset_concat['result'].map( {'draw':17, 'zero':0, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6,
       'seven':7, 'eight':8, 'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'thirteen':13,
       'fourteen':14, 'fifteen':15, 'sixteen':16})
dataset_concat['result'].unique()'''

#splitting class label
X = dataset_concat.iloc[:, :-1].values
y = dataset_concat.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#FP = cm.sum(axis=0) - np.diag(cm)  
#FN = cm.sum(axis=1) - np.diag(cm)
#TP = np.diag(cm)
#TN = cm.values.sum() - (FP + FN + TP)

# recall, or true positive rate
#TPR = TP/(TP+FN)
# Fall out or false positive rate
#FPR = FP/(FP+TN)
# Overall accuracy
#AUC = (TP+TN)/(TP+FP+FN+TN)

#precision recall flscore
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)

#accuracy
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)*100

#plotting test and predicted class label
sns.distplot(y_test, bins = 20, hist = False, label = 'actual')
sns.distplot(y_pred, bins = 20, hist = False, label = 'predicted')
plt.xlabel('result')
plt.title('Decision Tree Predicted vs. Actual class label')
#plt.savefig('Predictedvactual.png')
plt.show()
