import numpy as np
import pandas as  pd

data = pd.read_csv("BankNote_Authentication.csv")

#print(data.head())

X = data.iloc[:,:-1]
Y = data.iloc[:,-1]

print(X.head())
print("-----------------------------------------------")
print(Y.head())

X.describe()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.3)

y_train.head()

from sklearn.ensemble import RandomForestClassifier
rand_fo = RandomForestClassifier()
rand_fo.fit(X_train,y_train) 

y_pred = rand_fo.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)

from sklearn.metrics import precision_score,recall_score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("precision: ",precision,"Recall: ",recall)


import pickle
pickle_out = open("note_classifier.pkl","wb")
pickle.dump(rand_fo, pickle_out)
pickle_out.close()
