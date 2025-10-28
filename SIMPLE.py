#ML PROGRAM
#Import dependencies
import numpy as np
from sklearn.model_selection import train_test_split #to split dataset to two part-train and test
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Inpu/import dataset
# import pandas as pd
#x=Number of hours studied
x=np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
#y=Pass=1 or fail=0
y=np.array([[0],[0],[0],[1],[1],[1],[1],[1],[1],[1]])

#Split data into train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
# print("x_train",x_train)
# print("x_test",x_test)
# print("y_train",y_train)
# print("y_test",y_test)

#train
model=LogisticRegression()
#fit function is available in sklearn.
model.fit(x_train,y_train)

#Predict
#moment u give input to the model it wil assign the label.it will be stored in variable-y_pred
y_pred=model.predict(x_test)

#prediction
print(f"predicted labels:{y_test}")
print(f"predicted labels:{y_pred}")

#accuracy
print(f"accuracy_score:{accuracy_score(y_test,y_pred)}")

#make prediction onseen data
hours=np.array([[4.5],[3],[50],[1]])
result=model.predict(hours)
for h,r in zip(hours,result):
    print(f"if u study {h}  hrs:{"pass" if r else "fail"}")
    
    

