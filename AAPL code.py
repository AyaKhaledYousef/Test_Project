import numpy as np 
import pandas as pd 
import csv
import matplotlib.pyplot as plt
import time
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.svm import SVC ,NuSVC,LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


data=pd.read_csv("AAPL.csv")
data['Date'] = pd.to_datetime(data['Date'])  #add DateTime to my data to generate dt 

data['day_of_week'] = data['Date'].dt.weekday_name  #add the days standard to the date

#data['Open'] = data['Open'].astype(int) # Convert to int 
#data['Close'] = data['Close'].astype(int) 

data['same_day_delta'] = ((data['Close'] - data['Open']) / data['Open'] ) * 100 #percentage difference between 'open','close'

data['same_day_strategy'] = np.where(data['same_day_delta'] <= 0,'0' ,'1')
data['same_day_strategy']=pd.to_numeric(data['same_day_strategy'], errors='ignore')#convert from object to int
print(data['same_day_strategy'].dtypes)


data['next_close_delta'] = 100 * (1 - data.iloc[0].Close / data.Close)
data['next_close_strategy'] = np.where(data['next_close_delta'] <= 1,'0' ,'1')
data['next_close_strategy']=pd.to_numeric(data['next_close_strategy'], errors='ignore')#convert from object to int
print(data['next_close_strategy'].dtypes)


#information per month
data['year'] = pd.DatetimeIndex(data['Date']).year #extract year from Date
data['month'] = pd.DatetimeIndex(data['Date']).month # extract month from Date
monthly=data.groupby(['month','year']).agg({"Close":['mean','max'],"Open":['mean','min'],"High":['min','max'],"Low":['min','max']})
monthly.to_csv("monthly_analysis.csv")               

#plot
#data['same_day_delta'].value_counts().plot() 
plt.title('Plot the same_day_delta.')  
plt.plot(data["same_day_delta"])#plot the column
plt.show()

#plot “open”, “close” price in the same plot
plt.title('open and close')
plt.plot(data["Open"])
plt.plot(data["Close"])
plt.show()


#data['same_day_delta'].value_counts().plot.hist() #Plot the distribution of “same_day_delta”.
plt.title('Distribution of same_day_delta')  
plt.hist(data["same_day_delta"]) #Plot the distribution of “same_day_delta”.
plt.show()
######################
#the execution time

start_time = time.time()
print("--- %s data ---" % (time.time() - start_time))

#importing the data about same_day_strategy
X=data.iloc[:,[1,4]].values
y=data.iloc[:,9].values
           
#spliting the dataset        
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)
#fitting logistic reg to the trainig set
from sklearn.linear_model import LogisticRegression
classifer = LogisticRegression(random_state=0)
classifer.fit(X_train,y_train)
#predicting the test set result
y_pred=classifer.predict(X_test)
#Making the confusion Matix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

#excution time for this model model

start_time = time.time()
print("--- %s y_pred ---" % (time.time() - start_time))


#importing the data about next_close_strategy
a=data.iloc[:,[4,4]].values
b=data.iloc[:,11].values
           
#spliting the dataset        
from sklearn.model_selection import train_test_split
a_train,a_test,b_train,b_test=train_test_split(a,b,test_size=1/3,random_state=0)
#fitting logistic reg to the trainig set
from sklearn.linear_model import LogisticRegression
classifer = LogisticRegression(random_state=0)
classifer.fit(a_train,b_train)
#predicting the test set result
b_pred=classifer.predict(a_test)
#Making the confusion Matix
from sklearn.metrics import confusion_matrix
cm2=confusion_matrix(b_test,b_pred)
print(cm2)

#excution time for this model model

start_time = time.time()
print("--- %s b_pred ---" % (time.time() - start_time))