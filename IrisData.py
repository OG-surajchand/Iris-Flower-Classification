import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
iris.feature_names

Data_Iris = iris.data
Data_Iris = pd.DataFrame(Data_Iris,columns = iris.feature_names)

Data_Iris['Target'] = iris.target

plt.scatter(Data_Iris.iloc[:,2],Data_Iris.iloc[:,3],c=Data_Iris['Target'])
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.show()

#Feature
x = Data_Iris.iloc[:,0:4]

#Target
y = Data_Iris.iloc[:,4]

from sklearn.neighbors import KNeighborsClassifier
 
KNN = KNeighborsClassifier(n_neighbors=6,metric='minkowski',p=1)
KNN.fit(x,y)

x_new = np.array([[5.6,3.4,1.4,0.1]])
val = KNN.predict(x_new)


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,train_size=0.8,
                                                 random_state=88,shuffle=True,stratify=y)

from sklearn.neighbors import KNeighborsClassifier
 
KNN = KNeighborsClassifier(n_neighbors=6,metric='minkowski',p=1)
KNN.fit(x_train,y_train)

predicted = KNN.predict(x_test)

from sklearn import metrics

metrics.accuracy_score(y_test,predicted)













