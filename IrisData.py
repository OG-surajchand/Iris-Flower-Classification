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
