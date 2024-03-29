import pandas as pd
df= pd.read_csv('/gdrive/My Drive/Colab Notebooks/breast-cancer-wisconsin.data.txt')

import numpy as np
from sklearn import preprocessing,neighbors
from sklearn.model_selection import train_test_split

df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace = True)

x = np.array(df.drop(['class'],1))
y=np.array(df['class'])

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)

accuracy  =clf.score(x_test,y_test)
print(accuracy)

#example one data

example  =np.array([4,2,1,1,1,2,3,2,1])
example = example.reshape(1,-1)

result = clf.predict(example)
print(result)
