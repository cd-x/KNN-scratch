import pandas as pd
df= pd.read_csv('/gdrive/My Drive/Colab Notebooks/breast-cancer-wisconsin.data.txt')

import numpy as np
from collections import Counter
import random

def k_nearest_neighbours(data, predict,k=3):
  distances = []
  for group in data:
    for feature in data[group]:
      euclidean_distance = np.linalg.norm(np.array(feature)-np.array(predict))
      distances.append([euclidean_distance,group])
  
  votes = [i[1] for i in sorted(distances)[:k]]
  vote_result = Counter(votes).most_common(1)[0][0]
  return vote_result


df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)
float_data = df.astype(float).values.tolist()


# x = np.array(float_data.drop(['class'],1))
# y = np.array(float_data['class'])
random.shuffle(float_data)
test_size =0.2
train_data = float_data[:-int(len(float_data)*test_size)]
test_data = float_data[-int(len(float_data)*test_size):]

#test_size = 0.2
train_set ={2:[],4:[]}
test_set = {2:[],4:[]}
for i in train_data:
  train_set[i[-1]].append(i[:-1])
for i in test_data:
  test_set[i[-1]].append(i[:-1])


correct =0
total = 0
for group in test_set:
  for data in test_set[group]:
    vote =k_nearest_neighbours(train_set,data,k=5)
    if vote==group:
      correct+=1
    total+=1

print('accuracy  :',correct/total)
