import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


'''
if the column names werent there 

pd.read_csv("dow_jones_index.data")


cols = ["",""] #columns
#df = pd.read_csv("dow_jones_index.data", names=cols)

'''

df = pd.read_csv("dow_jones_index.data")

stocks = df["stock"].unique()

#classification
# creating a dictionary to replace stock with integers
stocks_dict = {}
for i in range(len(stocks)):
  stocks_dict[stocks[i]] = i

print(stocks_dict)

df["stock"] = df["stock"].replace(stocks_dict)
df

for label in df.columns[:-1]: #everything but the last column
  #getting everything from stock 1 with a specific label 
  plt.hist(df[df["stock"]==1][label], color='blue', label='AXP', alpha=0.7, density=True)
  #getting everything from stock 0 with a specific label
  plt.hist(df[df["stock"]==0][label], color='red', label='AA', alpha=0.7, density=True)

  plt.title(label)
  plt.ylabel("Probability")
  plt.xlabel(label)
  plt.legend()
  plt.show()

train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])
'''
  np.split(sample will shuffle data, 1st split 60%, 60-80% towards validation,80-100% test data)
'''

def scale_dataset(dataframe):
  X = dataframe[dataframe.columns[:-1]].values
  y = dataframe[dataframe.columns[-1]].values

  scaler = StandardScaler()

  #take x and fit the stand scalar to x, then transform values to get new x 
  X = scaler.fit.transform(X)

  #Take 2 arrays and stack them horizontally (side by side, not ontop of each other)
  #have to call numpy.reshape because X is 2d and y is 1d
  data = np.hstack((X, np.reshape(y, (-1,1))))

print("this is training data")
train

