import numpy as np
import pandas as pd

y = np.array([0,1,2])
x = np.array([[3,3],[4,4],[5,5]])

r = np.column_stack((y,x))

print(r)

df = pd.DataFrame(r)
print(df)
df.to_csv('test1.csv')