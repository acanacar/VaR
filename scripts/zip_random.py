import numpy as np
import functools
import random
import pandas as pd

'''
n = 4
df = pd.DataFrame()
lis = np.random.rand(n)
lis_sum = functools.reduce(lambda a, b: a + b, lis)
weights = list(map(lambda y: y / lis_sum, lis))
cols = random.sample(list(df.columns), k=n)
result = zip(weights,cols)
'''