import pandas as pd
import numpy as np

s = np.load('record.npy',allow_pickle=True).item()
print(s)