import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

filename1 = sys.argv[1]
filename2 = sys.argv[2]

data1 = pd.read_table(filename1, sep=' ', header=None, index_col=1, names=['lang', 'page', 'views', 'bytes'])
data2 = pd.read_table(filename2, sep=' ', header=None, index_col=1, names=['lang', 'page', 'views', 'bytes'])
data2['pre_views'] = data1['views']
data1 = data1.sort_values('views', ascending=False)
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.plot(data1['views'].values)
plt.title('Popularity Distribution')
plt.xlabel('Rank')
plt.ylabel('Views')

plt.subplot(1, 2, 2)
plt.plot(data2['pre_views'].values, data2['views'].values,'b.')
plt.xscale('log')
plt.yscale('log')
plt.title('Daily Correlation')
plt.xlabel('Day 1 views')
plt.ylabel('Day 2 views')
#plt.show()
plt.savefig('wikipedia.png')