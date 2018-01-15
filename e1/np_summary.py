import numpy as np
data = np.load('monthdata.npz')
totals = data['totals']
counts = data['counts']
# print(totals)
# print(counts)

# Row with lowest total precipitation:
total_year = np.sum(totals, axis=1)
min_row = np.argmin(total_year)
print("Row with lowest total precipitation:")
print(min_row)

# Average precipitation in each month:
total_mon = np.sum(totals, axis=0)
count_mon = np.sum(counts, axis=0)
ave_mon = np.divide(total_mon, count_mon)
print("Average precipitation in each month:")
print(ave_mon)

# Average precipitation in each city:
count_city = np.sum(counts,axis=1)
ave_city = np.divide(total_year, count_city)
print("Average precipitation in each city:")
print(ave_city)

# Quarterly precipitation totals:
n = totals.shape[0]
temp = np.reshape(totals, (4*n, 3))
total_quar = np.sum(temp, axis = 1)
total_quar = np.reshape(total_quar, (n, 4))
print("Quarterly precipitation totals:")
print(total_quar)