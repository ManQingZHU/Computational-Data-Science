import pandas as pd
totals = pd.read_csv('totals.csv').set_index(keys=['name'])
counts = pd.read_csv('counts.csv').set_index(keys=['name'])
# print(totals)
# print(counts)

# City with lowest total precipitation:
total_city = pd.DataFrame.sum(totals, axis=1)
min_city = pd.Series.argmin(total_city)
print("City with lowest total precipitation:")
print(min_city)

# Average precipitation in each month:
total_mon = pd.DataFrame.sum(totals, axis=0)
count_mon = pd.DataFrame.sum(counts, axis=0)
ave_mon = total_mon.divide(count_mon)
print("Average precipitation in each month:")
print(ave_mon)

# Average precipitation in each city:
count_city = pd.DataFrame.sum(counts, axis=1)
ave_city = total_city.divide(count_city)
print("Average precipitation in each city:")
print(ave_city)