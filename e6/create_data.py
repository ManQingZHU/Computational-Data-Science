import time
from implementations import all_implementations
import numpy as np 
import pandas as pd

def create_data(test_num):
	random_array = np.random.randint(-10000, 10000, size = 20000)
	time_data = []
	for i in range(test_num):
		times = []
		for sort in all_implementations:
			st = time.time()
			res = sort(random_array)
			en = time.time()
			times.append(en-st)
		time_data.append(times)
	return pd.DataFrame(time_data, columns=['qs1', 'qs2', 'qs3', 'qs4', 'qs5', 'merge1', 'partition_sort'])
		
def main():
	test_num = 50
	print("Creating data...")
	data = create_data(test_num)
	data.to_csv('data.csv', index = False)
	print("Data prepared.")
	
if __name__ == '__main__':
	main()