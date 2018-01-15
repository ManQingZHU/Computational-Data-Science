import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import sys

OUTPUT_TEMPLATE = (
    '\n"Are the mean time of all these sorting implementations same?" p-value: {all_same_mean_p:.3g}\n'
    'As the p-value is significantly small, we can conclude that there are some sorting implementations with different speed.'
)

def HSDtest(data):
	data_melt = pd.melt(data)
	return pairwise_tukeyhsd(data_melt['value'], data_melt['variable'], alpha=0.05)

def main():
	filename = sys.argv[1]
	data = pd.read_csv(filename)

	anova_p = stats.f_oneway(data['qs1'], data['qs2'], data['qs3'], data['qs4'], data['qs5'], data['merge1'], data['partition_sort']).pvalue
	posthoc = HSDtest(data)

	# Output
	print(50*"=")
	print(OUTPUT_TEMPLATE.format(all_same_mean_p=anova_p,))
	print(posthoc)
	print("For the pairs in the table with 'False' in the reject column, we can't conclude that they have different running speeds.\n")
	fig = posthoc.plot_simultaneous()
	plt.xlabel("running time")
	fig.set_size_inches(12, 5)
	plt.show()


if __name__ == '__main__':
	main()