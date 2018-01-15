import sys
import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

OUTPUT_TEMPLATE = (
    "Initial (invalid) T-test p-value: {initial_ttest_p:.3g}\n"
    "Original data normality p-values: {initial_weekday_normality_p:.3g} {initial_weekend_normality_p:.3g}\n"
    "Original data equal-variance p-value: {initial_levene_p:.3g}\n"
    "Transformed data normality p-values: {transformed_weekday_normality_p:.3g} {transformed_weekend_normality_p:.3g}\n"
    "Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n"
    "Weekly data normality p-values: {weekly_weekday_normality_p:.3g} {weekly_weekend_normality_p:.3g}\n"
    "Weekly data equal-variance p-value: {weekly_levene_p:.3g}\n"
    "Weekly T-test p-value: {weekly_ttest_p:.3g}\n"
    "Mann–Whitney U-test p-value: {utest_p:.3g}"
)

def date_year(date):
    return date.year

def date_weekday(date):
    return date.weekday()

def iso_year(date):
    return date.isocalendar()[0]

def week_num(date):
    date = pd.to_datetime(date)
    return date.isocalendar()[1]

def main():
	file = sys.argv[1]
	reddit_fh = gzip.open(file, 'rt', encoding='utf-8')
	reddits = pd.read_json(reddit_fh, lines=True)
	# r/canada
	reddits = reddits[(reddits["subreddit"]== 'canada')]
	
	# 2012 - 2013
	reddits["year"] = reddits["date"].apply(date_year)
	reddits = reddits[(reddits["year"] == 2012)|(reddits["year"] == 2013)]
	
	# seperate weekdays and weekends
	reddits["weekday"] = reddits["date"].apply(date_weekday)
	weekdays = pd.DataFrame(reddits[(reddits["weekday"] < 5)])
	weekends = pd.DataFrame(reddits[(reddits["weekday"] >= 5)])

	# initial t-test
	initial_ttest_p = stats.ttest_ind(weekdays["comment_count"],weekends["comment_count"]).pvalue
	initial_weekday_normality_p = stats.normaltest(weekdays["comment_count"]).pvalue
	initial_weekend_normality_p = stats.normaltest(weekends["comment_count"]).pvalue
	initial_levene_p = stats.levene(weekdays["comment_count"], weekends["comment_count"]).pvalue

	# have a look at the histogram
	# plt.hist([weekdays["comment_count"], weekends["comment_count"]])
	# plt.legend(["weekday", "weekend"])
	# plt.show()

	# transformation
	trans_weekday = np.sqrt(weekdays["comment_count"])
	trans_weekend = np.sqrt(weekends["comment_count"])
	transformed_weekday_normality_p = stats.normaltest(trans_weekday).pvalue
	transformed_weekend_normality_p = stats.normaltest(trans_weekend).pvalue
	transformed_levene_p = stats.levene(trans_weekday, trans_weekend).pvalue

	# group and aggregate
	weekdays["isoyear"] = weekdays["date"].apply(iso_year)
	weekdays["weekno"] = weekdays["date"].apply(week_num)
	weekends["isoyear"] = weekends["date"].apply(iso_year)
	weekends["weekno"] = weekends["date"].apply(week_num)
	grouped_weekdays = weekdays.groupby(["isoyear", "weekno"])
	mean_weekdays = grouped_weekdays.aggregate('mean').reset_index()
	grouped_weekends = weekends.groupby(["isoyear", "weekno"])
	mean_weekends = grouped_weekends.aggregate('mean').reset_index()

	# t-test
	weekly_weekday_normality_p=stats.normaltest(mean_weekdays["comment_count"]).pvalue
	weekly_weekend_normality_p=stats.normaltest(mean_weekends["comment_count"]).pvalue
	weekly_levene_p=stats.levene(mean_weekdays["comment_count"], mean_weekends["comment_count"]).pvalue
	weekly_ttest_p=stats.ttest_ind(mean_weekdays["comment_count"], mean_weekends["comment_count"]).pvalue
	
	# u-test
	utest_p = stats.mannwhitneyu(weekdays["comment_count"], weekends["comment_count"]).pvalue

	# output
	print(OUTPUT_TEMPLATE.format(
        initial_ttest_p = initial_ttest_p,
        initial_weekday_normality_p = initial_weekday_normality_p,
        initial_weekend_normality_p = initial_weekend_normality_p,
        initial_levene_p = initial_levene_p,
        transformed_weekday_normality_p = transformed_weekday_normality_p,
        transformed_weekend_normality_p = transformed_weekend_normality_p,
        transformed_levene_p = transformed_levene_p,
        weekly_weekday_normality_p = weekly_weekday_normality_p,
        weekly_weekend_normality_p = weekly_weekend_normality_p,
        weekly_levene_p = weekly_levene_p,
        weekly_ttest_p = weekly_ttest_p,
        utest_p = utest_p
    ))

if __name__ == '__main__':
	main()