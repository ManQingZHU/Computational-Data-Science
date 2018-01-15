import sys
import pandas as pd
from difflib import get_close_matches

filename1 = sys.argv[1]
filename2 = sys.argv[2]
outputfile = sys.argv[3]

movies = pd.read_table(filename1,header=None, names=["title"])
# ratings = pd.read_table(filename2,)
ratings = pd.Series(open(filename2).readlines())

def CleanRate(rating):
    rating = rating.replace('\n', '')
    return rating.rsplit(',',1)

ratings = ratings.apply(CleanRate)
ratings = ratings[1:]
rate_data = pd.DataFrame.from_items(zip(ratings.index,ratings.values)).T
rate_data = rate_data.rename(columns={0:'title', 1:'rating'})

join_str = ''
def MatchName(name):
    res = get_close_matches(name, movies['title'], n=1)
    return join_str.join(res)

rate_data['title'] = rate_data['title'].apply(MatchName)
clean_data = rate_data[rate_data['title'].str.len() != 0]
clean_data['rating'] = clean_data['rating'].apply(lambda x: int(x))
grouped_data = clean_data.groupby('title')
final_data = grouped_data.aggregate('mean').reset_index()
final_data['rating'] = final_data['rating'].round(2)
final_data = final_data.sort_values('title', axis=0)
final_data.to_csv(outputfile,index = False)

