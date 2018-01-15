import sys
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('reddit relative scores').getOrCreate()

assert sys.version_info >= (3, 4) # make sure we have Python 3.4+
assert spark.version >= '2.1' # make sure we have Spark 2.1+

schema = types.StructType([ # commented-out fields won't be read
    #types.StructField('archived', types.BooleanType(), False),
    types.StructField('author', types.StringType(), False),
    #types.StructField('author_flair_css_class', types.StringType(), False),
    #types.StructField('author_flair_text', types.StringType(), False),
    #types.StructField('body', types.StringType(), False),
    #types.StructField('controversiality', types.LongType(), False),
    #types.StructField('created_utc', types.StringType(), False),
    #types.StructField('distinguished', types.StringType(), False),
    #types.StructField('downs', types.LongType(), False),
    #types.StructField('edited', types.StringType(), False),
    #types.StructField('gilded', types.LongType(), False),
    #types.StructField('id', types.StringType(), False),
    #types.StructField('link_id', types.StringType(), False),
    #types.StructField('name', types.StringType(), False),
    #types.StructField('parent_id', types.StringType(), True),
    #types.StructField('retrieved_on', types.LongType(), False),
    types.StructField('score', types.LongType(), False),
    #types.StructField('score_hidden', types.BooleanType(), False),
    types.StructField('subreddit', types.StringType(), False),
    #types.StructField('subreddit_id', types.StringType(), False),
    #types.StructField('ups', types.LongType(), False),
])


def main():
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]

    comments = spark.read.json(in_directory, schema=schema)
    comments = comments.cache()

    
    grouped = comments.groupBy(comments['subreddit'])
    averages = grouped.agg(functions.avg(comments['score']).alias("avg_score"))
    bases = averages.filter(averages['avg_score'] > 0)
    bases = functions.broadcast(bases)
    joined = comments.join(bases, on='subreddit')
    data_rel = joined.select(
        joined['subreddit'],
        joined['author'],
        (joined['score']/joined['avg_score']).alias('rel_score')
        )
    data_rel = data_rel.cache()

    grouped_data = data_rel.groupBy(data_rel['subreddit'])
    data1 = grouped_data.agg(functions.max(data_rel['rel_score']).alias('max_rel_score'))
    data1 = functions.broadcast(data1)
    data2 = data_rel.join(data1, on='subreddit')
    data3 = data2.filter(data2['rel_score'] == data2['max_rel_score'])
    best_author = data3.select(
        data3['subreddit'],
        data3['author'],
        data3['rel_score']
        )
    best_author.write.json(out_directory, mode='overwrite')


if __name__=='__main__':
    main()
