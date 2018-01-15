import string, re
import sys
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('word count').getOrCreate()

assert sys.version_info >= (3, 4) # make sure we have Python 3.4+
assert spark.version >= '2.1' # make sure we have Spark 2.1+


wordbreak = r'[%s\s]+' % (re.escape(string.punctuation),)

def main():
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]

    lines = spark.read.text(in_directory)
    sep_lines = lines.select(
    	(functions.explode(functions.split(lines['value'], wordbreak))).alias('word')
    	)
    words = sep_lines.filter(sep_lines['word'] != "")
    lowered = words.select(
    	(functions.lower(words['word'])).alias('word')
    	)
    grouped = lowered.groupBy(lowered['word'])
    counted = grouped.agg(
    	(functions.count(lowered['word'])).alias('count')
    	)
    sorted = counted.sort(counted['count'], ascending=False)
    sorted.write.csv(out_directory, mode='overwrite')

if __name__ == '__main__':
	main()
    