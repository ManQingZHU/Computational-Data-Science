import sys
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('weather ETL').getOrCreate()

assert sys.version_info >= (3, 4) # make sure we have Python 3.4+
assert spark.version >= '2.2' # make sure we have Spark 2.2+

observation_schema = types.StructType([
    types.StructField('station', types.StringType(), False),
    types.StructField('date', types.StringType(), False),
    types.StructField('observation', types.StringType(), False),
    types.StructField('value', types.IntegerType(), False),
    types.StructField('mflag', types.StringType(), False),
    types.StructField('qflag', types.StringType(), False),
    types.StructField('sflag', types.StringType(), False),
    types.StructField('obstime', types.StringType(), False),
])


def main(in_directory, out_directory):
    weather = spark.read.csv(in_directory, schema=observation_schema)
    #weather.printSchema(); return
    first_clean = weather.filter(weather.qflag.isNull())
    #first_clean.show(); return
    first_clean = first_clean.filter( first_clean['station'].startswith('CA'))
    #first_clean.show(); return
    first_clean = first_clean.filter( first_clean['observation'] == 'TMAX')
    #first_clean.show(); return
    cleaned_data = first_clean.select(
    	first_clean['station'],
    	first_clean['date'],
    	(first_clean['value'] / 10).alias('tmax'),
    	)
    #cleaned_data.show(); return
    cleaned_data.write.json(out_directory, compression='gzip', mode='overwrite')


if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)