import sys
from pyspark.sql import SparkSession, functions, types, Row
import re
import math

spark = SparkSession.builder.appName('correlate logs').getOrCreate()

assert sys.version_info >= (3, 4) # make sure we have Python 3.4+
assert spark.version >= '2.1' # make sure we have Spark 2.1+

line_re = re.compile("^(\\S+) - - \\[\\S+ [+-]\\d+\\] \"[A-Z]+ \\S+ HTTP/\\d\\.\\d\" \\d+ (\\d+)$")


def line_to_row(line):
    """
    Take a logfile line and return a Row object with hostname and bytes transferred. Return None if regex doesn't match.
    """
    m = line_re.match(line)
    if m:
        return Row(hostname=m.group(1), transferred=m.group(2))
    else:
        return None

def not_none(row):
    """
    Is this None? Hint: .filter() with it.
    """
    return row is not None


def create_row_rdd(in_directory):
    log_lines = spark.sparkContext.textFile(in_directory)
    row_rdd = log_lines.map(line_to_row).filter(not_none)
    return row_rdd


def main():
    in_directory = sys.argv[1]
    logs = spark.createDataFrame(create_row_rdd(in_directory))
    #logs.show(); return
    grouped = logs.groupBy(logs['hostname'])
    counted = grouped.agg(
        functions.count(logs['transferred']).alias('x'),
        functions.sum(logs['transferred']).alias('y')
        )

    # add constant column to a Spark DataFrame
    # reference: 
    # https://stackoverflow.com/questions/32788322/how-to-add-a-constant-column-in-a-spark-dataframe
    data = counted.withColumn('1',functions.lit(1))

    cleaned_data = data.select(
        data['1'],
        data['x'],
        (data['x']*data['x']).alias('x2'),
        data['y'],
        (data['y']*data['y']).alias('y2'),
        (data['x']*data['y']).alias('xy')
        )
    #cleaned_data.show(); return

    sumed = cleaned_data.groupBy().agg(
        functions.sum(cleaned_data['1']),
        functions.sum(cleaned_data['x']),
        functions.sum(cleaned_data['x2']),
        functions.sum(cleaned_data['y']),
        functions.sum(cleaned_data['y2']),
        functions.sum(cleaned_data['xy'])
        )
    
    sum_1, sum_x, sum_x2, sum_y, sum_y2, sum_xy = sumed.first()

    r = (sum_1*sum_xy-sum_x*sum_y)/(math.sqrt(sum_1*sum_x2-sum_x**2)*math.sqrt(sum_1*sum_y2-sum_y**2))
    print("r = %g\nr^2 = %g" % (r, r**2))


if __name__=='__main__':
    main()