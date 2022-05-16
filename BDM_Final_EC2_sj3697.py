import csv
import sys
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import *
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark import SparkContext
import pandas as pd
import ast

from pyproj import Transformer
import shapely
from shapely.geometry import Point


def main(sc,sqlcontext):

    df = pd.read_csv('nyc_cbg_centroids.csv')
    outputCBG = df.set_index('cbg_fips').T.to_dict('list')

    def read_place_key(partId, part):
      if partId == 0: next(part)
      for x in csv.reader(part):
        s = str(x[9])
        if s[:4] == '4451':
          yield x[0]

    outputcoreplace = sc.textFile('/tmp/bdm/core-places-nyc.csv') \
                  .mapPartitionsWithIndex(read_place_key).collect()

    def readPatterns(partId, part):
      t = Transformer.from_crs(4326, 2263)
      if partId == 0: next(part)
      for x in csv.reader(part):
        if x[0] in outputcoreplace:
          if '2019-04-01' > x[12] >= '2019-03-01' or '2019-04-01' > x[13] >= '2019-03-01':
            temp = ast.literal_eval(x[19])
            id = int(x[18])
            for key in temp:
              k = int(key)
              if k in outputCBG:
                a = t.transform(outputCBG[k][0],outputCBG[k][1])
                b = t.transform(outputCBG[id][0],outputCBG[id][1])
                dis = Point(a).distance(Point((b)))/5280
                yield k, dis*temp[key], temp[key]
                
    def readPatterns_2019_10(partId, part):
      t = Transformer.from_crs(4326, 2263)
      if partId == 0: next(part)
      for x in csv.reader(part):
        if x[0] in outputcoreplace:
          if '2019-11-01' > x[12] >= '2019-10-01' or '2019-11-01' > x[13] >= '2019-10-01':
            temp = ast.literal_eval(x[19])
            id = int(x[18])
            for key in temp:
              k = int(key)
              if k in outputCBG:
                a = t.transform(outputCBG[k][0],outputCBG[k][1])
                b = t.transform(outputCBG[id][0],outputCBG[id][1])
                dis = Point(a).distance(Point((b)))/5280
                yield k, dis*temp[key], temp[key]

    def readPatterns_2020_03(partId, part):
      t = Transformer.from_crs(4326, 2263)
      if partId == 0: next(part)
      for x in csv.reader(part):
        if x[0] in outputcoreplace:
          if '2020-04-01' > x[12] >= '2020-03-01' or '2020-04-01' > x[13] >= '2020-03-01':
            temp = ast.literal_eval(x[19])
            id = int(x[18])
            for key in temp:
              k = int(key)
              if k in outputCBG:
                a = t.transform(outputCBG[k][0],outputCBG[k][1])
                b = t.transform(outputCBG[id][0],outputCBG[id][1])
                dis = Point(a).distance(Point((b)))/5280
                yield k, dis*temp[key], temp[key]

    def readPatterns_2020_10(partId, part):
      t = Transformer.from_crs(4326, 2263)
      if partId == 0: next(part)
      for x in csv.reader(part):
        if x[0] in outputcoreplace:
          if '2020-11-01' > x[12] >= '2020-10-01' or '2020-11-01' > x[13] >= '2020-10-01':
            temp = ast.literal_eval(x[19])
            id = int(x[18])
            for key in temp:
              k = int(key)
              if k in outputCBG:
                a = t.transform(outputCBG[k][0],outputCBG[k][1])
                b = t.transform(outputCBG[id][0],outputCBG[id][1])
                dis = Point(a).distance(Point((b)))/5280
                yield k, dis*temp[key], temp[key]

    output2019_03 = sc.textFile('/tmp/bdm/weekly-patterns-nyc-2019-2020') \
              .mapPartitionsWithIndex(readPatterns)

    deptColumns = ["cbg","dis","count"]
    df_2019_03 = output2019_03.toDF(deptColumns)

    df_2019_03 = df_2019_03.groupBy('cbg').sum('dis', 'count')
    df_2019_03 = df_2019_03.withColumn('2019_03', (df_2019_03[1]/df_2019_03[2])).select('cbg', '2019_03')

    output2019_10 = sc.textFile('/tmp/bdm/weekly-patterns-nyc-2019-2020') \
              .mapPartitionsWithIndex(readPatterns_2019_10)

    deptColumns = ["cbg","dis","count"]
    df_2019_10 = output2019_10.toDF(deptColumns)
    df_2019_10 = df_2019_10.groupBy('cbg').sum('dis', 'count')
    df_2019_10 = df_2019_10.withColumn('2019_10', (df_2019_10[1]/df_2019_10[2])) \
        .select('cbg', '2019_10')
    final = df_2019_03.join(df_2019_10, on = "cbg", how='full').cache()

    output2020_03 = sc.textFile('/tmp/bdm/weekly-patterns-nyc-2019-2020') \
              .mapPartitionsWithIndex(readPatterns_2020_03)

    deptColumns = ["cbg","dis","count"]
    df_2020_03 = output2020_03.toDF(deptColumns)
    df_2020_03 = df_2020_03.groupBy('cbg').sum('dis', 'count')
    df_2020_03 = df_2020_03.withColumn('2020_03', (df_2020_03[1]/df_2020_03[2])) \
        .select('cbg', '2020_03')
    final = final.join(df_2020_03, on = "cbg", how='full').cache()

    output2020_10 = sc.textFile('/tmp/bdm/weekly-patterns-nyc-2019-2020') \
              .mapPartitionsWithIndex(readPatterns_2020_10)

    deptColumns = ["cbg","dis","count"]
    df_2020_10 = output2020_10.toDF(deptColumns)
    df_2020_10 = df_2020_10.groupBy('cbg').sum('dis', 'count')
    df_2020_10 = df_2020_10.withColumn('2020_10', (df_2020_10[1]/df_2020_10[2])) \
        .select('cbg', '2020_10')
    final = final.join(df_2020_10, on = "cbg", how='full').cache()

    final = final.orderBy('cbg', ascending=True)
    
    final.write.format("csv").option("header", "true").save(sys.argv[1])

if __name__ == '__main__':
  sc = SparkContext()
  #spark = SparkSession(sc)
  sqlcontext=SQLContext(sc)
  main(sc,sqlcontext)
