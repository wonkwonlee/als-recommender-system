import getpass
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.mllib.evaluation import  RankingMetrics
from pyspark.sql.functions  import collect_list
from time import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

spark = SparkSession.builder.appName('baseline').getOrCreate()
netID = getpass.getuser()
sc = spark.sparkContext

test = spark.read.parquet('hdfs:/user/{}/test.parquet'.format(netID))
test.createOrReplaceTempView('test')

df = spark.read.csv('hdfs:/user/{}/baseline.csv'.format(netID))
# df = df.withColumn(df.cast('double'))
df.cache()

df_collected = df.collect()
model = NearestNeighbors(n_neighbors=100, algorithm='brute')
model = model.fit(df_collected)
bc = sc.broadcast(model)

result = df.rdd.map(lambda x: bc.value.kneighbors(x.values, return_distance=False))
# result = result.collect()
# result.toDF()
# result.show()

test = spark.sql('SELECT userId, movieId FROM test SORT BY rating DESC')
temp = test.groupBy('userId').agg(collect_list('movieId').alias('recommendations'))
temp.createOrReplaceTempView('temp')
true_label = spark.sql('SELECT temp.recommendations FROM temp')
true_label = true_label.collect()

predAndLabel = []
for row in result.collect():
    prediction = [x[1:] for x in row]
    label = true_label[row]
    predAndLabel += [(prediction, label)]

predAndLabel = sc.parallelize(predAndLabel)
evaluator = RankingMetrics(predAndLabel)

map = evaluator.meanAveragePrecision
# mapk = evaluator.meanAveragePrecisionAt(100)
ndcg = evaluator.ndcgAt(100)

print("Evaluation on test data")
print("Mean Average Precision: {}".format(map))
# print("Mean Average Precision at 100: {}".format(mapk))
print("NDCG at 100: {}".format(ndcg))