import getpass
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.mllib.evaluation import  RankingMetrics
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql.functions  import collect_list
import time 

spark = SparkSession.builder.appName('evaluation').getOrCreate()
netID = getpass.getuser()
sc = spark.sparkContext

# Load ALS model from HDFS
# model = ALSModel.load('hdfs:/user/{}/50_0.05_20_model'.format(netID))

train = spark.read.parquet('hdfs:/user/{}/train.parquet'.format(netID))
test = spark.read.parquet('hdfs:/user/{}/test.parquet'.format(netID))
test.createOrReplaceTempView('test')

start_time = time.time()
als = ALS(rank=50, regParam=0.05, maxIter=20, userCol='userId', itemCol='movieId', ratingCol='rating', nonnegative=True ,coldStartStrategy='drop')
model = als.fit(train)


# Evaluate model on test data
# preds = model.recommendForAllUsers(100)
preds = model.recommendForUserSubset(test, 100)
preds.createOrReplaceTempView('preds')
test = spark.sql('SELECT userId, movieId FROM test SORT BY rating DESC')
temp = test.groupBy('userId').agg(collect_list('movieId').alias('recommendations'))
temp.createOrReplaceTempView('temp')
# temp.show(10)

predRecommend = spark.sql('SELECT preds.recommendations, temp.recommendations FROM temp JOIN preds ON preds.userId = temp.userId')
# predRecommend = predRecommend.collect()
predAndLabel = []
for row in predRecommend.collect():
    prediction = [x.movieId for x in row[0]]
    label = row[1]
    predAndLabel += [(prediction, label)]
predAndLabel = sc.parallelize(predAndLabel)


evaluator = RankingMetrics(predAndLabel)
# evaluator.meanAveragePrecision
map = evaluator.meanAveragePrecision
mapk = evaluator.meanAveragePrecisionAt(100)
ndcg = evaluator.ndcgAt(100)

print("Evaluation on test data")
print("Mean Average Precision: {}".format(map))
print("Mean Average Precision at 100: {}".format(mapk))
print("NDCG at 100: {}".format(ndcg))
print('Total time: {}'.format(time.time() - start_time))