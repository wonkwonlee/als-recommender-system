import getpass
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.mllib.evaluation import  RankingMetrics
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions  import collect_list
import time 

spark = SparkSession.builder.appName('1004-Project').getOrCreate()
netID = getpass.getuser()
sc = spark.sparkContext

train = spark.read.parquet('hdfs:/user/{}/train.parquet'.format(netID))
val = spark.read.parquet('hdfs:/user/{}/val.parquet'.format(netID))
# test = spark.read.parquet('hdfs:/user/{}/test.parquet'.format(netID))
val.createOrReplaceTempView('val')

seed = 42
iterations = [10, 20, 30, 40, 50]
regs = [0.01, 0.05, 0.1, 0.5, 1.0]
ranks = [10, 20, 30, 40, 50]

best_rank = -1
best_regularization = -1
best_iter = -1
best_map = -1
best_mapk = -1
best_ndcg = -1
best_model = None

# Grid search 
for iter in iterations:
    for rank in ranks:
        for reg in regs:
            start_time = time.time()
            als = ALS(rank=rank, maxIter=iter, regParam=reg, userCol='userId', itemCol='movieId', ratingCol='rating', nonnegative=True ,coldStartStrategy='drop')
            model = als.fit(train)
            
            # Evaluate model on validation set
            # preds = model.recommendForAllUsers(100)
            preds = model.recommendForUserSubset(val, 100)
            preds.createOrReplaceTempView('preds')
            val = spark.sql('SELECT userId, movieId FROM val SORT BY rating DESC')
            temp = val.groupBy('userId').agg(collect_list('movieId').alias('recommendations'))
            temp.createOrReplaceTempView('temp')
            predRecommend = spark.sql('SELECT preds.recommendations, temp.recommendations FROM temp JOIN preds ON preds.userId = temp.userId')
            predRecommend = predRecommend.collect()
            predAndLabel = []
            for row in predRecommend:
                prediction = [x.movieId for x in row[0]]
                label = row[1]
                predAndLabel += [(prediction, label)]
            predAndLabel = sc.parallelize(predAndLabel)
            
            # Ranking metrics
            evaluator_rank = RankingMetrics(predAndLabel)
            map = evaluator_rank.meanAveragePrecision
            mapk = evaluator_rank.meanAveragePrecisionAt(100)
            ndcg = evaluator_rank.ndcgAt(100)
            
            print('Rank: {}, regularization parameter: {}, max iterations: {}'.format(rank, reg, iter))
            print('MAP: {}, MAP@100: {}, NDCG@100: {}'.format(map, mapk, ndcg))
            print('Total time: {}'.format(time.time() - start_time))
            
            if map > best_map:
                best_rank = rank
                best_regularization = reg
                best_iter = iter
                best_model = model
                best_map = map
                best_mapk = mapk
                best_ndcg = ndcg
            
# Best model on validation set
print('Best model hyperparameters = rank: {}, regularization parameter: {}, max iterations: {}'.format(best_rank, best_regularization, best_iter))
print('MAP: {}, MAP@100: {}, NDCG@100: {}'.format(best_map, best_mapk, best_ndcg))
best_model.write().overwrite().save(f'hdfs:/user/wl2733/{best_rank}_{best_regularization}_{best_iter}_model')
