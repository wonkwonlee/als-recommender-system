# Reference for data partition: https://github.com/yuwei-jacque-wang/Recommender-System-DSGA1004/blob/master/data.py

import sys
from pyspark.sql import SparkSession

def main(spark):
    ratings = spark.read.parquet(f'hdfs:/user/wl2733/ratings_large.parquet')
    movies = spark.read.parquet(f'hdfs:/user/wl2733/movies_large.parquet')
    ratings.createOrReplaceTempView('ratings')
    
    ratings_filter = spark.sql('SELECT * FROM ratings WHERE rating != 0')
    ratings_filter = ratings_filter.drop('timestamp')
    ratings_filter.createOrReplaceTempView('ratings_filter')
    users = spark.sql('SELECT DISTINCT userId FROM ratings_filter GROUP BY userId HAVING count(*) > 10')
    users.createOrReplaceTempView('users')
    
    # Data partition
    train_user, val_user, test_user = users.randomSplit([0.6, 0.2, 0.2])
    train_user.createOrReplaceTempView('train_user')
    val_user.createOrReplaceTempView('val_user')
    test_user.createOrReplaceTempView('test_user')

    train_rating = spark.sql('SELECT * FROM ratings_filter WHERE userId IN (SELECT userId FROM train_user)')
    val_rating = spark.sql('SELECT * FROM ratings_filter WHERE userId IN (SELECT userId FROM val_user)')
    test_rating = spark.sql('SELECT * FROM ratings_filter WHERE userId IN (SELECT userId FROM test_user)')
    
    # Validation set
    val_rating_rdd = val_rating.rdd.zipWithIndex()
    val_rating_df = val_rating_rdd.toDF()
    val_rating_df = val_rating_df.withColumn('userId', val_rating_df['_1'].getItem('userId'))
    val_rating_df = val_rating_df.withColumn('movieId', val_rating_df['_1'].getItem('movieId'))
    val_rating_df = val_rating_df.withColumn('rating', val_rating_df['_1'].getItem('rating'))
    
    temp_val = val_rating_df.select('_2', 'userId', 'movieId', 'rating')
    temp_val.createOrReplaceTempView('temp_val')
    temp_val0 = spark.sql('SELECT * FROM temp_val WHERE _2 %2 = 0')
    temp_val1 = spark.sql('SELECT * FROM temp_val WHERE _2 %2 = 1')
    temp_val0 = temp_val0.drop('_2')
    temp_val1 = temp_val1.drop('_2')
    temp_val0.createOrReplaceTempView('temp_val0')
    temp_val1.createOrReplaceTempView('temp_val1')
    
    # Test set
    test_rating_rdd = test_rating.rdd.zipWithIndex()
    test_rating_df = test_rating_rdd.toDF()
    test_rating_df = test_rating_df.withColumn('userId', test_rating_df['_1'].getItem('userId'))
    test_rating_df = test_rating_df.withColumn('movieId', test_rating_df['_1'].getItem('movieId'))
    test_rating_df = test_rating_df.withColumn('rating', test_rating_df['_1'].getItem('rating'))
    
    temp_test = test_rating_df.select('_2', 'userId', 'movieId', 'rating')
    temp_test.createOrReplaceTempView('temp_test')
    temp_test0 = spark.sql('SELECT * FROM temp_test WHERE _2 %2 = 0')
    temp_test1 = spark.sql('SELECT * FROM temp_test WHERE _2 %2 = 1')
    temp_test0 = temp_test0.drop('_2')
    temp_test1 = temp_test1.drop('_2')
    temp_test0.createOrReplaceTempView('temp_test0')
    temp_test1.createOrReplaceTempView('temp_test1')
    
    # Final overlapped train, validation, test set
    train_final = train_rating.union(temp_val0).union(temp_test0)
    val_final = temp_val1
    test_final = temp_test1
    
    # Save final dataset
    train_final.write.parquet('hdfs:/user/wl2733/train_large.parquet')
    val_final.write.parquet('hdfs:/user/wl2733/val_large.parquet')
    test_final.write.parquet('hdfs:/user/wl2733/test_large.parquet')
    

if __name__ == "__main__":
    spark = SparkSession.builder.appName('project').getOrCreate()
    main(spark)
    