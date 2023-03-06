import sys
from pyspark.sql import SparkSession

def main(spark):
    # Large dataset
    ratings = spark.read.csv(f'hdfs:/user/wl2733/ratings_large.csv', schema='userId INT, movieId INT, rating FLOAT, timestamp STRING')
    movies = spark.read.csv(f'hdfs:/user/wl2733/movies_large.csv', schema= 'movieId INT, title STRING, genres STRING')
    
    # Small dataset
    # ratings = spark.read.csv("/scratch/work/courses/DSGA1004-2021/movielens/ml-latest/ratings.csv")
    # movies = spark.read.csv("/scratch/work/courses/DSGA1004-2021/movielens/ml-latest/movies.csv")

    # Convert CSV to Parquet
    ratings.write.parquet('hdfs:/user/wl2733/ratings_large.parquet')
    movies.write.parquet('hdfs:/user/wl2733/movies_large.parquet')

    
if __name__ == "__main__":
    spark = SparkSession.builder.appName('project').getOrCreate()
    main(spark)
    