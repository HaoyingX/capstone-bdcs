from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.functions import row_number, col, lit, floor,when,rand

# --------------------------- Entry Point ---------------------------

def main(spark, userID):
    SEED = 5024
    ratings_csv = f'hdfs:/user/{userID}/ml-latest/ratings.csv'
    ratings_parquet = f'hdfs:/user/{userID}/ratings.parquet'

    ratings = spark.read.csv(ratings_csv, header=True, inferSchema=True)
    ratings.write.parquet(ratings_parquet, mode="overwrite")
    ratings = spark.read.parquet(ratings_parquet).repartition("userId")
    ratings = ratings.select("userId", "movieId", "rating")

    # Partition each users’ observations into trainl/val set separately


    # Step 1: Add row numbers per user
    window = Window.partitionBy("userId").orderBy(rand(seed = SEED))
    ratings_with_row = ratings.withColumn("row_num", row_number().over(window))

    # Step 2: Count how many ratings per user
    user_counts = ratings_with_row.groupBy("userId").count().withColumnRenamed("count", "user_count")

    # Step 3: Join to get count per row
    ratings_with_count = ratings_with_row.join(user_counts, on="userId")

    # Step 4: Define split thresholds using 6:2:2 ratio
    ratings_split = ratings_with_count.withColumn(
        "split", 
        when(col("row_num") <= floor(col("user_count") * 0.6), "train")
        .when(col("row_num") <= floor(col("user_count") * 0.8), "val")
        .otherwise("test")
    )

    # Step 5: Separate into three DataFrames
    train = ratings_split.filter(col("split") == "train").select("userId", "movieId", "rating")
    val   = ratings_split.filter(col("split") == "val").select("userId", "movieId", "rating")
    test  = ratings_split.filter(col("split") == "test").select("userId", "movieId", "rating")


    train.write.parquet(f'hdfs:/user/{userID}/train.parquet', mode="overwrite")
    val.write.parquet(f'hdfs:/user/{userID}/val.parquet', mode="overwrite")
    test.write.parquet(f'hdfs:/user/{userID}/test.parquet', mode="overwrite")
    print(f"Train: {train.count()}, Val: {val.count()}, Test: {test.count()}")



# --------------------------- Run ---------------------------

if __name__ == "__main__":
    spark = SparkSession.builder.appName("MinHashLSH_UserSimilarity").getOrCreate()
    userID = "hx2504_nyu_edu"
    main(spark, userID)
