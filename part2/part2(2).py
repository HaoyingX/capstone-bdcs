from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, udf
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml.evaluation import RankingEvaluator
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------------------
def MAP_Validation(movie_score_count, beta, val_set, spark):
    # Compute regularized average score: sum / (count + beta)
    movie_highest_rating = movie_score_count \
        .map(lambda x: (x[0], x[1][0] / (x[1][1] + beta))) \
        .sortBy(lambda x: x[1], ascending=False) \
        .take(100)

    top_highest_rated = [x[0] for x in movie_highest_rating]
    top_highest_rated = spark.sparkContext.broadcast(top_highest_rated)

    # Step 1: Actual relevant items per user from validation set (rating >= 3)
    val_set_movielst_per_user = val_set.rdd \
        .filter(lambda x: x[2] >= 3) \
        .map(lambda x: (x[0], float(x[1]))) \
        .combineByKey(
            lambda x: [x],
            lambda acc, x: acc + [x],
            lambda acc1, acc2: acc1 + acc2
        ) \
        .toDF(["user_id", "movie_list"])

    # Step 2: Add predicted movie list (broadcasted top 100)
    val_set_with_predictions = val_set_movielst_per_user \
        .withColumn("highest_rated_predictions", lit(top_highest_rated.value))

    # Step 3: Ensure predictions are float type
    def convert_to_double(array_of_ints):
        return [float(x) for x in array_of_ints]

    convert_to_double_udf = udf(convert_to_double, ArrayType(DoubleType()))
    val_set_with_predictions = val_set_with_predictions \
        .withColumn("highest_rated_predictions", convert_to_double_udf("highest_rated_predictions"))

    # Step 4: Evaluate MAP
    evaluator = RankingEvaluator(
        predictionCol="highest_rated_predictions",
        labelCol="movie_list",
        metricName="meanAveragePrecision"
    )
    popular_map = evaluator.evaluate(val_set_with_predictions)

    return popular_map

# ---------------------------------------
def main(spark, userID):
    # Coarser beta grid for efficiency
    BETA_LIST = [90000] # can be modified for tunning

    # ---------------------- Load Data ----------------------
    train_set = spark.read.parquet(f'hdfs:/user/{userID}/train.parquet')
    val_set = spark.read.parquet(f'hdfs:/user/{userID}/val.parquet')
    test_set = spark.read.parquet(f'hdfs:/user/{userID}/test.parquet')

    # ---------------------- Movie Statistics ----------------------
    # RDD of (movie_id, (rating_sum, count))
    movie_score_count = train_set.rdd.map(lambda x: (x[1], x[2])) \
        .combineByKey(
            lambda x: (x, 1),
            lambda acc, x: (acc[0] + x, acc[1] + 1),
            lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])
        )
    
    # ---------------------- MAP Evaluation Loop on Validation ----------------------
    beta_lst = []
    map_lst = []

    for beta in BETA_LIST:
        print(f"Evaluating beta = {beta}")
        popular_map = MAP_Validation(movie_score_count, beta, val_set, spark)
        beta_lst.append(beta)
        map_lst.append(popular_map)

    # ---------------------- Plot Results ----------------------
    plt.figure(figsize=(10, 6))
    plt.plot(beta_lst, map_lst, marker='o', linestyle='-')
    plt.xlabel('Beta')
    plt.ylabel('Mean Average Precision (MAP)')
    plt.title('MAP vs Beta in Popularity Baseline Model')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('beta_fitting.png')
    os.system(f"hdfs dfs -put -f beta_fitting.png /user/{userID}/beta_fitting.png")

    # ---------------------- Report Best Beta ----------------------
    best_index = np.argmax(map_lst)
    best_beta = beta_lst[best_index]
    best_map = map_lst[best_index]
    print(f"\n✅ Best Beta from validation: {best_beta} with MAP: {best_map:.4f}")

    # ---------------------- Compute MAP on Train & Test ----------------------
    map_train = MAP_Validation(movie_score_count, best_beta, train_set, spark)
    map_test = MAP_Validation(movie_score_count, best_beta, test_set, spark)

    print(f"📊 MAP on Train Set: {map_train:.4f}")
    print(f"📊 MAP on Test Set:  {map_test:.4f}")

    

# --------------------------- Run ---------------------------

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Popular").getOrCreate()
    userID = "hx2504_nyu_edu"
    main(spark, userID)





