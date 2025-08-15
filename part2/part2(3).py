import matplotlib.pyplot as plt
import numpy as np
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RankingEvaluator
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, expr
from pyspark.storagelevel import StorageLevel
import os

def main(spark, userID):
    # ---------------------- Load Data ----------------------
    THRESHOLD = 3
    train_set = spark.read.parquet(f'hdfs:/user/{userID}/train.parquet')\
        .repartition(400, "userId") \
        .persist(StorageLevel.MEMORY_AND_DISK)

    val_set = spark.read.parquet(f'hdfs:/user/{userID}/val.parquet') \
        .repartition(400, "userId")

    test_set = spark.read.parquet(f'hdfs:/user/{userID}/test.parquet') \
        .repartition(400, "userId")

    SEED = 5024
    regParams = [0.2] # can be modifed for tuning
    ranks = [20] # can be modifed for tuning
    
    
    map_scores_per_reg = {reg: [] for reg in regParams}
    map_scores_per_rank = {rank: [] for rank in ranks}
    best_model, best_rank, best_reg, best_MAP = None, None, None, -1

    # Get actual items (validation)
    actual = val_set.filter(col("rating") >= THRESHOLD) \
        .groupBy("userId") \
        .agg(collect_list("movieId").alias("actual")) \
        .withColumn("actual", expr("transform(actual, x -> cast(x as double))")) \
        .repartition(400, "userId").persist(StorageLevel.MEMORY_AND_DISK)

    evaluator = RankingEvaluator(predictionCol="predicted", labelCol="actual", metricName="meanAveragePrecision")

    for rank in ranks:
        for reg in regParams:
            als = ALS(
                userCol="userId",
                itemCol="movieId",
                ratingCol="rating",
                rank=rank,
                regParam=reg,
                coldStartStrategy="drop",
                nonnegative=True,
                implicitPrefs=True,
                seed=SEED,
                maxIter=12,
            )
            model = als.fit(train_set)

            predictions = model.recommendForAllUsers(100) \
                .withColumn("predicted", expr("transform(recommendations.movieId, x -> cast(x as double))")) \
                .select("userId", "predicted").repartition(400, "userId")

            ranked_val = predictions.join(actual, on="userId").select("predicted", "actual")
            map_score = evaluator.evaluate(ranked_val)
            map_scores_per_reg[reg].append(map_score)
            map_scores_per_rank[rank].append(map_score)
            if map_score > best_MAP:
                best_model, best_rank, best_reg, best_MAP = model, rank, reg, map_score

    print(f"\n✅ Best Model => Rank: {best_rank}, RegParam: {best_reg}, MAP: {best_MAP}")
    # ---------------------- Plot MAP scores ----------------------
    plt.figure(figsize=(10, 6))
    for reg in regParams:
        plt.plot(ranks, map_scores_per_reg[reg], label=f"Reg: {reg}")
    plt.title("MAP vs. Ranks for Regularization Parameter (regParam)")
    plt.xlabel("Ranks")
    plt.ylabel("Mean Average Precision (MAP)")
    plt.legend(title="Reg")
    plt.grid(True)
    plt.savefig("MAP_plot_Reg.png")
    os.system(f"hdfs dfs -put -f MAP_plot_Reg.png /user/{userID}/MAP_plot.png")
    
    plt.figure(figsize=(10, 6))
    for rank in ranks:
        plt.plot(regParams,  map_scores_per_rank[rank], label=f"Rank: {rank}")
    plt.title("MAP vs. Reg for Ranks")
    plt.xlabel("Reg")
    plt.ylabel("Mean Average Precision (MAP)")
    plt.legend(title="Rank")
    plt.grid(True)
    plt.savefig("MAP_plot_Rank.png")
    os.system(f"hdfs dfs -put -f MAP_plot_Rank.png /user/{userID}/MAP_plot.png")
   # ---------------------- Evaluate on Training Set ----------------------
    predictions = best_model.recommendForAllUsers(100) \
       .withColumn("predicted", expr("transform(recommendations.movieId, x -> cast(x as double))")) \
       .select("userId", "predicted")

    actual_train = train_set.filter(col("rating") >= THRESHOLD) \
       .groupBy("userId") \
       .agg(collect_list("movieId").alias("actual")) \
       .withColumn("actual", expr("transform(actual, x -> cast(x as double))"))

    ranked_train = predictions.join(actual_train, on="userId").select("predicted", "actual")
    map_score_test = evaluator.evaluate(ranked_train)
    print(f"✅ Best Model on train set => MAP: {map_score_test}")
    
   # ---------------------- Evaluate on Test Set ----------------------

    actual_test = test_set.filter(col("rating") >= THRESHOLD) \
        .groupBy("userId") \
        .agg(collect_list("movieId").alias("actual")) \
        .withColumn("actual", expr("transform(actual, x -> cast(x as double))"))

    ranked_test = predictions.join(actual_test, on="userId").select("predicted", "actual")
    map_score_test = evaluator.evaluate(ranked_test)
    print(f"✅ Best Model on test set => MAP: {map_score_test}")




if __name__ == "__main__":
    spark = SparkSession.builder.appName("MovieRecommendationOptimized").getOrCreate()
    userID = "hx2504_nyu_edu"
    main(spark, userID)


