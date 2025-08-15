from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, rand
import matplotlib.pyplot as plt
import time

def compute_avg_correlation(spark, ratings_df, user_pairs_df, label):
    print(f"[{label}] Joining pairs with ratings...")

    r1 = ratings_df.alias('r1')
    r2 = ratings_df.alias('r2')

    # Add ratings to user pairs where both users rated the same movie
    joined = user_pairs_df \
        .join(r1, col('user1') == col('r1.userId')) \
        .join(r2, (col('user2') == col('r2.userId')) & (col('r1.movieId') == col('r2.movieId'))) \
        .select('user1', 'user2', col('r1.movieId').alias('movieId'),
                col('r1.rating').alias('rating1'), col('r2.rating').alias('rating2'))

    # Group by user pairs and compute Pearson correlation
    print(f"[{label}] Computing Pearson correlations...")
    corr_df = joined.groupBy('user1', 'user2') \
                    .agg(expr('corr(rating1, rating2)').alias('corr')) \
                    .na.drop()

    # Compute average correlation
    avg_corr = corr_df.agg({'corr': 'avg'}).collect()[0][0]
    print(f"[{label}] Average correlation: {avg_corr:.4f}")
    return avg_corr

def main():
    spark = SparkSession.builder.appName("part2_threshold100").getOrCreate()

    RATINGS_PATH = "hdfs:/user/yz6553_nyu_edu/target/ml-latest/ratings.csv"
    TOP100_PATH = "hdfs:/user/yz6553_nyu_edu/result1_threshold40.csv"
    LOCAL_IMAGE_PATH = "/home/yz6553_nyu_edu/capstone-bdcs_66/part2_threshold40.jpg"

    start_time = time.time()

    # Load all data needed
    print("Loading ratings data...")
    ratings = spark.read.csv(RATINGS_PATH, header=True, inferSchema=True) \
        .select('userId', 'movieId', 'rating')

    print("Loading top100 similar pairs (threshold=100)...")
    top100_pairs = spark.read.csv(TOP100_PATH, header=True, inferSchema=True)

    print("Computing correlation for Top 100 pairs...")
    top_corr = compute_avg_correlation(spark, ratings, top100_pairs, "TOP100")

    # Sampling 100 random user pairs
    print("Sampling 100 random user pairs...")
    user_ids = ratings.select('userId').distinct().orderBy(rand()).limit(200).rdd.map(lambda r: r[0]).collect()
    random_pairs = spark.createDataFrame([(user_ids[i], user_ids[i+1]) for i in range(0, 200, 2)],
                                         ['user1', 'user2'])

    # Compute correlation for 100 random pairs
    print("Computing correlation for Random 100 pairs...")
    random_corr = compute_avg_correlation(spark, ratings, random_pairs, "RANDOM100")

    # Plotting and saving the results
    print("Plotting and saving result...")
    categories = ['Top 100 Similar Pairs', 'Random 100 Pairs']
    values = [top_corr, random_corr]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(categories, values, color=['blue', 'gray'])
    plt.ylabel('Average Pearson Correlation')
    plt.title('Threshold=40: Validation of User Similarity via Ratings Correlation')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')
    plt.ylim(0, max(values) + 0.1)
    plt.tight_layout()
    plt.savefig(LOCAL_IMAGE_PATH)
    print(f"Finish in {time.time() - start_time:.2f}s")
    print(f"\n=== FINAL RESULTS ===")
    print(f"Top 100 Avg Correlation: {top_corr:.4f}")
    print(f"Random 100 Avg Correlation: {random_corr:.4f}")
    print(f"Figure saved as: {LOCAL_IMAGE_PATH}")

    spark.stop()

if __name__ == "__main__":
    main()

