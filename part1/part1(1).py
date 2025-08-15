import os
import time
import random
import hashlib
from itertools import combinations
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, size

# === Parameters ===
NUM_BLOCKS = 8 # Number of bands (for LSH)
ROWS_PER_BLOCK = 16 # Hash functions per band
NUM_HASHES = NUM_BLOCKS * ROWS_PER_BLOCK # Total number of hash functions
PRIME = 2**31 - 1
SEED = 2504
MIN_MOVIE_THRESHOLD = 40 # Minimum number of movies per user to be considered

# === Hash Functions ===
def generate_hash_functions(n, seed):
    random.seed(seed)
    # Create n different hash functions of the form h(x) = (a*x + b) % PRIME
    return [(lambda a, b: lambda x: (a * x + b) % PRIME)(random.randint(1, PRIME - 1), random.randint(0, PRIME - 1)) for _ in range(n)]

# === Signature Computation ===
def compute_minhash_signature(movie_list, hash_funcs):
    # Compute the complete MinHash signature vector for a user's set of movies
    return [min(h(m) for m in movie_list) for h in hash_funcs] if movie_list else [0] * len(hash_funcs)

# === Band Hashing ===
def lsh_band_hash(band):
    # Hash a band (a subvector of the signature) into a single integer using SHA1
    return int(hashlib.sha1("_".join(map(str, band)).encode('utf-8')).hexdigest(), 16)

# === Band Mapper ===
def band_mapper(user_id, movie_list, signature):
    # seperate the signature into bands and hash each band
    for block_idx in range(NUM_BLOCKS):
        start = block_idx * ROWS_PER_BLOCK
        band = signature[start:start + ROWS_PER_BLOCK]
        # generate (band_index, band hash) → (user_id, movie_set) key-value pairs
        # note: (band_index, band hash) is the bucket key
        yield ((block_idx, lsh_band_hash(band)), (user_id, set(movie_list)))

# === Pair Generator ===
def generate_candidate_pairs(user_list):
    # Generate candidate pairs from the list of users in same bucket
    return list(combinations(sorted(user_list, key=lambda x: x[0]), 2)) if len(user_list) > 1 else []

# === Main Function ===
def main(spark, userID):
    start_time = time.time()
    print('Starting result1.py...')

    # Load the data from HDFS
    print('Loading ratings from HDFS...')
    ratings_path = f'hdfs:/user/{userID}/target/ml-latest/ratings.csv'
    ratings = spark.read.csv(ratings_path, header=True, inferSchema=True)
    ratings = ratings.select('userId', 'movieId').repartitionByRange("userId")

    # Group by userId and collect movieId into a list, 
    # also filter out users with less than MIN_MOVIE_THRESHOLD movies
    print('Grouping movies per user...')
    user_movies = ratings.groupBy("userId").agg(collect_list("movieId").alias("movies"))
    user_movies = user_movies.filter(size("movies") >= MIN_MOVIE_THRESHOLD).cache()
    print(f"Users retained after filtering: {user_movies.count()}")

    # Compute minhas signatures
    print('Computing MinHash signatures...')
    hash_funcs = generate_hash_functions(NUM_HASHES, SEED)
    user_sigs = user_movies.rdd.map(lambda row: (
        row['userId'],
        row['movies'],
        compute_minhash_signature(row['movies'], hash_funcs)
    )).cache()
    print('Signatures computed.')

    # Generate band hashes
    print('Mapping to bands...')
    band_mapped = user_sigs.flatMap(lambda row: band_mapper(row[0], row[1], row[2]))

    # Group by bucket key (band_index, band_hash) and collect user_id and movie_set in the same bucket
    print('Grouping by band hash...')
    grouped = band_mapped.combineByKey(
        lambda x: [x],
        lambda acc, x: acc + [x],
        lambda acc1, acc2: acc1 + acc2
    ).cache()

    # Genrate candidate pairs from the each bucket
    print('Generating candidate pairs...')
    candidates = grouped.flatMap(lambda x: generate_candidate_pairs(x[1]))
    print(f'Candidate pairs generated: {candidates.count()}')

    # Compute Jaccard similarity for each candidate pair
    print('Calculating Jaccard similarity...')
    similarities = candidates.map(lambda pair: (
        pair[0][0],
        pair[1][0],
        len(pair[0][1] & pair[1][1]) / len(pair[0][1] | pair[1][1]) if (pair[0][1] | pair[1][1]) else 0.0
    )).distinct()

    sim_df = similarities.toDF(["user1", "user2", "similarity"])

    print('Selecting top 100 most similar user pairs...')
    top100 = sim_df.orderBy(col("similarity").desc()).limit(100)

    # Save the results
    output_path = f'/home/{userID}/capstone-bdcs_66/result1_threshold40.csv'
    print(f'Saving to local path: {output_path}')
    top100.toPandas().to_csv(output_path, index=False)

    print(f'Done in {time.time() - start_time:.2f}s. Output written to {output_path}')

# === Entry Point ===
if __name__ == '__main__':
    spark = SparkSession.builder.appName("MinHashLSH_UserSimilarity").getOrCreate()
    userID = "yz6553_nyu_edu"
    main(spark, userID)
    spark.stop()

