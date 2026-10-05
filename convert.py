import duckdb
# pip install duckdb

# convert X_train_N1UvY30.csv into Xtrain.parquet
# X_train.csv in the same folder as this file


'''duckdb.sql("""
COPY (
    SELECT * FROM "X_train.csv"
)
TO 'Xtrain.parquet'
(FORMAT PARQUET, COMPRESSION ZSTD);
""")'''

duckdb.sql("""
COPY (
    SELECT * FROM "X_test.csv"
)
TO 'Xtest.parquet'
(FORMAT PARQUET, COMPRESSION ZSTD);
""")
