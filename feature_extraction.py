import pandas as pd

# Load data
df = pd.read_csv("X_train.csv")

# Create binary columns for counting events
df["is_add"] = (df["action"] == "A").astype(int)
df["is_delete"] = (df["action"] == "D").astype(int)
df["is_update"] = (df["action"] == "U").astype(int)

df["is_bid"] = (df["side"] == "B").astype(int)
df["is_ask"] = (df["side"] == "A").astype(int)

df["is_trade"] = df["trade"].astype(int)

df["spread"] = df["ask"] - df["bid"]

df["imbalance"] = (
    (df["bid_size"] - df["ask_size"]) /
    (df["bid_size"] + df["ask_size"])
)


# Venues: there are 5 venues in the data
# Convert the categorical column "venue" into one-hot encoded columns.
#
# Example:
# df["venue"]:
# 4
# 1
# 5
# 4
#
# becomes:
# venue_1  venue_4  venue_5
#    0        1        0
#    1        0        0
#    0        0        1
#    0        1        0

venue_features = pd.get_dummies(df["venue"], prefix="venue")
# add the features back to df
df = pd.concat([df, venue_features], axis=1)


# Aggregate features: one row per obs_id
features = (
    df.groupby("obs_id")
      .agg(
          # Syntax:
          # new_feature_name = ("column_to_use", "aggregation_function")
          #
          # Example:
          # n_events = ("action", "size")
          # means: count how many rows/events exist in each obs_id group.
          
        n_events=("action", "size"),

        n_add=("is_add", "sum"),
        n_delete=("is_delete", "sum"),
        n_update=("is_update", "sum"),

        n_bid=("is_bid", "sum"),
        n_ask=("is_ask", "sum"),
        n_trades=("is_trade", "sum"),

        mean_flux=("flux", "mean"),
        std_flux=("flux", "std"),

        mean_price=("price", "mean"),
        std_price=("price", "std"),

        mean_bid_size=("bid_size", "mean"),
        mean_ask_size=("ask_size", "mean"),

        

        # Venue "profile"
        venue_0=("venue_0", "sum"),
        venue_1=("venue_1", "sum"),
        venue_2=("venue_2", "sum"),
        venue_3=("venue_3", "sum"),
        venue_4=("venue_4", "sum"),
        venue_5=("venue_5", "sum"),

        ## Order book imbalance
        mean_imbalance=("imbalance", "mean"),
        std_imbalance=("imbalance", "std"),
        min_imbalance=("imbalance", "min"),
        max_imbalance=("imbalance", "max"),

        # Spread features
        mean_spread=("spread", "mean"),
        std_spread=("spread", "std"),
        min_spread=("spread", "min"),
        max_spread=("spread", "max"),
      )
      .reset_index()
)

# Save features
features.to_csv("X_train_features.csv", index=False)

print(features.head())