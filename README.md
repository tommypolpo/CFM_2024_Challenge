# Challenge Data - CFM 2024

## Challenge goals

The aim of this challenge is to attempt to identify from which stock a piece of tick-by-tick exchange data belongs. The problem is thus a classification task. The exchange data includes each atomic update of the order-book giving information about the best bid and offer prices, any trades that may have occurred, orders that have been placed in the book or cancelled. The order-book is also an aggregated order-book that is built out of multiple exchanges from where we can buy and sell the same shares. 

## Data description

**X**: One instance in the dataset consists of 100 sequential order-book observations: for convenience, we will possibly call one such instance window, or short movie, or observation. Each observation has 100 rows of data: these
are the timestamps. Each row in the dataset corresponds to **one atomic event** that modified the order book. An observation (`obs_id`) is simply a sequence of 100 such events.

There are 20 observations randomly taken per stock and per day. There are 504 days in the dataset (approximately 2 years) and 24 stocks. This means there are 100 x 20 x 504 x 24 rows of data = 24'192'000. 

The columns correspond to the following items:
* **obs_id**: uniquely identifies one observation: a sequence of 100 order book events drawn from a random stock inside a random day; **All rows with same obs_id concern the same stock on the same day**.
* **venue**: The exchange on which the event occurs. It can be NASDAQ, BATY, etc. but they are just encoded in the data as integers.
* **action**: This is type of order-book event that occurred, it can be ‘A’, ‘D’ or ‘U’. A means volume was added to the book in the form of a new order. ‘D’ means an order was deleted from the book and ‘U’ means an order was updated.
* **order_id**: A unique identifier of an order **within a single observation**. The ids have been re-numbered independently for each observation, so `order_id = 8` in one observation has nothing to do with `order_id = 8` in another.
This allows you to track the life of an order:
    - when it is inserted,
    - when it is updated,
    - when it is deleted,
    - whether it disappears because it traded or because it was cancelled.

* **side**: The side of the order-book on which the event took place ‘A’ (Ask, sell) or ‘B’ (Bid, buy). Buy orders always live on the bid side, while sell orders live on the ask side. 
* **price**: The price of the order that was modified.
The important point is that **this is not the absolute stock price**.
To prevent the classification task from becoming trivial, the organizers subtracted the best bid price of the first event in the observation from every price.
* **bid**: The price of the best bid.
* **ask**: The price of the best ask.
* **bid_size**: The volume of orders at the best bid of the aggregated book.
* **ask_size**: The volume of orders at the best ask of the aggregated book.
* **flux**: The change in quantity caused by this event.
Positive values mean liquidity was added.
Negative values mean liquidity was removed.
Its interpretation depends on the action:
    - `A`: positive flux (new shares entered the book).
    - `D`: negative flux (shares left the book).
    - `U`: can be positive or negative depending on whether the order increased or decreased in size.
    
* **trade**: A boolean true or false to indicate whether a deletion or update event was due to a trade (True) or due to a cancellation/update (False). 

### Observations
* If `trade` is True, the `flux` can only be negative: 'trades consume liquidity'

| **`action`** | **`flux`** | **`trade`** | **Market Meaning** |
|:-:|:-:|:-:|:-:|
| **A (Add)** | Positive (+100) | False | A new limit order was placed in the book. |
| **D (Delete)** | Negative (-100) | False | A trader **cancelled** their order. |
| **D (Delete)** | Negative (-100) | True | A market order hit this limit order and **executed a trade**. |
| **U (Update)** | Positive (+50) | False | A trader increased their existing order size. |
| **U (Update)** | Negative (-50) | True or False | A partial fill occurred (True) or size was manually lowered (False). |

* Example 1:  Trader added (`A`) an ask (side A, selling) at price 0.3 of 100 units

| action | side | price | bid | ask  | bid_size | ask_size | trade | flux |
|--------|------|-------|-----|------|----------|----------|-------|------|
| A      | A    | 0.3   | 0.0 | 0.01 | 100      | 1        | False | 100  |

* Example 2: Trader Deleted (`D`) an ask (side A, selling) at price 0.28 of 100 units

| action | side | price | bid | ask  | bid_size | ask_size | trade | flux |
|--------|------|-------|-----|------|----------|----------|-------|------|
| D      | A    | 0.28  | 0.0 | 0.01 | 100      | 1        | False | -100 |


## Files

* `convert.py`: convert `X_train.csv` (1.2GB) to a 70MB file called `Xtrain.parquet` that we can quickly upload to Google Drive and open it using Colab. (It can also be applied to `X_test.csv`)
* `feature_extraction.py`. **Old version** of the feature extraction tool (that is, extract information from each 100-steps observation, e.g. average price).
* `old_model.py`. Old version of our model. Useless now.

## Models

#### `gru_model.pt` 
* 2 bidirectional GRU layers: input dimension 19, hidden dimension 64
* 1 linear layer 128-->24
* `venue`, `action`, `side`, `trade` 1-hot encoded (13 dimensions)
* `price`, `bid`, `ask`, `bid_size`, `ask_size`, `flux` not normalized
* `obs_id`, `order_id` dropped
* 30 epochs training
* Training metric: 61%
* Validation metric 61%
* **Public score:** 0.36392156862745095

#### `gru_model2_scaled.pt` 
* 2 bidirectional GRU layers: input dimension 19, hidden dimension 64
* 1 linear layer 128-->24
* `venue`, `action`, `side`, `trade` 1-hot encoded (13 dimensions)
* `price`, `bid`, `ask`, `bid_size`, `ask_size`, `flux` normalized over the training set (not validation)
* `obs_id`, `order_id` dropped
* 40 epochs training
* Training metric: 75.88%
* Validation metric 66.02%

#### `inception_scaled_1.pt`
* inception + pooling + linear
* inception layer with kernels: top 1,3,5,1 and bottom 1,1,3. 
* permutation: `[batch, 100, 19]-->[batch, 19, 100]`
* each inception branch: `[batch, 19, 100]-->[batch, 4, 100]`
* inception layer: `[batch, 19, 100]-->[batch, 16, 100]`
* pooling layer: `[batch, 16, 100]-->[batch, 16]`
* linear layer `[batch, 16]-->[batch, 24]`
* `venue`, `action`, `side`, `trade` 1-hot encoded (13 dimensions)
* `price`, `bid`, `ask`, `bid_size`, `ask_size`, `flux` normalized over the training set (not validation)
* `obs_id`, `order_id` dropped
* 80 epochs training
* Training metric: 30.13%
* Validation metric 30.01% 

#### `inception_scaled2.pt`
Notice that here each branch does not act as a bottleneck: the output still has 16 channels.
* inception + pooling + 2 linear layers
* inception layer with kernels: top 1,3,5,1 and bottom 1,1,3
* permutation: `[batch, 100, 19]-->[batch, 19, 100]`
* each inception branch: `[batch, 19, 100]-->[batch, 16, 100]`
* inception layer: `[batch, 19, 100]-->[batch, 64, 100]`
* pooling layer: `[batch, 64, 100]-->[batch, 64]`
* linear layers: `[batch, 64]-->[batch, 32]-->[batch, 24]`
* `venue`, `action`, `side`, `trade` 1-hot encoded (13 dimensions)
* `price`, `bid`, `ask`, `bid_size`, `ask_size`, `flux` normalized over the training set (not validation)
* `obs_id`, `order_id` dropped
* 40 epochs training
* Training metric: 45.9%
* Validation metric: 45.06%

#### `stacked_inception1.pt`
* 2 stacked inception layers + pooling + 2 linear layers
* each inception layer with kernels: top 1,3,5,1 and bottom 1,1,3
* permutation: `[batch, 100, 19]-->[batch, 19, 100]`
* first inception layer: `[batch, 19, 100]-->[batch, 64, 100]`
* second inception layer: `[batch, 64, 100]-->[batch, 64, 100]`
* pooling layer: `[batch, 64, 100]-->[batch, 64]`
* linear layers: `[batch, 64]-->ReLu-->[batch, 32]-->[batch, 24]`
* `venue`, `action`, `side`, `trade` 1-hot encoded (13 dimensions)
* `price`, `bid`, `ask`, `bid_size`, `ask_size`, `flux` normalized over the training set (not validation)
* `obs_id`, `order_id` dropped
* 100 epochs
* Training metric: 63.75%
* Validation metric: 61.91%