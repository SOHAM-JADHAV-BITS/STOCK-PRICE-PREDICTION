# STOCK-PRICE-PREDICTION
This project aims to predict Amazon’s stock closing prices using a Long Short-Term Memory (LSTM) network trained on a multivariate time series dataset. The dataset integrates traditional market indicators such as SMA, EMA, and RSI to better capture market momentum, trends, and potential reversal zones like golden/death crosses or overbought/oversold regions. The pipeline includes:
Data cleaning and feature engineering.
Normalization using MinMaxScaler.
Time series visualization.
LSTM modeling with customized activation.
Backtesting and performance visualization.


#Historical daily stock price data was taken from Macrotrends.net, providing a robust long-term dataset for Amazon.

 Cleaning
The CSV was cleaned using Pandas, where missing values were removed (e.g., for rolling windows of SMA/EMA/RSI).
Dates were converted to datetime format and sorted to preserve time series structure.

Feature Engineering
We created the following technical indicators:
SMA (Simple Moving Averages): SMA_50, SMA_100, SMA_200 to capture short/medium/long trends.
EMA (Exponential Moving Averages): EMA_30, EMA_200 to weigh recent price movements more.
RSI_30 (Relative Strength Index): Helps identify overbought (>70) or oversold (<30) zones.


#Using matplotlib, we visualized:
EMA and close price overlay to identify golden/death crosses.
RSI trendline to observe market momentum shifts.
Monthly mean trends for smoothed analysis.
Daily returns % for volatility evaluation.
Files used:
visualization_amazon.py 
visualization_2_amazon.py 
visualisation_3_amazon.py 


#To feed into LSTM, we used MinMaxScaler to normalize:
close, volume, SMA_*, EMA_*, RSI_30
Prevents any one feature (like volume) from dominating due to scale.
Brings uniformity for faster and stable gradient descent.

# SMA and EMA
Golden Cross: When short-term SMA/EMA crosses above a long-term one → bullish signal.
Death Cross: Opposite case → bearish signal.
EMA reacts faster than SMA and is used for timely trend detection.

# RSI
Identifies momentum exhaustion — whether a stock is overbought (likely reversal) or oversold (likely rebound).All of these are critical in technical analysis and help the LSTM detect underlying trend changes, not just raw price movement.

#LSTM is ideal for time-dependent sequential data like stock prices:
It retains long-term dependencies through gated memory cells.Handles vanishing gradient better than traditional RNNs.Suitable for modeling lagged financial indicators like RSI and EMA crossovers.
Hyperparameter fine tuning is done as per accuracy shoot ups and as per the model response , adding a dropout of 0.5 ensures that the model doesnt learn the data to an overextent and is always ready to encounter new concepts at new points 

sliding window technique here also rolls over the entire processed data - how ? -> 0-99 index as input to predict the closing price of 100 th index which is already available for validation , similarly 1-100 again for data and predicting 101st day closing price and so on for  about 7000 enteries for the 25 years.

80% training, 20% testing Shuffle disabled to maintain chronological order (crucial for time series) Window size: 100 → last 100 days of indicators used to predict next closing price.

#Model Evaluation
MSE (Mean Squared Error)
RMSE (Root Mean Squared Error)
MAE (Mean Absolute Error)
R² Score

#Plots:
Training vs Validation Loss
Predicted vs Actual Prices (scatter)
Last 200 days prediction vs real values (line plot)


# GELU WAS USED INSTEAD OF RELU DUE TO FASTER CONVERGENCE 
Why GELU Performs Better for Time Series / Stock Data
Feature	                         ReLU	                              GELU
Smoothness	                    Harsh                             (discontinuous)	Smooth transition
Gradient flow     	            Zero for x < 0	                  Non-zero gradient even for x < 0
Captures subtleties	          Ignores small negative signals	   Retains and weighs them
Helps with vanishing gradient	 No	                               Yes (like swish or ELU)
Used in...	Standard CNNs, RNNs	Transformers (BERT, GPT), LSTM extensions











