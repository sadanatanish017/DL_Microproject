📈 Bitcoin Price Prediction using LSTM + Temporal Fusion Transformer (TFT)

This project implements a hybrid deep learning pipeline to predict Bitcoin next-minute prices using:

✔ LSTM-based temporal embeddings

✔ A lightweight custom Temporal Fusion Transformer (TFT)

✔ Scaled OHLCV features

✔ Time-based covariates (hour, weekday, month)

The workflow supports scalable training, embedding generation, and returns-based price forecasting.

🚀 Project Overview
1. Data Source

Minute-level BTC/USDT data sourced from CryptoDataDownload (Binance).

2. LSTM Encoder

A 60-step sliding-window LSTM is trained to generate dense temporal embeddings that capture short-term price dynamics.

3. Embedding Generation

All embeddings (~1.5M rows) are generated efficiently and merged into the original dataset for downstream modeling.

4. Hybrid TFT Model

A custom PyTorch implementation of the Temporal Fusion Transformer predicts next-step returns, which are then converted into final price predictions.

5. Evaluation & Visualization

Performance is evaluated using RMSE, MAE, MAPE, direction accuracy, and multiple diagnostic charts (scatter & line plots).

📊 Final Model Performance
Metric	Value
Samples Evaluated	150,528
RMSE (price units)	7,713.99
MAE (price units)	6,634.98
MAPE (%)	22.82%
Direction Accuracy	0.75%
True Up Movements	73,899
True Down Movements	75,409
Model Predicted Up	0
Model Predicted Down	0
True–Pred No-Change Matches	1,129
🔍 Interpretation

The model currently collapses to predicting constant values, leading to:

High RMSE

Almost zero direction accuracy

No detection of up/down movements

Despite this, the end-to-end pipeline is correct, modular, and fully functional, making it ready for:

Hyperparameter tuning

Architectural enhancements

Loss-function experimentation (focal loss, return-scaled loss, etc.)
