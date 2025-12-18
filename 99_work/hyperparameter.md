# RF
- Criterion: squared_error
- Max_features: sqrt or 1/3 (Standard for regression, helps with the high dimensionality of 270-day inputs).
- Bootstrap: True

Grid Search
- N_estimators: R[100,300,500], I[100,300,500]
- Min_samples_leaf: R[2,5], I[5,10]
- Max_depth: R[None(Unlimited)], I[10,20]

# XGBoost
- Objective: reg:squarederror 
- Booster: gbtree
- Subsample: 0.8 (Using 80% of data per tree prevents overfitting).
- Colsample_bytree: 0.8 (Using 80% of features per tree speeds up training).
- Min_child_weight: 1 (Default).

Grid Search
- N_estimators: R[500,1000], I[100,300]
- Max_depth: R[5,7], I[3,4]
- Learning Rate: R[0.05,0.1], I[0.05,0.1]

# TCN
> Application of temporal convolutional network for flood forecasting Xu et al., 2021 Hydrology Research
- Optimizer: Adam
- Padding: Causal (Crucial: prevents future data leakage).
- Dilations: [1, 2, 4, 8, 16, 32, 64, 128] (Ensures coverage > 270 days).
- Stride: 1
- Learning Rate: RI 1e-3, A 1e-4
- Loss Function: MSE

Grid Search
- Num. Filters: R[32,64], I[16,32]
- Kernel Size: R[2,4], I[2,4]
- Dropout: R[0.2,0.4], I[0.2,0.4]


# LSTM
> based on Kratzert et al. (2018, 2019)
- Optimizer: Adam
- Loss: MSE
- Number of Layers:1
- Batch size: R 256, I Unknown/all
- Epochs: R 30 ~ 50, I 20 
- Input sequence length: 270
- Learning Rate: RI[1e-3], A[1e-4]

Grid Search
- Hidden size: R [128,256], I[16,32]
- Dropout Rate: R[0.2,0.4], I[0.2,0.4]

