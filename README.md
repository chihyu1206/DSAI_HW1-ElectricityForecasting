# NCKU DSAI Homework 1 - Electricity Forecasting
[usage] 
1. pip install -r requirements.txt (Python版本為3.6.4 安裝dependency)
2. python app.py --training "training_data.csv" --output "submission.csv" (traing_data.csv是輸入不能改 輸出檔名可改)
 
## Description
我選擇台電所提供自2020/01/01~2020/01/31的發電數據csv檔，使用XGBoost做迴歸，以預測2021/03/23~2021/03/29的備轉容量(MW)。
### Data Analysis
如hw1.ipynb所示，首先使用pandas.DataFrame.corr以得到各項feature與備轉電力相關的相關係數，因為column太多用heatmap太雜，所以後來就把輸出移掉。
### Feature Engineering
參考了[相關係數之介紹](http://amebse.nchu.edu.tw/new_page_517.htm"相關係數之介紹")，了解到**r值於0～0.25或0～ -0.25，兩者缺乏相關**，所以使用下列程式碼
```
    mask1 = corr_mat["備轉容量(MW)"] > 0.2
    mask2 = corr_mat["備轉容量(MW)"] < -0.2
    cap = corr_mat["備轉容量(MW)"]
    cols_name = list(cap[mask1 | mask2].index)
```
將關聯度太低的特徵排除。
### Training 
接著使用sklearn之train_test_split和xgboost的XGBRegressor搭配做擬合，並以xgb.score判斷model效果。
```
    x = X[:-57]
    y = Y[57:]
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, shuffle=False)
    xgb = XGBRegressor(learning_rate=0.3, n_estimators=150, max_depth=2, min_child_weight=7,
                   subsample=1, colsample_bytree=1, gamma=0.02, reg_lambda=48, eta=0.1)
    xgb = xgb.fit(x_train, y_train)
    print(xgb.score(x_val, y_val))
```
