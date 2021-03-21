# You can write code above the if-main block.
if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                        default='training_data.csv',
                        help='input training data file name')
    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()
    
    import pandas as pd
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split
    # read the data
    df = pd.read_csv("./training_data.csv")

    # Use Correlation coefficient to choose the features for training
    # if  -0.2 < r < 0.2  it has poor relation with the operating reserve
    corr_mat = df.corr()
    mask1 = corr_mat["備轉容量(MW)"] > 0.2
    mask2 = corr_mat["備轉容量(MW)"] < -0.2
    cap = corr_mat["備轉容量(MW)"]
    cols_name = list(cap[mask1 | mask2].index)
    
    X = df[cols_name]
    Y = df["備轉容量(MW)"]
    # The last date of traing_data is 2021/01/31, the last date we must predict is 2021/03/29
    # the period between these two dates is 57 days
    x = X[:-57]
    y = Y[57:]
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, shuffle=False)
    xgb = XGBRegressor(learning_rate=0.3, n_estimators=150, max_depth=2, min_child_weight=7,
                   subsample=1, colsample_bytree=1, gamma=0.02, reg_lambda=48, eta=0.1)
    xgb = xgb.fit(x_train, y_train)
    
    # predict the result
    predictions = xgb.predict(X[-57:])
    last7days = predictions[-7:]
    
    # write the result
    converted_dict = vars(args)
    with open('./' + converted_dict["output"], "w") as fp:
        print("date,operating_reserve(MW)", file=fp)
        date_param = 20210323
        for i in range(len(last7days)):
            print(date_param, predictions[i], sep=',', file=fp)
            date_param += 1
    fp.close()
