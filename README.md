# Elo-Customer-Loyalty
Welcome,

This repo walk through an independent project that analysis transaction data in a Brazil Shopping mall.

You can download the [dataset](https://www.kaggle.com/competitions/elo-merchant-category-recommendation/data) and get an overview of the compettion on [Kaggle](https://www.kaggle.com/competitions/elo-merchant-category-recommendation) also.

## Stage I: Data Engineering
Here is an overview of the data that we received and how we perform data engineer.
* train.csv - the training set
* test.csv - the test set
* merchants.csv - additional information about all merchants / merchant_ids in the dataset.
* new_merchant_transactions.csv - two months' worth of data for each card_id containing ALL purchases that card_id made at merchant_ids that were not visited in the historical data.

We first identify numeric columns and category columns

```python
numeric_cols = ['purchase_amount', 'installments']

category_cols = ['authorized_flag', 'city_id', 'category_1',
       'category_3', 'merchant_category_id','month_lag','most_recent_sales_range',
                 'most_recent_purchases_range', 'category_4',
                 'purchase_month', 'purchase_hour_section', 'purchase_day']

id_cols = ['card_id', 'merchant_id']
```

Then cross between two different kind of columns to generate first round of data engineering.
```python
columns = transaction.columns.tolist()
idx = columns.index('card_id')
category_cols_index = [columns.index(col) for col in category_cols]
numeric_cols_index = [columns.index(col) for col in numeric_cols]

for i in range(transaction.shape[0]):
    va = transaction.loc[i].values
    card = va[idx]
    for cate_ind in category_cols_index:
        for num_ind in numeric_cols_index:
            col_name = '&'.join([str(columns[cate_ind]), str(columns[num_ind]), str(va[cate_ind])])
            features[card][col_name] = features[card].get(col_name, 0) + va[num_ind]
    num += 1
```
We incorprate NLP ideas for feature generating, like shown below
```python
# first to activation day
cardid_features['first_to_activation_day']  =  (cardid_features['first_day'] - cardid_features['activation_day']).dt.days
# activation to reference day 
cardid_features['activation_to_reference_day']  =  (cardid_features['reference_day'] - cardid_features['activation_day']).dt.days
# first to last day 
cardid_features['first_to_reference_day']  =  (cardid_features['reference_day'] - cardid_features['first_day']).dt.days
# reference day to now 
```

## Stage II: Baseline Training
* ```Model:``` Random Forest
* ```Max_depth:``` 10
* ```Features:``` 80
* ```min_samples_leaf:``` 31
* ```n_estimators:``` 81
* ```RMSE``` after Cross-validation: 3.686


## Stage III: Assemle Learning & Merge
![image](https://github.com/EthanWTL/Elo-Customer-Loyalty/assets/97998419/d03eeb47-f0fc-46e9-ac7b-55eb95c53721)

From the graph above we can see there's anomaly group presented in our dataset, further investigation proof that we need to build first layer of model to filter out this group.

we built four model template, we will use Catboost for binary classification between anomaly group and normal customer group, and rest of the three models for further regression.

```python
if model_type == 'lgb':
       trn_data = lgb.Dataset(X[trn_idx], y[trn_idx])
       val_data = lgb.Dataset(X[val_idx], y[val_idx])
       clf = lgb.train(params, trn_data, num_boost_round=20000, valid_sets=[trn_data, val_data], verbose_eval=100, early_stopping_rounds=300)
       oof[val_idx] = clf.predict(X[val_idx], num_iteration=clf.best_iteration)
       predictions += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits
        
if model_type == 'xgb':
       trn_data = xgb.DMatrix(X[trn_idx], y[trn_idx])
       val_data = xgb.DMatrix(X[val_idx], y[val_idx])
       watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
       clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=100, params=params)
       oof[val_idx] = clf.predict(xgb.DMatrix(X[val_idx]), ntree_limit=clf.best_ntree_limit)
       predictions += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits
        
if (model_type == 'cat') and (eval_type == 'regression'):
       clf = CatBoostRegressor(iterations=20000, eval_metric='RMSE', **params)
       clf.fit(X[trn_idx], y[trn_idx], eval_set=(X[val_idx], y[val_idx]),cat_features=[], use_best_model=True, verbose=100)
       oof[val_idx] = clf.predict(X[val_idx])
       predictions += clf.predict(X_test) / folds.n_splits
            
if (model_type == 'cat') and (eval_type == 'binary'):
       clf = CatBoostClassifier(iterations=20000, eval_metric='Logloss', **params)
       clf.fit(X[trn_idx], y[trn_idx],eval_set=(X[val_idx], y[val_idx]),cat_features=[], use_best_model=True, verbose=100)
       oof[val_idx] = clf.predict_proba(X[val_idx])[:,1]
       predictions += clf.predict_proba(X_test)[:,1] / folds.n_splits

print(predictions)
if eval_type == 'regression':
       scores.append(mean_squared_error(oof[val_idx], y[val_idx])**0.5)
if eval_type == 'binary':
       scores.append(log_loss(y[val_idx], oof[val_idx]))
```

Step II: 
apply regression models on both datasets and concat the results

![3](https://user-images.githubusercontent.com/97998419/223620509-b918c8f0-e03f-4305-abad-f99dd9c59e00.png)
![4](https://user-images.githubusercontent.com/97998419/223620581-a6bc2903-cb81-48ad-b1b4-0b207353a981.png)
