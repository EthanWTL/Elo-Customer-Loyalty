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

## Stage II: create baseline models in Ranfom Forest.


## Stage III:

Step I:
Split datasets into normal datas and anomaly data

![1](https://user-images.githubusercontent.com/97998419/223620361-47d5a857-406b-4b32-bac8-132b9682fcd9.png)

![2](https://user-images.githubusercontent.com/97998419/223620440-e8b16f85-c2ee-433f-a4fe-efb185d330ad.png)

Step II: 
apply regression models on both datasets and concat the results

![3](https://user-images.githubusercontent.com/97998419/223620509-b918c8f0-e03f-4305-abad-f99dd9c59e00.png)
![4](https://user-images.githubusercontent.com/97998419/223620581-a6bc2903-cb81-48ad-b1b4-0b207353a981.png)
