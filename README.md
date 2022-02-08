# Supervised Machine Learning - Predicting Credit Risk

In this project, a machine learning model  has been built ,that attempts to predict whether a loan from LendingClub will become high risk or not. 

## Background

LendingClub is a peer-to-peer lending services company that allows individual investors to partially fund personal loans as well as buy and sell notes backing the loans on a secondary market. LendingClub offers their previous data through an API.

This data is used to create machine learning models to classify the risk level of given loans. Specifically,to compare the Logistic Regression model and Random Forest Classifier.


### Retrieve the data

In the `Generator` folder in `Resources`, there is a [GenerateData.ipynb](/Resources/Generator/GenerateData.ipynb) notebook that will download data from LendingClub and output two CSVs: 

* `2019loans.csv`
* `2020Q1loans.csv`

An entire year's worth of data (2019) is used to predict the credit risk of loans from the first quarter of the next year (2020).

These two CSVs have been undersampled to give an even number of high risk and low risk loans. In the original dataset, only 2.2% of loans are categorized as high risk. To get a truly accurate model, special techniques need to be used on imbalanced data. Undersampling is one of those techniques. Oversampling and SMOTE (Synthetic Minority Over-sampling Technique) are other techniques that are also used.

## Preprocessing: Convert categorical data to numeric

Create a training set from the 2019 loans using `pd.get_dummies()` to convert the categorical data to numeric columns. Similarly, create a testing set from the 2020 loans, also using `pd.get_dummies()`. Note! There are categories in the 2019 loans that do not exist in the testing set. If you fit a model to the training set and try to score it on the testing set as is, you will get an error. You need to use code to fill in the missing categories in the testing set. 

## Consider the models

Two models has been used in creating and comparing  on this data: a logistic regression, and a random forests classifier. Before doing a  create, fit, and score the models,  a prediction has been made on which model will perform better. 

## Fit a LogisticRegression model and RandomForestClassifier model

Created a LogisticRegression model, fit it to the data, and print the model's score. Did the same for a RandomForestClassifier. 

## LogisticRegression model score on unscaled data
Training Data Score: 0.6498357963875205
Testing Data Score: 0.5163760102084219
## RandomForestClassifier model on unscaled data
Training Score: 0.9998357963875205
Testing Score: 0.6225010633772863

## Revisit the Preprocessing: Scale the data

The data going into these models was never scaled, an important step in preprocessing. Used `StandardScaler` to scale the training and testing sets.

Fit and score the LogisticRegression and RandomForestClassifier models on the scaled data.
## LogisticRegression model score on scaled data
Training Data Score: 0.7078817733990148
Testing Data Score: 0.6637601020842194

## RandomForestClassifier model on scaled data
Training Score: 0.9998357963875205
Testing Score: 0.5555082943428329

## Comparison between Prediction and actual results:
After scaling ,Scores for Logistic Regression model improved , but for Random Forest Classifier model ,there is not much difference.

## Conclusion:
Random Forest Classifier model performed better for this dataset as it has categorical data. 
Logistic regression is a little confusing when comes to categorical data.

### References

LendingClub (2019-2020) _Loan Stats_. Retrieved from: [https://resources.lendingclub.com/](https://resources.lendingclub.com/)
