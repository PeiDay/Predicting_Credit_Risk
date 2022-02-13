# Predicting_Credit_Risk

## Background

LendingClub is a peer-to-peer lending services company that allows individual investors to partially fund personal loans as well as buy and sell notes backing the loans on a secondary market. 

Created machine learning models to classify the risk level of given loans using Logistic Regression model and Random Forest Classifier.


## Limitations
CSVs have been undersampled to give an even number of high risk and low risk loans. In the original dataset, only 2.2% of loans are categorized as high risk.

## Preprocessing: Convert categorical data to numeric

Create a training set from the 2019 loans using `pd.get_dummies()` to convert the categorical data to numeric columns. Similarly, create a testing set from the 2020 loans, also using `pd.get_dummies()`. 

## Prediction with Unscaled Data: Logistic Regression vs Random Forest

**Logistic Regression** is a classification algorithm used to predict a discrete set of classes or categories. it uses the sigmoid function to return a probability value. Meanwhile, **Random Forest Classifier** is an ensemble learning method that constructs a set of decision trees from a randomly selected subset of training set and combines them together to return a prediction.
__I think the Random Forest Classifier will perform better than Logistic Regression because the dataset consists of categorical data, which tends to work best with Random Forest models, and the Logistic Regression performs best with linearly separable datasets.__

## Results with Unscaled Data

**The Random Forest Classifier performed better based on the score. However there is an overfitting problem on the training dataset, showing that complexity may need to be reduced for Random Forest.**


## Prediction with Scaled data

The score for Logistic Regression will improve due to scaling whereas the score for Random Forest will remain the same. Graphical-model classifiers like Random Forest are invariant to feature scaling.
__I think the Logistic Regression will perform better than the Random Forest.__

## Results with Scaled Data

**Overall, scaling greatly improved the score of the Logistic Regression model so that it outperformed the Random Forest model. This shows that sometimes a simple model with scaled data can be a better fit than one with more complexity.**