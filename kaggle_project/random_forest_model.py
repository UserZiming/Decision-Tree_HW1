import numpy as np
import pandas as pd
import random_forest

# import training and testing dataset
train_data_csv = "income_data/train_final.csv"
test_data_csv = "income_data/test_final.csv"
train_df = pd.read_csv(train_data_csv)
test_df = pd.read_csv(test_data_csv)

# import the features and features types manually
attributes = ['age', 'workclass', 'fnlwgt', 'education', 'education.num', 'marital.status', 'occupation',
              'relationship', 'race', 'sex', 'capital.gain', 'capital.loss', 'hours.per.week', 'native.country']
attribute_types = {'age': ['numeric'],
                   'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov',
                                 'Without-pay', 'Never-worked', '?'],
                   'fnlwgt': ['numeric'],
                   'education': ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm',
                                 'Assoc-voc', '9th',
                                 '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool',
                                 '?'],
                   'education.num': ['numeric'],
                   'marital.status': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed',
                                      'Married-spouse-absent', 'Married-AF-spouse'],
                   'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
                                  'Prof-specialty',
                                  'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
                                  'Transport-moving',
                                  'Priv-house-serv', 'Protective-serv', 'Armed-Forces', '?'],
                   'relationship': ['Wife', 'Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried'],
                   'race': ['White', 'other-race'],
                   'sex': ['Female', 'Male'],
                   'capital.gain': ['numeric'],
                   'capital.loss': ['numeric'],
                   'hours.per.week': ['numeric'],
                   'native.country': ['United-States', 'other-country']}

# change the name of the label and the value from 0 to -1
train_df = train_df.rename(columns={'income>50K': 'label'})
train_df.loc[train_df['label'] == 0, 'label'] = -1
train_df = train_df.drop(columns=['education', 'fnlwgt'])
test_df = test_df.drop(columns=['education', 'fnlwgt'])
train_df = pd.concat(
    [train_df[train_df['label'] == 1], train_df[train_df['label'] == -1].sample(len(train_df[train_df['label'] == 1]),
                                                                                replace=False, ignore_index=True)], axis=0)
train_df = train_df.sample(len(train_df), replace=False, ignore_index=True)

# change the race to white and other race
train_df.loc[train_df['race'] != 'White', 'race'] = 'other-race'
test_df.loc[test_df['race'] != 'White', 'race'] = 'other-race'

# change the native country to USA or other country
# change the relationship to married or unmarried
# change the occupation to exec-managerial or other
# change marital status to married civ spouse or other
train_df.loc[train_df['native.country'] == 'Japan', 'native.country'] = 'United-States'
train_df.loc[train_df['native.country'] == 'Canada', 'native.country'] = 'United-States'
train_df.loc[train_df['native.country'] == 'India', 'native.country'] = 'United-States'
train_df.loc[train_df['native.country'] == 'Iran', 'native.country'] = 'United-States'
train_df.loc[train_df['native.country'] == 'Germany', 'native.country'] = 'United-States'
train_df.loc[train_df['native.country'] == 'England', 'native.country'] = 'United-States'
train_df.loc[train_df['native.country'] != 'United-States', 'native.country'] = 'other-country'
test_df.loc[test_df['native.country'] == 'Japan', 'native.country'] = 'United-States'
test_df.loc[test_df['native.country'] == 'Canada', 'native.country'] = 'United-States'
test_df.loc[test_df['native.country'] == 'India', 'native.country'] = 'United-States'
test_df.loc[test_df['native.country'] == 'Iran', 'native.country'] = 'United-States'
test_df.loc[test_df['native.country'] == 'Germany', 'native.country'] = 'United-States'
test_df.loc[test_df['native.country'] == 'England', 'native.country'] = 'United-States'
test_df.loc[test_df['native.country'] != 'United-States', 'native.country'] = 'other-country'
train_df.loc[train_df['relationship'] == 'Own-child', 'relationship'] = 'Unmarried'
train_df.loc[train_df['relationship'] == 'Other-relative', 'relationship'] = 'Unmarried'
train_df.loc[train_df['relationship'] == 'Not-in-family', 'relationship'] = 'Unmarried'
test_df.loc[test_df['relationship'] == 'Own-child', 'relationship'] = 'Unmarried'
test_df.loc[test_df['relationship'] == 'Other-relative', 'relationship'] = 'Unmarried'
test_df.loc[test_df['relationship'] == 'Not-in-family', 'relationship'] = 'Unmarried'
attribute_types['relationship'] = ['Wife', 'Husband', 'Unmarried']
train_df.loc[train_df['occupation'] == 'Prof-specialty', 'occupation'] = 'Exec-managerial'
train_df.loc[train_df['occupation'] != 'Exec-managerial', 'occupation'] = 'other'
test_df.loc[test_df['occupation'] == 'Prof-specialty', 'occupation'] = 'Exec-managerial'
test_df.loc[test_df['occupation'] != 'Exec-managerial', 'occupation'] = 'other'
attribute_types['occupation'] = ['Exec-managerial', 'other']
train_df.loc[train_df['marital.status'] != 'Married-civ-spouse', 'marital.status'] = 'other'
test_df.loc[test_df['marital.status'] != 'Married-civ-spouse', 'marital.status'] = 'other'
attribute_types['marital.status'] = ['Married-civ-spouse', 'other']

# classify the age, hours, education numbers, gain, loss
train_df['age'] = pd.cut(train_df['age'], bins=[0, 35, 70, 100], labels=['1', '2', '3'])
test_df['age'] = pd.cut(test_df['age'], bins=[0, 35, 70, 100], labels=['1', '2', '3'])
train_df['hours.per.week'] = pd.cut(train_df['hours.per.week'], bins=[0, 30, 50, 120], labels=['1', '2', '3'])
test_df['hours.per.week'] = pd.cut(test_df['hours.per.week'], bins=[0, 30, 50, 120], labels=['1', '2', '3'])
train_df['education.num'] = pd.cut(train_df['education.num'], bins=[0, 7, 14, 21], labels=['1', '2', '3'])
test_df['education.num'] = pd.cut(test_df['education.num'], bins=[0, 7, 14, 21], labels=['1', '2', '3'])
train_df['capital.gain'] = pd.cut(train_df['capital.gain'], bins=[-1, 100, 1000000], labels=['1', '2'])
test_df['capital.gain'] = pd.cut(test_df['capital.gain'], bins=[-1, 100, 1000000], labels=['1', '2'])
train_df['capital.loss'] = pd.cut(train_df['capital.loss'], bins=[-1, 100, 1000000], labels=['1', '2'])
test_df['capital.loss'] = pd.cut(test_df['capital.loss'], bins=[-1, 100, 1000000], labels=['1', '2'])
attribute_types['age'] = ['1', '2', '3']
attribute_types['hours.per.week'] = ['1', '2', '3']
attribute_types['education.num'] = ['1', '2', '3']
attribute_types['capital.gain'] = ['1', '2']
attribute_types['capital.loss'] = ['1', '2']

# train the random forest model
prediction_train, prediction = random_forest.run(train_df, test_df.drop(columns=['ID']), attribute_types, 15, 4)

for i in range(len(prediction)):
    if prediction[i] > 0:
        prediction[i] = 1
    else:
        prediction[i] = 0
predictions = pd.DataFrame({'ID': test_df['ID'], "Prediction": prediction})
predictions.to_csv("prediction_random_tree.csv", index=False)
