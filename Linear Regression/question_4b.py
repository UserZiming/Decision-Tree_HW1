import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import StochasicGradiantDecent

# import dataset from bank folder
train_data_csv = "./concrete/train.csv"
test_data_csv = "./concrete/test.csv"

# import the attributes manually
attributes = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'label']
train_df = pd.read_csv(train_data_csv, names=attributes)
test_df = pd.read_csv(test_data_csv, names=attributes)

# run the stochastic gradiant decent
iterations = 200
r = 0.01
print("Learning Rate is ", r)
out = StochasicGradiantDecent.stochasic_gradiant_decent(train_df, r, iterations)

# get the cost function with test dataframe
StochasicGradiantDecent.cost_function(test_df, out[0])
print("Weights:", out[0])
