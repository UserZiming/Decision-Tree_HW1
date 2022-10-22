import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import BatchGradientDescent
import StochasicGradiantDecent
import analytical

# import dataset from bank folder
train_data_csv = "./concrete/train.csv"
test_data_csv = "./concrete/test.csv"

# import the attributes manually
attributes = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'label']
train_df = pd.read_csv(train_data_csv, names=attributes)
test_df = pd.read_csv(test_data_csv, names=attributes)

# analytical solution
agd = analytical.optimal_calculate(train_df, attributes)
print('optimal weights: ' + str(agd))

plt.show()
