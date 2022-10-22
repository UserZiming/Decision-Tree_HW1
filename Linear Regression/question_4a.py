import pandas as pd
import numpy as np
import BatchGradientDescent
import matplotlib.pyplot as plt

# import dataset from bank folder
train_data_csv = "./concrete/train.csv"
test_data_csv = "./concrete/test.csv"

# import the attributes manually
attributes = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'label']
train_df = pd.read_csv(train_data_csv, names=attributes)
test_df = pd.read_csv(test_data_csv, names=attributes)

# run the batch gradient descent
iterations = 10000
r = 0.01
print("Learning Rate is ", r)
out = BatchGradientDescent.batch_gradient_descent(train_df, r, iterations)

for i in range(10):
    print('Prediction:', BatchGradientDescent.test(train_df.iloc[i, :], out[0]))

# get the cost function with test dataframe
BatchGradientDescent.cost_function(test_df, out[0])
print("Weights:", out[0])

# output the figure
fig = plt.figure()
plt.xlabel('Number of iterations')
plt.ylabel('Cost Function')
plt.plot(out[1])
plt.title('The Batch Gradient Descent Cost Function')
plt.legend()
plt.show()
