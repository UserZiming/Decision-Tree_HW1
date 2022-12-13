import pandas as pd
import numpy as np
import neural_network_zero_weight


def train_test_error(widths, epochs, learning_rate):
    for i in widths:
        print('-------------The width is ' + str(i), '-------------')
        # Train the neural network
        network = neural_network_zero_weight.neural_network(i, d=4)
        network.run(train_df, learning_rate, 10, epochs)
        # verify the neural network by using train dataset
        prediction = network.test(train_df.drop(columns=['label']))
        # Change the prediction label from 0 to -1
        for i in range(len(prediction)):
            if prediction[i] > 0:
                prediction[i] = 1
            else:
                prediction[i] = -1
        error = sum(abs(prediction - train_df['label'].to_numpy())) / (2 * len(train_df))
        print('Train error for the current width is ' + str(error))

        # verify the neural network by using test dataset
        prediction = network.test(test_df.drop(columns=['label']))
        # Change the prediction label from 0 to -1
        for i in range(len(prediction)):
            if prediction[i] > 0:
                prediction[i] = 1
            else:
                prediction[i] = -1
        error = sum(abs(prediction - test_df['label'].to_numpy())) / (2 * len(test_df))
        print('Test error for the current width is ' + str(error))


# import train/test dataset from folder
train_data_csv = "bank-note/train.csv"
test_data_csv = "bank-note/test.csv"
# import the features manually
attributes = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
train_df = pd.read_csv(train_data_csv, names=attributes).astype(float)
test_df = pd.read_csv(test_data_csv, names=attributes).astype(float)
# convert the label value from 0 to -1
train_df.loc[train_df['label'] == 0, 'label'] = -1
test_df.loc[test_df['label'] == 0, 'label'] = -1

# adjustable parameters
epochs = 100
learning_rate = 0.1
widths = [5, 10, 25, 50, 100]
train_test_error(widths, epochs, learning_rate)
