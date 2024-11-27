import numpy as np
import pandas as pd

def primal_svm_lr1(a, learning_rate, C, X, y):
    num_samples, num_features = X.shape
    weight_vector = np.zeros(num_features)
    indices = np.arange(num_samples)
    for T in range(100):
        np.random.shuffle(indices)
        X = X[indices,:]
        y = y[indices]
        for sample in range(num_samples):
            condition = y[sample] * np.sum(np.multiply(weight_vector, X[sample,:]))
            temp_weights = np.copy(weight_vector)
            temp_weights[num_features-1] = 0
            if condition <= 1:
                temp_weights = temp_weights - C * num_samples * y[sample] * X[sample,:]
            learning_rate = learning_rate / (1 + learning_rate / a * T)
            weight_vector = weight_vector - learning_rate * temp_weights

    return weight_vector, learning_rate

def primal_svm_lr2(learning_rate, C, X, y):
    num_samples, num_features = X.shape
    weight_vector = np.zeros(num_features)
    indices = np.arange(num_samples)
    for T in range(100):
        np.random.shuffle(indices)
        X = X[indices,:]
        y = y[indices]
        for sample in range(num_samples):
            condition = y[sample] * np.sum(np.multiply(weight_vector, X[sample,:]))
            temp_weights = np.copy(weight_vector)
            temp_weights[num_features-1] = 0
            if condition <= 1:
                temp_weights = temp_weights - C * num_samples * y[sample] * X[sample,:]
            learning_rate = learning_rate / (1 + T)
            weight_vector = weight_vector - learning_rate * temp_weights

    return weight_vector, learning_rate

def primal_svm_evaluate(X, y, weights):
    prediction = np.matmul(X, weights)
    prediction[prediction>0] = 1
    prediction[prediction<=0] = -1
    error = np.sum(np.abs(prediction - np.reshape(y,(-1,1)))) / 2 / len(y)
    return error

if __name__ == "__main__":
    df_train = pd.read_csv("/Users/u1503285/CS-6350-ML/SVM/bank-note/train.csv", header=None)
    values = df_train.values
    num_columns = values.shape[1]
    df_train_X = np.copy(values)
    df_train_X[:,num_columns-1] = 1
    df_train_y = values[:,num_columns-1]
    df_train_y = 2 * df_train_y -1

    df_test = pd.read_csv("//Users/u1503285/CS-6350-ML/SVM/bank-note/test.csv", header=None)
    values = df_test.values
    num_columns = values.shape[1]  
    df_test_X = np.copy(values)
    df_test_X[:,num_columns-1] = 1
    df_test_y = values[:,num_columns-1]
    df_test_y = 2 * df_test_y -1
    
    C_list = np.array([100, 500, 700])
    C_list = C_list/ 873

    for C in C_list:
        print('\nC: ', C)
        final_weights_lr1, final_lr1 = primal_svm_lr1(0.1, 0.1, C, df_train_X, df_train_y)
        final_weights_lr1 = np.reshape(final_weights_lr1, (5,1))
        print("\nPrimal SVM Linear with Learning Rate 1\n")
        print("Weights: ", final_weights_lr1)
        print("Learning Rate: ", final_lr1)
        train_error = primal_svm_evaluate(df_train_X, df_train_y, final_weights_lr1)
        print("Trian Error: ", train_error)
        test_error = primal_svm_evaluate(df_test_X, df_test_y, final_weights_lr1)
        print("Test Error: ", test_error)

        final_weights_lr2, final_lr2 = primal_svm_lr2(0.1, C, df_train_X, df_train_y)
        final_weights_lr2 = np.reshape(final_weights_lr2, (5,1))
        print("\nPrimal SVM Linear with learning rate 2\n")
        print("Weights: ", final_weights_lr2)
        print("Learning Rate: ", final_lr2)
        train_error = primal_svm_evaluate(df_train_X, df_train_y, final_weights_lr2)
        print("Trian Error: ", train_error)
        test_error = primal_svm_evaluate(df_test_X, df_test_y, final_weights_lr2)
        print("Test Error: ", test_error)