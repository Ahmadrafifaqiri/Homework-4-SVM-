import pandas as pd
import numpy as np
    
def kernel_perceptron(X, y,gamma):
    num_samples = X.shape[0]
    indices= np.arange(num_samples)
    #alpha = [0, 1, 2,...]
    alpha = np.array([x for x in range(num_samples)])
    alpha = np.reshape(alpha, (-1, 1))
    y = np.reshape(y, (-1, 1))
    #K(X, Z, gamma) = e^{ - ||x_i - x_j||^2/ gamma}
    kernel = gaussian_kernel(X,X, gamma)
    for t in range(100):
        np.random.shuffle(indices)
        for i in range(num_samples):
            #(alpha_i * y_i)
            alpha_y = np.multiply(alpha, y)
            #(alpha_i * y_i * K)
            alpha_y_kernel = np.matmul(kernel[indices[i], :], alpha_y)
            if alpha_y_kernel * y[indices[i]] <= 0:
                alpha[indices[i]] = alpha[indices[i]] + 1
    return alpha

def kernel_perceptron_evaluate(alpha, train_X, train_y, test_X, test_y, gamma):
    kernel = gaussian_kernel(train_X, test_X, gamma)
    alpha_y = np.reshape(np.multiply(alpha, np.reshape(train_y, (-1, 1))), (1, -1))
    prediction = np.matmul(alpha_y, kernel)
    prediction = np.reshape(prediction, (-1,1))
    prediction[prediction > 0] = 1
    prediction[prediction <=0] = -1
    error = np.sum(np.abs(prediction - np.reshape(test_y,(-1,1)))) / 2 / len(test_y)
    return error

def gaussian_kernel(x1, x2, gamma):
    a = np.tile(x1, (1, x2.shape[0]))
    a = np.reshape(a, (-1,x1.shape[1]))
    b = np.tile(x2, (x1.shape[0], 1))
    kernel = np.exp(np.sum(np.square(a - b),axis=1) / -gamma)
    kernel = np.reshape(kernel, (x1.shape[0], x2.shape[0]))
    return kernel

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

gamma_list = np.array([0.1, 0.5, 1, 5, 100])

print("Kernel Perceptron")
for gamma in gamma_list:
    print("Gamma: ", gamma)
    
    p = kernel_perceptron(df_train_X, df_train_y,gamma)
    train_error = kernel_perceptron_evaluate(p, df_train_X, df_train_y, df_train_X, df_train_y, gamma)
    test_error = kernel_perceptron_evaluate(p, df_train_X, df_train_y, df_test_X, df_test_y, gamma)

    print("Train error: ", train_error)
    print("Test error: ", test_error)