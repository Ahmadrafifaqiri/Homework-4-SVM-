import numpy as np
import pandas as pd
import scipy.optimize as opt

#alpha_i * y_i is the constraint
def constraint(alpha, y):
    return np.matmul(np.reshape(alpha,(1, -1)), np.reshape(y, (-1,1)))[0]

def gaussian_kernel(x1, x2, gamma):
    a = np.tile(x1, (1, x2.shape[0]))
    a = np.reshape(a, (-1,x1.shape[1]))
    b = np.tile(x2, (x1.shape[0], 1))
    kernel = np.exp(np.sum(np.square(a - b),axis=1) / -gamma)
    kernel = np.reshape(kernel, (x1.shape[0], x2.shape[0]))
    return kernel

def guassian_objective_function(alpha, k, y):
    objective_function = - np.sum(alpha)
    #(alpha_i * y_i)
    alpha_y = np.multiply(np.reshape(alpha,(-1,1)), np.reshape(y, (-1,1)))
    #f(x) = - Sum(alpha_i) + 1/2(Sum( alpha_i * y_i * (alpha_j * y_j) * K))
    objective_function = objective_function + 0.5 * np.sum(np.multiply(np.matmul(alpha_y, np.transpose(alpha_y)), k))
    return objective_function

def gaussian_svm(C, gamma, X, y):
    num_samples = X.shape[0]
    cons = ({'type': 'eq', 'fun': lambda alpha: constraint(alpha, y)})
    #K(X, Z, gamma) = e^{ - ||x_i - x_j||^2/ gamma}
    kernel = gaussian_kernel(X, X, gamma)
    alpha_optimized = opt.minimize(lambda alpha: guassian_objective_function(alpha, kernel, y), np.zeros(num_samples),  method='SLSQP', bounds=[(0, C)] * num_samples, constraints=cons, options={'disp': False})
    return alpha_optimized.x

def gaussian_svm_evaluate(gamma, alpha, train_X, train_y, test_X, test_y):
    kernel = gaussian_kernel(train_X, test_X, gamma)
    kernel = np.multiply(np.reshape(train_y, (-1,1)), kernel)
    prediction = np.sum(np.multiply(np.reshape(alpha, (-1,1)), kernel), axis=0)
    prediction = np.reshape(prediction, (-1,1))
    prediction[prediction > 0] = 1
    prediction[prediction <=0] = -1
    error = np.sum(np.abs(prediction - np.reshape(test_y,(-1,1)))) / 2 / len(test_y)
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
    gamma_list = np.array([0.1, 0.5, 1, 5, 100])

    print("Gaussian Dual form of SVM non-Linear")

    index =0
    for C in C_list:
        
        for gamma in gamma_list:
            optimized_alpha_gaussian_svm = gaussian_svm(C, gamma, df_train_X[:,[x for x in range(num_columns - 1)]], df_train_y)
            
            support_vectors = np.where(optimized_alpha_gaussian_svm > 0)[0]
            print("\nC: ", C)
            
            print("Support Vectors: ", len(support_vectors))

            train_error = gaussian_svm_evaluate(gamma, optimized_alpha_gaussian_svm, df_train_X[:,[x for x in range(num_columns - 1)]], df_train_y, df_train_X[:,[x for x in range(num_columns - 1)]], df_train_y)
            test_error = gaussian_svm_evaluate(gamma, optimized_alpha_gaussian_svm, df_train_X[:,[x for x in range(num_columns - 1)]], df_train_y, df_test_X[:,[x for x in range(num_columns - 1)]], df_test_y)

            print("Gamma: ", gamma)
            print("Train Error: ", train_error)
            print("Test Error: ", test_error)
            if(C == 500/873):
                if index > 0:
                    overlapped_support_vectors = len(np.intersect1d(support_vectors, old_support_vectors))
                    print("Overlapped support vectors between gamma: {g1} & {g2} = {val}".format(g1 = gamma_list[index], g2 = gamma_list[index-1], val =  overlapped_support_vectors))
                index = index + 1
                old_support_vectors = support_vectors