import numpy as np
import pandas as pd
import scipy.optimize as opt

#this is the objective function for dual SVM
def objective_function(alpha, X, y):
    obj_function = - np.sum(alpha)
    #(alpha_i * y_i * x_i)
    alpha_y_X = np.multiply(np.multiply(np.reshape(alpha,(-1,1)), np.reshape(y, (-1,1))), X)
    #f(x) =  - Sum(alpha_i) + 1/2(Sum(alpha_i * alpha_j * y_i * y_j* x_i * x_j))
    obj_function = obj_function + 0.5 * np.sum(np.matmul(alpha_y_X, np.transpose(alpha_y_X)))
    return obj_function

#alpha_i * y_i is the constraint
def constraint(alpha, y):
    return np.matmul(np.reshape(alpha,(1, -1)), np.reshape(y, (-1,1)))[0]

def dual_svm(C,X, y):
    num_samples = X.shape[0]
    cons = ({'type': 'eq', 'fun': lambda alpha: constraint(alpha, y)})
    #this is to minimize alpha as part of the dual optimization problem
    alpha_minimized = opt.minimize(lambda alpha: objective_function(alpha, X, y), np.zeros(num_samples),  method='SLSQP', bounds=[(0, C)] * num_samples, constraints=cons, options={'disp': False})
    #weight vector is calculated using the minimized alpha_i * y_i * X_i
    weight_vector = np.sum(np.multiply(np.multiply(np.reshape(alpha_minimized.x,(-1,1)), np.reshape(y, (-1,1))), X), axis=0)
    #the support vectors for the minimized alpha is in the range (0,C)
    support_vectors = np.where((alpha_minimized.x > 0) & (alpha_minimized.x < C))
    #bias is calculated using y_i - weight^{T} * X_i
    bias =  np.mean(y[support_vectors] - np.matmul(X[support_vectors,:], np.reshape(weight_vector, (-1,1))))
    weight_vector = weight_vector.tolist()
    weight_vector.append(bias)
    weight_vector = np.array(weight_vector)
    return weight_vector, bias

def dual_svm_evaluate(X, y, weights):
    weights = np.reshape(weights, (5,1))
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
    gamma_list = np.array([0.1, 0.5, 1, 5, 100])
    print("\nDual form of SVM Linear\n")
    for C in C_list:
        final_weights_dual_svm, bias_dual_svm = dual_svm(C, df_train_X[:,[x for x in range(num_columns - 1)]], df_train_y)
        train_error = dual_svm_evaluate(df_train_X, df_train_y, final_weights_dual_svm) 
        test_error = dual_svm_evaluate(df_test_X, df_test_y, final_weights_dual_svm)
        print("\nC: ", C)
        print("Weight Vector: ", final_weights_dual_svm)
        print("Bias: ", bias_dual_svm)
        print("Train Error: ", train_error)
        print("Test Error: ", test_error)