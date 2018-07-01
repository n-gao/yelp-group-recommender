import src.MF as MF
import scipy.sparse as sp
import numpy as np
from sklearn.model_selection import train_test_split

class Dataset():
    
    def __init__(self, data):
        self.data = data
        self.train_ind, self.val_ind, self.val_values, self.test_ind, self.test_values = self.prepare_data(data)
    
    def prepare_data(self, data):
        print("Original size:{}".format(data.shape))
        ind = np.transpose(np.nonzero(data))
        print("Nonzero entries:{}".format(len(ind)))
        train, test = train_test_split(ind, test_size=0.2, random_state=42)
        test, val = train_test_split(test, test_size=0.5, random_state=42)
        train_ind = tuple(np.transpose(train))
        test_ind = tuple(np.transpose(test))
        val_ind = tuple(np.transpose(val))

        #print(relations[r_test_ind])
        print("Train:{}, Val:{}, Test:{}".format(len(train_ind[0]), len(val_ind[0]), len(test_ind[0])))

        val_values = data[val_ind]
        val_values = np.transpose(val_values).toarray()

        test_values = data[test_ind]
        test_values = np.transpose(test_values).toarray()

        return train_ind, val_ind, val_values, test_ind, test_values
    
def trainSingle(data, k_list, max_steps=10, log_every=1, reg_lambda = 0.1):
    print("train")
    best_test = 0
    best_k = 0
    Q = 0
    P = 0
    best_val = []
    best_train = []
    test_losses = []
    for i, k in enumerate(k_list):
        print("Training for k = ", k)
        U, V, val_loss, train_loss, conv = MF.latent_factor_alternating_optimization(data.data, data.train_ind, k,
                                                                                     data.val_ind, data.val_values,
                                                                           reg_lambda = reg_lambda, max_steps=max_steps, init='random',
                                                                           log_every=log_every, patience=10, eval_every=1)

        test_loss = test(U, V, data.test_ind, data.test_values)
        test_losses.append(test_loss)
        print("Test loss= ", test_loss)
        if i == 0 or test_loss<best_test:
            best_test = test_loss
            Q = U
            P = V
            best_val = val_loss
            best_train = train_loss
            best_k = k

    return Q, P, best_k, test_losses, best_val, best_train

def trainThree(friends, business_attributes, r_data, k_list, max_steps=10, log_every=1, reg_lambda = 0.1, center_data=True):
    print("train")
    best_test = 0
    best_k = 0
    best_U = 0
    best_V = 0
    best_W = 0
    best_train = []
    test_losses = []
    rel_data = r_data.data
    rel_data[r_data.test_ind] = 0
    rel_data[r_data.val_ind] = 0
    if center_data:
        rel_data, means, _ = center(rel_data)
    for i, k in enumerate(k_list):
        print("Training for k = ", k)
        U, V, W, _, train_loss, conv = MF.three_latent_factor_alternating_optimization(friends, business_attributes, rel_data, k, reg_lambda=reg_lambda, max_steps=max_steps)

        test_loss = test(U, V, r_data.test_ind, r_data.test_values)
        test_losses.append(test_loss)
        print("Test loss= ", test_loss)
        if i == 0 or test_loss<best_test:
            best_test = test_loss
            best_train = train_loss
            best_U = U
            best_V = V
            best_W = W
            best_k = k

    if center_data:
        return best_U, best_V, best_W, best_k, test_losses, best_train, means
    else:
        return best_U, best_V, best_W, best_k, test_losses, best_train

def train_three_graph(F, B, r_data, A, k_list, max_steps=10, log_every=1, reg_lambda=0.01, gamma=0.0000001, center_data=True):
    print("train")
    best_test = 0
    best_k = 0
    best_U = 0
    best_V = 0
    best_W = 0
    best_train = []
    test_losses = []
    rel_data = r_data.data
    rel_data[r_data.test_ind] = 0
    rel_data[r_data.val_ind] = 0
    if center_data:
        print('Centering data...', end='\r')
        rel_data, means, _ = center(rel_data)
        print('Centered data.')
    for i, k in enumerate(k_list):
        print("Training for k = ", k)
        U, V, W, _, train_loss, conv = MF.three_latent_factor_graph_alternating_optimization(F, B, rel_data, A, k, reg_lambda=reg_lambda, max_steps=max_steps, gamma=gamma)

        test_loss = test(U, V, r_data.test_ind, r_data.test_values)
        test_losses.append(test_loss)
        print("Test loss= ", test_loss)
        if i == 0 or test_loss<best_test:
            best_test = test_loss
            best_train = train_loss
            best_U = U
            best_V = V
            best_W = W
            best_k = k

    if center_data:
        return best_U, best_V, best_W, best_k, test_losses, best_train, means
    else:
        return best_U, best_V, best_W, best_k, test_losses, best_train
    

def test(U, V, test_ind, test_values, means=None):
    test_loss = 0
    for ind in range(len(test_ind[0])):
        i, j = int(test_ind[0][ind]), int(test_ind[1][ind])
        #print(test_loss)
        prod = U[j].dot(V[i])
        
        if means is not None:
            prod += means[j]
        
        #print(test_arr[ind])
        #prod = prod - test_values[ind]
        test_loss += (prod - test_values[ind]) ** 2

    return test_loss

def RMSE(U, V, test_ind, test_values, means=None):
    loss = test(U, V, test_ind, test_values, means)
    rmse = np.sqrt(loss/len(test_values))
    return rmse

def center(A, axis=1, epsilon=1e-12):
    mat = A.copy()
    mat = mat.tocsc() if axis==1 else mat.tocsr()
    means = np.ndarray(mat.shape[axis])
    for i in range(mat.shape[axis]):
        data = mat[i].data if axis == 0 else mat[:,i].data
        if len(data) == 0:
            means[i] = 0
        else:
            means[i] = data.mean()
            
    lil = mat.tolil()
    nnz = lil.nonzero()
    for i in range(lil.nnz):
        j = nnz[axis][i]
        lil[nnz[0][i], nnz[1][i]] -= means[j] + epsilon
    return lil, means, nnz
