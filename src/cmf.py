import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.linear_model import LinearRegression, Ridge
import time
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, PorterStemmer, WordNetLemmatizer

def collective_alternating_optimization(F, B, R, S, P,k, 
                                           reg_lambda=0.1, max_steps=100,
                                           log_every=1):
    """
    Perform matrix factorization using alternating optimization. Training is done via patience,
    i.e. we stop training after we observe no improvement on the validation loss for a certain
    amount of training steps. We then return the best values for Q and P oberved during training.
    
    Parameters
    ----------
    F                 : sp.spmatrix, shape [N, N]
                        The friendship relationship matrix => U * U.trans

    B                 : sp.spmatrix, shape [M, D]
                        The bussiness-attributes matrix => V * W

    R                 : sp.spmatrix, shape [M, N]
                        The bussiness-ratings matrix => V * U.trans

    S                 : sp.spmatrix, shape [M, C]
                        The bussiness-words matrix => V * Z      

    P                 : sp.spmatrix, shape [N, C]
                        The user-words matrix => U * Z
  
    k                 : int
                        The latent factor dimension.
                      
    reg_lambda        : float
                        The regularization strength.
                      
    max_steps         : int, optional, default: 100
                        Maximum number of training steps. 
    
    log_every         : int, optional, default: 1
                        Log the training status every X iterations.

    Returns
    -------
    best_U            : np.array, shape [N, k]
                        Best value for U
                      
    best_V            : np.array, shape [M, k]
                        Best value for V

    best_W            : np.array, shape [k, D]
                        Best value for W 

    best_Z            : np.array, shape [k, C]
                        Best value for Z 
                        
    train_losses      : list of floats
                        Training loss for every evaluation iteration, can be used for plotting the training
                        loss over time.                     

    """
    N, _ = F.shape
    M, D = B.shape
    _, C = S.shape
    
    #F_csc = F.tocsc()
    F_csr = F.tocsr()
    #F_lil = F.tolil()
    B_csc = B.tocsc()
    B_csr = B.tocsr()
    #B_lil = B.tolil()
    R_csc = R.tocsc()
    R_csr = R.tocsr() #mask the non zero values
    #R_lil = R.tolil()
    S_csc = S.tocsc()
    S_csr = S.tocsr()
    P_csc = P.tocsc()
    P_csr = P.tocsr()

    # initialization
    U = np.random.randn(N, k)
    V = np.random.randn(M, k)
    W = np.random.randn(k, D)
    Z = np.random.randn(k, C)
    best_U, best_V, best_W, best_Z = U.copy(), V.copy(), W.copy(), Z.copy()
    step = 0
    train_losses = []
    reconstruction_errors = []
    best_loss = -1
    
    def compute_loss(target, A, B):
        loss = 0
        for i, row in enumerate(target.rows):
            for j, x in enumerate(row):
                loss += (A[i].dot(B[:,x]) - target.data[i][j]) ** 2
        return loss
    
    while True:
        if step % log_every == 0:
            # compute training loss
            train_loss = 0
            train_loss += compute_loss(F, U, U.T)
            train_loss += compute_loss(B, V, W)
            train_loss += compute_loss(R, V, U.T)
            train_loss += compute_loss(S, V, Z)
            train_loss += compute_loss(P, U, Z)
            reconstruction_errors.append(train_loss)
            train_loss += reg_lambda * (np.sum(U * U) + np.sum(V * V) + np.sum(W * W) + np.sum(Z * Z))
            train_losses.append(train_loss)

            # logging
            if step % log_every == 0:
                print('Iteration %i, training_loss: %f, reconstruction error: %f' % (step, train_loss, reconstruction_errors[-1]))
                
        if step >= max_steps:
            break
        
        # Optimize U
        print('Optimizing U', end='\r')
        for i in range(N):
            F_R_i = F_csr[i].indices
            R_R_i = R_csc[:,i].indices
            P_R_i = P_csr[i].indices
            F_length = len(F_R_i)
            R_length = len(R_R_i)
            P_lenght = len(P_R_i)
            if F_length == 0 and R_length == 0 and P_lenght:
                continue

            factor1 = U[F_R_i].T.dot(U[F_R_i]).diagonal().sum() * (1/F_length) if len(F_R_i) > 0 else 0
            factor2 = V[R_R_i].T.dot(V[R_R_i]).diagonal().sum() * (1/R_length) if len(R_R_i) > 0 else 0
            factor3 = Z[:,P_R_i].T.dot(Z[:,P_R_i]).diagonal().sum() * (1/P_lenght) if len(P_R_i) > 0 else 0

            factor = 1/(factor1 + factor2 + factor3 + reg_lambda)

            F_r_i = F_csr[i].data
            R_r_i = R_csc[:,i].data
            P_r_i = P_csr[i].data

            result = 0
            if F_length > 0:
                result += ((1/F_length) * F_r_i * U[F_R_i].T).sum(1)
            if R_length > 0:
                result += ((1/R_length) * R_r_i * V[R_R_i].T).sum(1)
            if P_lenght > 0:
                result += ((1/P_lenght) * P_r_i * Z[:,P_R_i]).sum(1)    
                
            U[i] = factor * result
            
        # Optimize V
        print('Optimizing V', end='\r')
        for i in range(M):
            B_R_i = B_csr[i].indices
            R_R_i = R_csr[i].indices
            S_R_i = S_csr[i].indices
            B_length = len(B_R_i)
            R_length = len(R_R_i)
            S_lenght = len(S_R_i)
            if B_length == 0 and R_length == 0 and S_lenght == 0:
                continue
            factor1 = U[R_R_i].T.dot(U[R_R_i]).diagonal().sum() * (1/R_length) if len(R_R_i) > 0 else 0
            factor2 = W[:,B_R_i].dot(W[:,B_R_i].T).diagonal().sum() * (1/B_length) if len(B_R_i) > 0 else 0
            factor3 = Z[:,S_R_i].T.dot(Z[:,S_R_i]).diagonal().sum() * (1/S_lenght) if len(S_R_i) > 0 else 0

            factor = 1/(factor1 + factor2 + factor3 + reg_lambda)

            B_r_i = B_csr[i].data
            R_r_i = R_csr[i].data
            S_r_i = S_csr[i].data

            result = 0
            if B_length > 0:
                result += ((1/B_length) * B_r_i * W[:,B_R_i]).sum(1)
            if R_length > 0:
                result += ((1/R_length) * R_r_i * U[R_R_i].T).sum(1)
            if S_lenght > 0:
                result += ((1/S_lenght) * S_r_i * Z[:, S_R_i]).sum(1)    

            V[i] = factor * result
        
        # Optimize Z
        print('Optimizing Z', end='\r')
        for i in range(C):
            S_R_i = S_csc[:,i].indices
            P_R_i = P_csc[:,i].indices
            S_length = len(S_R_i)
            P_length = len(P_R_i)
            if S_length == 0 and P_length == 0:
                continue
            factor1 = U[P_R_i].T.dot(U[P_R_i]).diagonal().sum() * (1/P_length) if len(P_R_i) > 0 else 0
            factor2 = V[S_R_i].T.dot(V[S_R_i]).diagonal().sum() * (1/S_length) if len(S_R_i) > 0 else 0

            factor = 1/(factor1 + factor2 + reg_lambda)

            S_r_i = S_csc[:,i].data
            P_r_i = P_csc[:,i].data

            result = 0
            if P_length > 0:
                result += ((1/P_length) * P_r_i * U[P_R_i].T).sum(1)
            if R_length > 0:
                result += ((1/S_length) * S_r_i * V[S_R_i].T).sum(1)

            Z[:,i] = factor * result

        # Optimize W
        print('Optimizing W', end='\r')
        for i in range(D):
            B_R_i = B_csc[:,i].indices
            B_length = len(B_R_i)
            if B_length == 0:
                continue
            factor1 = V[B_R_i].T.dot(V[B_R_i]).diagonal().sum() * (1/B_length)
            factor = 1/(factor1 + reg_lambda)
            B_r_i = B_csc[:,i].data
            W[:,i] = factor * ((1/B_length) * B_r_i * V[B_R_i].T).sum(1)
        
        step += 1
    return U, V, W, Z, train_losses

