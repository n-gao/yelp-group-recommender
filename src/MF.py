import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.linear_model import LinearRegression, Ridge
import time
import torch
import src.preprocessing as pre

def initialize_Q_P(matrix, k, init='random'):
    """
    Initialize the matrices Q and P for a latent factor model.
    
    Parameters
    ----------
    matrix : sp.spmatrix, shape [N, D]
             The matrix to be factorized.
    k      : int
             The number of latent dimensions.
    init   : str in ['svd', 'random'], default: 'random'
             The initialization strategy. 'svd' means that we use SVD to initialize P and Q, 'random' means we initialize
             the entries in P and Q randomly in the interval [0, 1).

    Returns
    -------
    Q : np.array, shape [N, k]
        The initialized matrix Q of a latent factor model.

    P : np.array, shape [k, D]
        The initialized matrix P of a latent factor model.
    """

    N, D = matrix.shape
    
    if init == 'svd':
        Q, _, P = svds(matrix, k=k)
    elif init == 'random':
        Q = np.random.randn(N, k)
        P = np.random.randn(k, D)
    else:
        raise ValueError
        
    assert Q.shape == (matrix.shape[0], k)
    assert P.shape == (k, matrix.shape[1])
    return Q, P

def three_latent_factor_alternating_optimization(F, B, R, k, val_idx = None, val_values = None, reg_lambda=0.1, max_steps=100, init='random', log_every=1, patience=10, eval_every=1):
    N, _ = F.shape
    M, D = B.shape
    
    #F_csc = F.tocsc()
    F_csr = F.tocsr()
    #F_lil = F.tolil()
    B_csc = B.tocsc()
    B_csr = B.tocsr()
    #B_lil = B.tolil()
    R_csc = R.tocsc()
    R_csr = R.tocsr() #mask the non zero values
    #R_lil = R.tolil()
    
    reg = Ridge(alpha=reg_lambda, fit_intercept=False)
    
    # initialization
    U = np.random.randn(N, k)
    V = np.random.randn(M, k)
    W = np.random.randn(k, D)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    step = 0
    train_losses = []
    validation_losses = []
    reconstruction_errors = []
    converged_after = 0
    best_loss = -1
    _patience = patience
    
    while True:
        if step % log_every == 0 or  step % eval_every == 0:
            # compute training loss
            #friend_loss = sse(F, U, U.T, F.nonzero())
            review_loss = sse(R, V, U.T, R.nonzero())
            business_loss = sse(B, V, W, B.nonzero())
            #train_loss = friend_loss + review_loss + business_loss
            train_loss = review_loss + business_loss
            train_losses.append(train_loss)

            validation_loss = sse_array(val_values, V[val_idx[0]].dot(U[val_idx[1]].T).diagonal())
            validation_losses.append(validation_loss)

            # check if we improved
            if step % eval_every == 0 and val_idx is not None:
                if validation_loss < best_loss or best_loss == -1:
                    converged_after = step
                    best_loss = validation_loss
                    _patience = patience                    
                else:
                    _patience -= 1
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                # stop if there is no change
                if _patience <= 0:
                    print('Converged after %i iterations' % converged_after)
                    break
            
            # logging
            if step % log_every == 0:
                print('Iteration %i, training_loss: %f, review error: %f, validation loss: %f' % (step, train_loss, review_loss, validation_losses[-1]))

                
        if step >= max_steps:
            break
        
        # Optimize U
        print('Optimizing U', end='\r')
        for i in range(N):
            #F_R_i = F_csr[i].indices
            R_R_i = R_csc[:,i].indices
            #F_length = len(F_R_i)
            R_length = len(R_R_i)
            if R_length == 0: #and F_length == 0
                continue
            #F_r_i = F_csr[i].data
            R_r_i = R_csc[:,i].data
            
            #_X = np.concatenate([U[F_R_i], V[R_R_i]])
            _X = V[R_R_i]
            X = _X.T @ _X + reg_lambda * np.eye(k)
            #y = np.concatenate([F_r_i, R_r_i])
            y = R_r_i
            y = _X.T.dot(y)
            U[i] = np.linalg.solve(X, y)
            
        # Optimize V
        print('Optimizing V', end='\r')
        for i in range(M):
            B_R_i = B_csr[i].indices
            R_R_i = R_csr[i].indices
            B_length = len(B_R_i)
            R_length = len(R_R_i)
            if B_length == 0 and R_length == 0:
                continue
            B_r_i = B_csr[i].data
            R_r_i = R_csr[i].data
            
            _X = np.concatenate([U[R_R_i], W[:, B_R_i].T])
            X = _X.T @ _X + reg_lambda * np.eye(k)
            y = np.concatenate([R_r_i, B_r_i])
            y = _X.T.dot(y)
            V[i] = np.linalg.solve(X, y)
            
        # Optimize W
        print('Optimizing W', end='\r')
        for i in range(D):
            B_R_i = B_csc[:,i].indices
            B_length = len(B_R_i)
            if B_length == 0:
                continue
            B_r_i = B_csc[:,i].data
            X = V[B_R_i].T @ V[B_R_i] + reg_lambda * np.eye(k) # TODO: cache
            y = V[B_R_i].T.dot(B_r_i)
            W[:, i] = np.linalg.solve(X, y)
        
        step += 1
    return best_U, best_V, best_W, validation_losses, train_losses, converged_after


def three_latent_factor_connected_alternating_optimization(F, B, R, A, k, val_idx = None, val_values = None, reg_lambda=0.1, max_steps=100, init='random', log_every=1, patience=10, eval_every=1):
    N, _ = F.shape
    M, D = B.shape
    
    #F_csc = F.tocsc()
    F_csr = F.tocsr()
    #F_lil = F.tolil()
    B_csc = B.tocsc()
    B_csr = B.tocsr()
    #B_lil = B.tolil()
    R_csc = R.tocsc()
    R_csr = R.tocsr() #mask the non zero values
    A_csr = A.tocsr()
    #R_lil = R.tolil()
    
    reg = Ridge(alpha=reg_lambda, fit_intercept=False)
    
    # initialization
    U = np.random.randn(N, k)
    V = np.random.randn(M, k)
    W = np.random.randn(k, D)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    step = 0
    train_losses = []
    validation_losses = []
    reconstruction_errors = []
    converged_after = 0
    best_loss = -1
    _patience = patience
    
    while True:
        if step % log_every == 0 or  step % eval_every == 0:
            # compute training loss
            friend_loss = sse(F, U, U.T, F.nonzero())
            review_loss = sse(R, V, U.T, R.nonzero())
            business_loss = sse(B, V, W, B.nonzero())
            bus_conn_loss = sse(A, V, V.T, A.nonzero())
            train_loss = friend_loss + review_loss + business_loss + bus_conn_loss
            train_losses.append(train_loss)

            validation_loss = sse_array(val_values, V[val_idx[0]].dot(U[val_idx[1]].T).diagonal())
            validation_losses.append(validation_loss)

            # check if we improved
            if step % eval_every == 0 and val_idx is not None:
                if validation_loss < best_loss or best_loss == -1:
                    converged_after = step
                    best_loss = validation_loss
                    _patience = patience                    
                else:
                    _patience -= 1
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                # stop if there is no change
                if _patience <= 0:
                    print('Converged after %i iterations' % converged_after)
                    break
            
            # logging
            if step % log_every == 0:
                print('Iteration %i, training_loss: %f, review error: %f, validation loss: %f' % (step, train_loss, review_loss, validation_losses[-1]))

                
        if step >= max_steps:
            break
        
        # Optimize U
        print('Optimizing U', end='\r')
        for i in range(N):
            F_R_i = F_csr[i].indices
            R_R_i = R_csc[:,i].indices
            F_length = len(F_R_i)
            R_length = len(R_R_i)
            if F_length == 0 and R_length == 0:
                continue
            F_r_i = F_csr[i].data
            R_r_i = R_csc[:,i].data
            
            _X = np.concatenate([U[F_R_i], V[R_R_i]])
            X = _X.T @ _X + reg_lambda * np.eye(k)
            y = np.concatenate([F_r_i, R_r_i])
            y = _X.T.dot(y)
            U[i] = np.linalg.solve(X, y)
            
        # Optimize V
        print('Optimizing V', end='\r')
        for i in range(M):
            B_R_i = B_csr[i].indices
            R_R_i = R_csr[i].indices
            A_R_i = A_csr[i].indices
            B_length = len(B_R_i)
            R_length = len(R_R_i)
            A_length = len(A_R_i)
            if B_length == 0 and R_length == 0:
                continue
            B_r_i = B_csr[i].data
            R_r_i = R_csr[i].data
            A_r_i = A_csr[i].data
            
            _X = np.concatenate([U[R_R_i], W[:, B_R_i].T, V[A_R_i]])
            X = _X.T @ _X + reg_lambda * np.eye(k)
            y = np.concatenate([R_r_i, B_r_i, A_r_i])
            y = _X.T.dot(y)
            V[i] = np.linalg.solve(X, y)
            
        # Optimize W
        print('Optimizing W', end='\r')
        for i in range(D):
            B_R_i = B_csc[:,i].indices
            B_length = len(B_R_i)
            if B_length == 0:
                continue
            B_r_i = B_csc[:,i].data
            X = V[B_R_i].T @ V[B_R_i] + reg_lambda * np.eye(k) # TODO: cache
            y = V[B_R_i].T.dot(B_r_i)
            W[:, i] = np.linalg.solve(X, y)
        
        step += 1
    return best_U, best_V, best_W, validation_losses, train_losses, converged_after



def three_latent_factor_graph_alternating_optimization(F, B, R, A, k, val_idx = None, val_values = None, reg_lambda=0.1, gamma=0.01, max_steps=100, init='random', log_every=1, patience=10, eval_every=1):
    N, _ = F.shape
    M, D = B.shape
    
    #F_csc = F.tocsc()
    F_csr = F.tocsr()
    #F_lil = F.tolil()
    B_csc = B.tocsc()
    B_csr = B.tocsr()
    #B_lil = B.tolil()
    R_csc = R.tocsc()
    R_csr = R.tocsr() #mask the non zero values
    A_csr = A.tocsr()
    #R_lil = R.tolil()
    
    reg = Ridge(alpha=reg_lambda, fit_intercept=False)
    
    # initialization
    U = np.random.randn(N, k)
    V = np.random.randn(M, k)
    W = np.random.randn(k, D)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    step = 0
    train_losses = []
    validation_losses = []
    reconstruction_errors = []
    converged_after = 0
    best_loss = -1
    _patience = patience
    
    while True:
        if step % log_every == 0 or  step % eval_every == 0:
            # compute training loss
            friend_loss = sse(F, U, U.T, F.nonzero())
            review_loss = sse(R, V, U.T, R.nonzero())
            business_loss = sse(B, V, W, B.nonzero())
            bus_conn_loss = sse(A, V, V.T, A.nonzero())
            train_loss = friend_loss + review_loss + business_loss + bus_conn_loss
            train_losses.append(train_loss)

            validation_loss = sse_array(val_values, V[val_idx[0]].dot(U[val_idx[1]].T).diagonal())
            validation_losses.append(validation_loss)

            # check if we improved
            if step % eval_every == 0 and val_idx is not None:
                if validation_loss < best_loss or best_loss == -1:
                    converged_after = step
                    best_loss = validation_loss
                    _patience = patience                    
                else:
                    _patience -= 1
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                # stop if there is no change
                if _patience <= 0:
                    print('Converged after %i iterations' % converged_after)
                    break
            
            # logging
            if step % log_every == 0:
                print('Iteration %i, training_loss: %f, review error: %f, validation loss: %f' % (step, train_loss, review_loss, validation_losses[-1]))

                
        if step >= max_steps:
            break
        
        # Optimize U
        print('Optimizing U', end='\r')
        for i in range(N):
            F_R_i = F_csr[i].indices
            R_R_i = R_csc[:,i].indices
            F_length = len(F_R_i)
            R_length = len(R_R_i)
            if F_length == 0 and R_length == 0:
                continue
            R_r_i = R_csc[:,i].data
            
            _X = V[R_R_i]
            y = R_r_i
            if F_length > 0:
                _X = np.concatenate([_X, np.eye(k) * gamma])
                y = np.concatenate([y, U[F_R_i].mean(0)])
            X = _X.T @ _X + reg_lambda * np.eye(k)
            y = _X.T.dot(y)
            U[i] = np.linalg.solve(X, y)
            
        # Optimize V
        print('Optimizing V', end='\r')
        for i in range(M):
            B_R_i = B_csr[i].indices
            R_R_i = R_csr[i].indices
            A_R_i = A_csr[i].indices
            B_length = len(B_R_i)
            R_length = len(R_R_i)
            A_length = len(A_R_i)
            if B_length == 0 and R_length == 0:
                continue
            B_r_i = B_csr[i].data
            R_r_i = R_csr[i].data
            
            _X = np.concatenate([U[R_R_i], W[:, B_R_i].T])
            y = np.concatenate([R_r_i, B_r_i])
            if A_length > 0:
                _X = np.concatenate([_X, np.eye(k) * gamma])
                y = np.concatenate([y, V[A_R_i].mean(0)])
            X = _X.T @ _X + reg_lambda * np.eye(k)
            y = _X.T.dot(y)
            V[i] = np.linalg.solve(X, y)
            
        # Optimize W
        print('Optimizing W', end='\r')
        for i in range(D):
            B_R_i = B_csc[:,i].indices
            B_length = len(B_R_i)
            if B_length == 0:
                continue
            B_r_i = B_csc[:,i].data
            X = V[B_R_i].T @ V[B_R_i] + reg_lambda * np.eye(k) # TODO: cache
            y = V[B_R_i].T.dot(B_r_i)
            W[:, i] = np.linalg.solve(X, y)
        
        step += 1
    return best_U, best_V, best_W, validation_losses, train_losses, converged_after



def three_latent_factor_connected_graph_alternating_optimization(F, B, R, A, k, val_idx = None, val_values = None, reg_lambda=0.1, gamma=0.01, max_steps=100, learning_rate=1, learning_iters=1, learning_rate_decay=0.1, init='random', log_every=1, patience=10, eval_every=1):
    N, _ = F.shape
    M, D = B.shape
    
    #F_csc = F.tocsc()
    F_csr = F.tocsr()
    #F_lil = F.tolil()
    B_csc = B.tocsc()
    B_csr = B.tocsr()
    #B_lil = B.tolil()
    R_csc = R.tocsc()
    R_csr = R.tocsr() #mask the non zero values
    A_csr = A.tocsr()
    #R_lil = R.tolil()
    
    reg = Ridge(alpha=reg_lambda, fit_intercept=False)
    
    # initialization
    if type(init) is tuple:
        U, V, W = init
        print('Used provided parameters')
    else:
        U = np.random.randn(N, k)
        V = np.random.randn(M, k)
        W = np.random.randn(k, D)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    step = 0
    train_losses = []
    validation_losses = []
    reconstruction_errors = []
    converged_after = 0
    best_loss = -1
    _patience = patience
    
    while True:
        if step % log_every == 0 or  step % eval_every == 0:
            # compute training loss
            friend_loss = sse(F, U, U.T, F.nonzero())
            review_loss = sse(R, V, U.T, R.nonzero())
            business_loss = sse(B, V, W, B.nonzero())
            bus_conn_loss = sse(A, V, V.T, A.nonzero())
            train_loss = friend_loss + review_loss + business_loss + bus_conn_loss
            train_losses.append(train_loss)

            validation_loss = val_sse(V, U.T, val_values, val_idx)
            validation_losses.append(validation_loss)

            # check if we improved
            if step % eval_every == 0 and val_idx is not None:
                if validation_loss < best_loss or best_loss == -1:
                    converged_after = step
                    best_loss = validation_loss
                    _patience = patience
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                else:
                    _patience -= 1
                # stop if there is no change
                if _patience <= 0:
                    print('Converged after %i iterations' % converged_after)
                    break
            
            # logging
            if step % log_every == 0:
                print('Iteration %i, training_loss: %f, review error: %f, validation loss: %f' % (step, train_loss, review_loss, validation_losses[-1]))

                
        if step >= max_steps:
            break
        
        # Optimize U
        print('Optimizing U', end='\r')
        for i in range(N):
            F_R_i = F_csr[i].indices
            R_R_i = R_csc[:,i].indices
            F_length = len(F_R_i)
            R_length = len(R_R_i)
            if F_length == 0 and R_length == 0:
                continue
            R_r_i = R_csc[:,i].data
            
            _X = np.concatenate([V[R_R_i], U[F_R_i]])
            y = np.concatenate([R_r_i, np.ones(F_length)])
            if F_length > 0:
                _X = np.concatenate([_X, np.eye(k) * gamma])
                y = np.concatenate([y, U[F_R_i].mean(0)])
            X = _X.T @ _X + reg_lambda * np.eye(k)
            y = _X.T.dot(y)
            U[i] = np.linalg.solve(X, y)
            #U[i] += (np.linalg.solve(X, y) - U[i]) * learning_rate
            
        # Optimize V
        print('Optimizing V', end='\r')
        for i in range(M):
            B_R_i = B_csr[i].indices
            R_R_i = R_csr[i].indices
            A_R_i = A_csr[i].indices
            B_length = len(B_R_i)
            R_length = len(R_R_i)
            A_length = len(A_R_i)
            if B_length == 0 and R_length == 0:
                continue
            B_r_i = B_csr[i].data
            R_r_i = R_csr[i].data
            
            _X = np.concatenate([U[R_R_i], W[:, B_R_i].T, V[A_R_i]])
            y = np.concatenate([R_r_i, B_r_i, np.ones(A_length)])
            if A_length > 0:
                _X = np.concatenate([_X, np.eye(k) * gamma])
                y = np.concatenate([y, V[A_R_i].mean(0)])
            X = _X.T @ _X + reg_lambda * np.eye(k)
            y = _X.T.dot(y)
            V[i] = np.linalg.solve(X, y)
            #V[i] += (np.linalg.solve(X, y) - V[i]) * learning_rate
            
        # Optimize W
        print('Optimizing W', end='\r')
        for i in range(D):
            B_R_i = B_csc[:,i].indices
            B_length = len(B_R_i)
            if B_length == 0:
                continue
            B_r_i = B_csc[:,i].data
            X = V[B_R_i].T @ V[B_R_i] + reg_lambda * np.eye(k) # TODO: cache
            y = V[B_R_i].T.dot(B_r_i)
            W[:, i] = np.linalg.solve(X, y)
            #W[:, i] += (np.linalg.solve(X, y) - W[:, i]) * learning_rate
        
        step += 1
        
    learning_iters -= 1
    if learning_iters > 0:
        return three_latent_factor_connected_graph_alternating_optimization(F, B, R, A, k, val_idx = val_idx, val_values = val_values, reg_lambda=reg_lambda, gamma=gamma, max_steps=max_steps, learning_rate=learning_rate*learning_rate_decay, learning_iters=learning_iters, init=(best_U, best_V, best_W), log_every=log_every, patience=patience, eval_every=eval_every)
    
    return best_U, best_V, best_W, validation_losses, train_losses, converged_after



def four_latent_factor_connected_graph_alternating_optimization(F, B, R, A, UW, BW, k, val_idx = None, val_values = None, reg_lambda=1, gamma=1, max_steps=100, init='random', log_every=1, patience=10, eval_every=1):
    N, _ = F.shape
    M, D = B.shape
    _, C = UW.shape
    
    #F_csc = F.tocsc()
    F_csr = F.tocsr()
    #F_lil = F.tolil()
    B_csc = B.tocsc()
    B_csr = B.tocsr()
    #B_lil = B.tolil()
    R_csc = R.tocsc()
    R_csr = R.tocsr() #mask the non zero values
    A_csr = A.tocsr()
    
    UW_csc = UW.tocsc()
    UW_csr = UW.tocsr()
    BW_csc = BW.tocsc()
    BW_csr = BW.tocsr()
    #R_lil = R.tolil()
    
    reg = Ridge(alpha=reg_lambda, fit_intercept=False)
    
    # initialization
    U = np.random.randn(N, k)
    V = np.random.randn(M, k)
    W = np.random.randn(k, D)
    Z = np.random.randn(k, C)
    best_U, best_V, best_W, best_Z = U.copy(), V.copy(), W.copy(), Z.copy()
    step = 0
    train_losses = []
    validation_losses = []
    reconstruction_errors = []
    converged_after = 0
    best_loss = -1
    _patience = patience
    
    while True:
        if step % log_every == 0 or  step % eval_every == 0:
            # compute training loss
            friend_loss = sse(F, U, U.T, F.nonzero())
            review_loss = sse(R, V, U.T, R.nonzero())
            business_loss = sse(B, V, W, B.nonzero())
            bus_conn_loss = sse(A, V, V.T, A.nonzero())
            uw_loss = sse(UW, U, Z, UW.nonzero())
            bw_loss = sse(BW, V, Z, BW.nonzero())
            train_loss = friend_loss + review_loss + business_loss + bus_conn_loss + uw_loss + bw_loss
            train_losses.append(train_loss)
            #print(friend_loss, review_loss, business_loss, bus_conn_loss, uw_loss, bw_loss)
            
            validation_loss = val_sse(V, U.T, val_values, val_idx)
            validation_losses.append(validation_loss)

            # check if we improved
            if step % eval_every == 0 and val_idx is not None:
                if validation_loss < best_loss or best_loss == -1:
                    converged_after = step
                    best_loss = validation_loss
                    _patience = patience                    
                else:
                    _patience -= 1
                    best_U, best_V, best_W, best_Z = U.copy(), V.copy(), W.copy(), Z.copy()
                # stop if there is no change
                if _patience <= 0:
                    print('Converged after %i iterations' % converged_after)
                    break
            
            # logging
            if step % log_every == 0:
                print('Iteration %i, training_loss: %f, review error: %f, validation loss: %f' % (step, train_loss, review_loss, validation_losses[-1]))

                
        if step >= max_steps:
            break
        
        # Optimize U
        print('Optimizing U', end='\r')
        for i in range(N):
            F_R_i = F_csr[i].indices
            R_R_i = R_csc[:,i].indices
            W_R_i = UW_csr[i].indices
            F_length = len(F_R_i)
            R_length = len(R_R_i)
            W_length = len(W_R_i)
            if F_length == 0 and R_length == 0 and W_length == 0:
                continue
            R_r_i = R_csc[:,i].data
            
            _X = np.concatenate([V[R_R_i], U[F_R_i]])
            y = np.concatenate([R_r_i, np.ones(F_length)])
            if F_length > 0:
                _X = np.concatenate([_X, np.eye(k) * gamma])
                y = np.concatenate([y, U[F_R_i].mean(0)])
            if W_length > 0:
                _X = np.concatenate([_X, Z.T[W_R_i]])
                y = np.concatenate([y, UW_csr[i].data])
            X = _X.T @ _X + reg_lambda * np.eye(k)
            y = _X.T.dot(y)
            U[i] = np.linalg.solve(X, y)
            
        # Optimize V
        print('Optimizing V', end='\r')
        for i in range(M):
            B_R_i = B_csr[i].indices
            R_R_i = R_csr[i].indices
            A_R_i = A_csr[i].indices
            W_R_i = BW_csr[i].indices
            B_length = len(B_R_i)
            R_length = len(R_R_i)
            A_length = len(A_R_i)
            W_length = len(W_R_i)
            if F_length == 0 and R_length == 0 and W_length == 0:
                continue
            B_r_i = B_csr[i].data
            R_r_i = R_csr[i].data
            
            _X = np.concatenate([U[R_R_i], W[:, B_R_i].T, V[A_R_i]])
            y = np.concatenate([R_r_i, B_r_i, np.ones(A_length)])
            if A_length > 0:
                _X = np.concatenate([_X, np.eye(k) * gamma])
                y = np.concatenate([y, V[A_R_i].mean(0)])
            if W_length > 0:
                _X = np.concatenate([_X, Z.T[W_R_i]])
                y = np.concatenate([y, BW_csr[i].data])
            X = _X.T @ _X + reg_lambda * np.eye(k)
            y = _X.T.dot(y)
            V[i] = np.linalg.solve(X, y)
            
        # Optimize W
        print('Optimizing W', end='\r')
        for i in range(D):
            B_R_i = B_csc[:,i].indices
            B_length = len(B_R_i)
            if B_length == 0:
                continue
            B_r_i = B_csc[:,i].data
            X = V[B_R_i].T @ V[B_R_i] + reg_lambda * np.eye(k) # TODO: cache
            y = V[B_R_i].T.dot(B_r_i)
            W[:, i] = np.linalg.solve(X, y)
        
        # Optimize Z
        print('Optimizing Z', end='\r')
        for i in range(C):
            U_R_i = UW_csc[:,i].indices
            B_R_i = BW_csc[:,i].indices
            U_length = len(U_R_i)
            B_length = len(B_R_i)
            if U_length == 0 and B_length == 0:
                continue
            _X = np.concatenate([U[U_R_i], V[B_R_i]])
            X = _X.T @ _X + reg_lambda * np.eye(k)
            y = np.concatenate([UW_csc[:, i].data, BW_csc[:, i].data])
            y = _X.T.dot(y)
            Z[:,i] = np.linalg.solve(X, y)
        
        step += 1
    return best_U, best_V, best_W, best_Z, validation_losses, train_losses, converged_after


def latent_factor_alternating_optimization_pytorch(M, k, val_idx, val_values,
                                           reg_lambda, max_steps=100, init='random',
                                           log_every=1, patience=10, eval_every=1):
    
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    N, D = M.shape
    # get the data in different formats for faster access
    csc = M.tocsc()
    csr = M.tocsr()
    
    M_vals = torch.tensor(M[M.nonzero()].todense(), dtype=torch.float32)
    M_inds = torch.tensor(M.nonzero(), dtype=torch.int64)
    
    val_values = torch.tensor(val_values, dtype=torch.float32)
    val_idx = torch.tensor(val_idx, dtype=torch.int64)
    
    d_indices = []
    d_data = []
    d_nnz = []
    for i in range(D):
        inds = csc[:,i].indices
        if len(inds) != 0:
            d_indices.append(torch.tensor(inds, dtype=torch.int64))
            d_data.append(torch.tensor(csc[:,i].data, dtype=torch.float32))
            d_nnz.append(i)
    d_nnz_l = len(d_nnz)
    d_nnz = torch.tensor(d_nnz, dtype=torch.int64)
    
    n_indices = []
    n_data = []
    n_nnz = []
    for i in range(N):
        inds = csr[i].indices
        if len(inds) != 0:
            n_indices.append(torch.tensor(inds, dtype=torch.int64))
            n_data.append(torch.tensor(csc[i].data, dtype=torch.float32))
            n_nnz.append(i)
    n_nnz_l = len(n_nnz)
    n_nnz = torch.tensor(n_nnz, dtype=torch.int64)
    print('Precomutation done.')
    #torch_sparse = torch.sparse.FloatTensor(torch.tensor(M.nonzero(), dtype=torch.int64).cpu(), torch.tensor(M[M.nonzero()].todense(), dtype=torch.float32).reshape(-1).cpu(), M.shape)
    #print(torch_sparse)
    #print(torch_sparse[0])
        
    # initialization
    #Q, P = initialize_Q_P(M, k, init=init)
    Q, P = torch.randn(N, k), torch.randn(k, D)
    #Q, P = torch.from_numpy(Q).cuda(), torch.from_numpy(P).cuda()
    best_Q, best_P = torch.tensor(Q), torch.tensor(P)
    step = 0
    train_losses = []
    validation_losses = []
    converged_after = 0
    best_loss = -1
    d_Xs = torch.zeros(d_nnz_l, k, k)
    d_ys = torch.zeros(d_nnz_l, k)
    n_Xs = torch.zeros(n_nnz_l, k, k)
    n_ys = torch.zeros(n_nnz_l, k)
    while step < max_steps:
        
        if step % log_every == 0 or  step % eval_every == 0:
            # compute training loss
            train_loss = val_loss_torch(M_vals, Q, P, reg_lambda, M.inds)
            train_losses.append(train_loss)

            validation_loss = val_loss_torch(val_values, Q, P, val_idx)
            validation_losses.append(validation_loss)

            # check if we improved
            if validation_loss < best_loss or best_loss == -1:
                converged_after = step
                best_loss = validation_loss
                best_Q, best_P = torch.tensor(Q), torch.tensor(P)
            
            # logging
            if step % log_every == 0:
                print('Iteration %i, training_loss: %f, validation loss: %f' % (step, train_loss, validation_loss))

            # stop if there is no change
            if step - converged_after >= patience:
                print('Converged after %i iterations' % converged_after)
                break
        
        # Optimize P
        for i in range(d_nnz_l):
            start = time.time()
            R_i = d_indices[i]
            r_i = d_data[i]
            _X = Q[R_i]
            _X_t = torch.t(_X)
            X = _X_t @ _X
            X += reg_lambda * torch.eye(k)
            y = _X_t @ r_i
            
            d_Xs[i] = X
            d_ys[i] = y
            
        X_LU = torch.btrifact(d_Xs)
        sol = torch.btrisolve(d_ys, *X_LU)
        P[:, d_nnz] = torch.t(sol)
        
        
        # Optimize Q
        for i in range(n_nnz_l):
            R_i = n_indices[i]
            r_i = n_data[i]
            _X = P[:, R_i]
            X = _X @ torch.t(_X) 
            X += reg_lambda * torch.eye(k)
            y = _X @ r_i
            
            n_Xs[i] = X
            n_ys[i] = y
            
        X_LU = torch.btrifact(n_Xs)
        sol = torch.btrisolve(n_ys, *X_LU)
        Q[n_nnz] = sol
            
        step += 1
    return best_P.cpu().numpy().T, best_Q.cpu().numpy(), validation_losses, train_losses, converged_after



def latent_factor_alternating_optimization(M, non_zero_idx, k, val_idx, val_values,
                                           reg_lambda, max_steps=100, init='random',
                                           log_every=1, patience=10, eval_every=1):
    N, D = M.shape
    # get the data in different formats for faster access
    csc = M.tocsc()
    csr = M.tocsr()
    lil = M.tolil()
    
    # precompute some arrays for faster computations
    n_zeros = [[] for i in range(N)]
    d_zeros = [[] for i in range(D)]
    nnz_values = {}
    
    non_zero_idx = np.array(non_zero_idx).T
    
    for ind, (i, j) in enumerate(non_zero_idx):
        n_zeros[i].append(j)
        d_zeros[j].append(i)
        nnz_values[i, j] = M[i, j]
        
    # initialization
    Q, P = initialize_Q_P(M, k, init=init)
    best_Q, best_P = Q.copy(), P.copy()
    step = 0
    train_losses = []
    validation_losses = []
    converged_after = 0
    best_loss = -1
    while step < max_steps:
        
        if step % log_every == 0 or  step % eval_every == 0:
            # compute training loss
            train_loss = loss(M, Q, P, 0, tuple(non_zero_idx.T))
            train_losses.append(train_loss)

            validation_loss = sse_array(val_values, Q.dot(P)[val_idx])
            validation_loss = sse_array(val_values, Q[val_idx[0]].dot(P[:,val_idx[1]]).diagonal())
            validation_losses.append(validation_loss)

            # check if we improved
            if validation_loss < best_loss or best_loss == -1:
                converged_after = step
                best_loss = validation_loss
                best_Q = Q.copy()
                best_P = P.copy()
            
            # logging
            if step % log_every == 0:
                print('Iteration %i, training_loss: %f, validation loss: %f' % (step, train_loss, validation_loss))

            # stop if there is no change
            if step - converged_after >= patience:
                print('Converged after %i iterations' % converged_after)
                break
        
        # Optimize P
        for i in range(D):
            R_i = d_zeros[i]
            length = len(R_i)
            if length == 0:
                continue
            r_i = np.zeros(length)
            for j in range(length):
                r_i[j] = nnz_values[(R_i[j], i)]
            X = Q[R_i].T.dot(Q[R_i]) + reg_lambda * np.eye(k)
            y = Q[R_i].T.dot(r_i)
            P[:,i] = np.linalg.solve(X, y)
            
        # Optimize Q
        for i in range(N):
            R_i = n_zeros[i]
            length = len(R_i)
            if length == 0:
                continue
            r_i = np.zeros(length)
            for j in range(length):
                r_i[j] = nnz_values[(i, R_i[j])]
            X = P[:,R_i].dot(P[:,R_i].T) + reg_lambda * np.eye(k)
            y = P[:,R_i].dot(r_i)
            Q[i] = np.linalg.solve(X, y)
            
        step += 1
    return best_P.T, best_Q, validation_losses, train_losses, converged_after



def sse(M, Q, P, ixs):
    step_size = 500
    result = 0
    for i in range(len(ixs[0])//step_size + 1):
        _ixs = tuple(j[i*step_size:(i+1)*step_size] for j in ixs)
        result += np.sum((M[_ixs] - Q[_ixs[0]].dot(P[:,_ixs[1]]).diagonal()).A1**2)
    return result

def val_sse(Q, P, vals, ixs):
    step_size = 500
    result = 0
    for i in range(len(ixs[0])//step_size + 1):
        _ixs = tuple(j[i*step_size:(i+1)*step_size] for j in ixs)
        result += np.sum((vals[i*step_size:(i+1)*step_size] - Q[_ixs[0]].dot(P[:,_ixs[1]]).diagonal())**2)
    return result

def sse_array(arr, vals):
    return np.sum((arr-vals)**2)

def loss(M, Q, P, reg_lambda, ixs):
    return sse(M,Q,P, ixs) + np.sum(reg_lambda * np.linalg.norm(P, axis=0)**2) + np.sum(reg_lambda * np.linalg.norm(Q, axis=1) ** 2)

def loss_torch(M, Q, P, reg_lambda, ixs):
    return val_loss_torch(M[ixs].todense(), Q, P, ixs)
    
def val_loss_torch(data, Q, P, ixs):
    data = data.reshape(-1)
    step_size = 1000
    result = 0
    for i in range(len(ixs[0])//step_size + 1):
        _ixs = tuple(j[i*step_size:(i+1)*step_size] for j in ixs)
        _data = data[i*step_size:(i+1)*step_size]
        result += torch.sum((_data - Q[_ixs[0]].matmul(P[:,_ixs[1]]).diag()) ** 2)
    return result

def cold_start_preprocessing(matrix, min_entries):
    """
    Recursively removes rows and columns from the input matrix which have less than min_entries nonzero entries.
    
    Parameters
    ----------
    matrix      : sp.spmatrix, shape [N, D]
                  The input matrix to be preprocessed.
    min_entries : int
                  Minimum number of nonzero elements per row and column.

    Returns
    -------
    matrix      : sp.spmatrix, shape [N', D']
                  The pre-processed matrix, where N' <= N and D' <= D
        
    """
    print("Shape before: {}".format(matrix.shape))
    
    shape = (-1, -1)
    while matrix.shape != shape:
        shape = matrix.shape
        nnz = matrix>0
        row_ixs = nnz.sum(1).A1 > min_entries
        matrix = matrix[row_ixs]
        nnz = matrix>0
        col_ixs = nnz.sum(0).A1 > min_entries
        matrix = matrix[:,col_ixs]
    print("Shape after: {}".format(matrix.shape))
    nnz = matrix>0
    assert (nnz.sum(0).A1 > min_entries).all()
    assert (nnz.sum(1).A1 > min_entries).all()
    return matrix

def cold_start_triple_preprocessing(R, F=None, B=None, A=None, UW=None, BW=None, min_entries=10):
    return pre.cold_start_preprocessing(R, F, B, A, UW, BW, min_entries)
    print("Shape before: {}".format(R.shape))
    
    M, N = R.shape
    user_idx = np.arange(N, dtype=np.int64)
    business_idx = np.arange(M, dtype=np.int64)
    
    shape = (-1, -1)
    while R.shape != shape:
        shape = R.shape
        nnz = R>0
        row_ixs = nnz.sum(1).A1 > min_entries
        R = R[row_ixs]
        if B is not None:
            B = B[row_ixs]
        if A is not None:
            A = A[row_ixs]
            A = A[:,row_ixs]
        if BW is not None:
            BW = BW[row_ixs]
        
        nnz = R>0
        col_ixs = nnz.sum(0).A1 > min_entries
        R = R[:,col_ixs]
        if F is not None:
            F = F[col_ixs]
            F = F[:,col_ixs]
        if UW is not None:
            UW = UW[col_ixs]
        # Correct indices
        M, N = R.shape
        removed_rows = np.where(row_ixs == False)[0]
        business_idx = update_idx(business_idx, removed_rows)
        
        removed_cols = np.where(col_ixs == False)[0]
        user_idx = update_idx(user_idx, removed_cols)
        
    print("Shape after: {}".format(R.shape))
    nnz = R>0
    assert (nnz.sum(0).A1 > min_entries).all()
    assert (nnz.sum(1).A1 > min_entries).all()
    result = [R, user_idx, business_idx]
    if F is not None:
        result.append(F)
    if B is not None:
        result.append(B)
    if A is not None:
        result.append(A)
    if UW is not None:
        result.append(UW)
    if BW is not None:
        result.append(BW)
    return tuple(result)

def update_idx(idx, removed):
    return idx[np.where(np.invert(np.isin(np.arange(len(idx)), removed)))]
