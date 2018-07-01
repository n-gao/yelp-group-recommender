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
            train_loss = loss(M, Q, P, reg_lambda, tuple(non_zero_idx.T))
            train_losses.append(train_loss)

            validation_loss = sse_array(val_values, Q.dot(P)[val_idx])
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