def algorithm(K1,K2, B, weight_vec, datapoints, true_labels, lambda_lasso, penalty_func_name='norm1', calculate_score=False):
    '''
    Outer loop is gradient ascent algorithm, the variable 'new_weight_vec' here is the dual variable we are interested in
    
    Inner loop is still our Nlasso algorithm
    
    :param K1,K2 : the number of iterations
    :param D: the block incidence matrix 
    :param weight_vec: a list containing the edges's weights of the graph
    :param datapoints: a dictionary containing the data of each node in the graph needed for the algorithm 1 
    :param true_labels: a list containing the true labels of the nodes
    :param samplingset: the sampling set 
    :param lambda_lasso: the parameter lambda 
    :param penalty_func_name: the name of the penalty function used in the algorithm
    
    
    :return new_w: the predicted weigh vectors for each node
    '''
    
    
    '''
    Sigma: the block diagonal matrix Sigma
    
    
    '''
    Sigma = np.diag(np.full(weight_vec.shape, 0.9 / 2))
    T_matrix = np.diag(np.array((1.0 / (np.sum(abs(B), 0)))).ravel())
    '''
    T_matrix: the block diagonal matrix T
    '''

    E, N = B.shape
    '''
    shape of the graph
    '''
    m, n = datapoints[1]['features'].shape
    '''
    shape of the feature vectors of each node in the graph
    '''
#     # define the penalty function
#     if penalty_func_name == 'norm1':
#         penalty_func = Norm1Pelanty(lambda_lasso, weight_vec, Sigma, n)
#     elif penalty_func_name == 'norm2':
#         penalty_func = Norm2Pelanty(lambda_lasso, weight_vec, Sigma, n)
#     elif penalty_func_name == 'mocha':
#         penalty_func = MOCHAPelanty(lambda_lasso, weight_vec, Sigma, n)
#     else:
#         raise Exception('Invalid penalty name')

    new_w = np.array([np.zeros(n) for i in range(N)])
    
    new_u = np.array([np.zeros(n) for i in range(E)])
    new_weight_vec = weight_vec
    
    # starting algorithm 1
    Loss = {}
    iteration_scores = []
    for j in range(K1):
        
        new_B = np.dot(np.diag(new_weight_vec),B)
        T_matrix = np.diag(np.array((1.0 / (np.sum(abs(new_B), 0)))).ravel())
        T = np.array((1.0 / (np.sum(abs(new_B), 0)))).ravel()
        
        
        for iterk in range(K2):
            # if iterk % 100 == 0:
            #     print ('iter:', iterk)

            prev_w = np.copy(new_w)

            # line 2 algorithm 1
            hat_w = new_w - np.dot(T_matrix, np.dot(new_B.T, new_u))


            for i in range(N):
                optimizer = datapoints[i]['optimizer']
                new_w[i] = optimizer.optimize(datapoints[i]['features'], 
                                              datapoints[i]['label'], 
                                              hat_w[i], 
                                              T[i])


            # line 9 algortihm 1
            tilde_w = 2 * new_w - prev_w
            new_u = new_u + np.dot(Sigma, np.dot(new_B, tilde_w))

            penalty_func = Norm1Pelanty(lambda_lasso, new_weight_vec, Sigma, n)
            new_u = penalty_func.update(new_u)
        new_weight_vec = new_weight_vec +0.1*np.linalg.norm(np.dot(B, new_w),ord=1,axis=1)
            
            
#         # calculate the MSE of the predicted weight vectors
#         if calculate_score:
#             Y_pred = []
#             for i in range(N):
#                 Y_pred.append(np.dot(datapoints[i]['features'], new_w[i]))

#             iteration_scores.append(mean_squared_error(true_labels.reshape(N, m), Y_pred))

        Loss[j] = total_loss(datapoints,new_w,new_B,new_weight_vec)

    return new_w, new_weight_vec,Loss,iteration_scores