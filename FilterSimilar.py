
def get_B_and_weight_vec_from_SimilarityMatrix(distributions, neigh_cnt=3):
    
    ''' Calculate the adjacency matrix and the weight vector of the empirical graph G
    Args:
        trained_models_train_images: A list containing the images used for training each model
        neigh_cnt: number of the neighbors for each node of the empirical graph G
    Returns:
        the adjacency matrix of the empirical graph G
        the weight vector of the edges of the empirical graph G

    '''
    
    N = len(distributions)
    E = int(N * (N - 1) / 2)

    weight_vec = np.zeros(E)
    '''
    the weight vector of the edges of the empirical graph G
    '''
    B = np.zeros((E, N))
    '''
    the adjacency matrix of the empirical graph G
    '''
    
    cnt = 0
    '''
    number of edges of the empirical graph G
    '''
    for i in range(N):
        node_dists = []
        '''
        a list containing the distance between node i and other nodes of the graph
        '''
        for j in range(N):
            if j == i:
                continue
            node_dists.append(bhattacharyya_dist(distributions[i], distributions[j]))
        
        # sort node_dists in order to pick the nearest nodes to the node i 
        node_dists.sort(reverse=True)

        node_cnt = 0
        for j in range(N):
            
            if node_cnt >= neigh_cnt:
                break
                
            if j == i:
                continue
                
            # calculate the distance between node i and j of the graph
            dist = bhattacharyya_dist(distributions[i], distributions[j])
            if dist == 0 or dist < node_dists[neigh_cnt]:
                continue

            node_cnt += 1
            B[cnt][i] = 1
            B[cnt][j] = -1
            weight_vec[cnt] = dist
            cnt += 1

    B = B[:cnt, :]
    weight_vec = weight_vec[:cnt]
    return B, weight_vec