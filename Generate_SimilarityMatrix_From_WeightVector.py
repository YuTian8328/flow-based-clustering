def generate_similarity_matrix(weight_vector):
    '''
    Generate similarity matrix from the weight vector
    '''
    
    similarity_matrix = np.zeros((20,20))
    cnt=0
    for i in range(20):
        similarity_matrix[i,i+1:]=weight_vector[cnt:cnt+20-i]
        cnt+=20-i
    return similarity_matrix+similarity_matrix.T