import numpy as np

def get_laplacian(edges,numVertices,isDirected = False):
    # Initialize Adjacency Matrix
    adj_mat = np.zeros((numVertices,numVertices),dtype=int)
  
    # Compute Adjacency Matrix
    for edge in edges:
        i = edge[1]  #head
        j = edge[0]  #tail
        adj_mat[i][j] = 1
        if not isDirected :
            adj_mat[j][i] = 1

    # Initialize Degree Matrix
    deg_mat = np.zeros((numVertices,numVertices),dtype=int)
  
    # Compute Degree Matrix
    for edge in edges:
        k = edge[1]  # for directed graph only in-degree
        deg_mat[k][k] += 1
        if not isDirected :
            l = edge[0]  # for undirected graph count both in and out degree
            deg_mat[l][l] += 1
    
    # Compute Laplacian Matrix
    lap_mat = np.subtract(deg_mat,adj_mat)

    # print("Adjacency Matrix:\n",adj_mat)
    # print("Degree Matrix:\n",deg_mat)
    return lap_mat

if __name__ == '__main__':

    # list of edges example 1
    E = [[0,3],[0,5],[0,4],[3,5],[5,4],[4,3], \
        [1,6],[1,7],[1,8],[6,7],[7,8],[8,6], \
        [2,9],[2,10],[2,11],[9,10],[10,11],[11,9], \
        [0,1],[1,2],[2,0]]
    n_vertices = 12

    # list of edges example 2
    # E = [ [ 0 , 1 ] , [ 1 , 2 ] , [ 2 , 0 ] ]
    # n_vertices = 3

    # list of edges example 3
    # E = [[1,0],[1,2],[2,4],[4,3],[3,2]]
    # n_vertices = 5

    # list of edges example 4
    # E = [[0,2],[1,2],[3,2],[2,4],[2,6],[5,4],[7,4],[4,6],[5,6],[7,6],[8,6]]
    # n_vertices = 9

    # list of edges example 5
    # E = [[1,0],[2,0],[0,3],[0,4],[0,5],[5,4],[4,3],[1,6],[1,7],[1,8],[8,7],[7,6],[2,9],[2,10],[2,11],[11,10],[10,9]]
    # n_vertices = 12

    # list of edges example 6
    # E = [[0,1],[0,2],[0,3],[2,3],[3,4],[4,5],[5,6],[6,7],[5,7]]
    # n_vertices = 8

    laplacian = get_laplacian(E,n_vertices,False)
    print("Laplacian:\n",laplacian)

    eigenVals = np.linalg.eigvals(laplacian)

    print("Eigen values are: ",np.sort(eigenVals))