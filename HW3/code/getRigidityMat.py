import numpy as np
import getConstraints as gc

def get_rigidity_mat(edges,pos_vect):
    ''' In this function rigidity matrix is calculated per the edges passed
    '''
    no_cols = len(pos_vect)   # since the robots are in 2 dimension therfore 2 * # robots
    no_rows = len(edges)                 # no of edges in graph is 6

    rigidity_mat = np.matrix(np.zeros((no_rows,no_cols)))

    pos_vect = np.reshape(pos_vect,(5,2))

    for i,edge in enumerate(edges):
        # print("i = {0}, edge = {1}".format(i,edge))
        [rigidity_mat[i,2*edge[0]] , rigidity_mat[i,2*edge[0]+1]] = 2*np.subtract(pos_vect[edge[0]],pos_vect[edge[1]])
        [rigidity_mat[i,2*edge[1]] , rigidity_mat[i,2*edge[1]+1]] = -2*np.subtract(pos_vect[edge[0]],pos_vect[edge[1]])

    return rigidity_mat

if __name__ == "__main__":
    # Define graph 
    E = [[0,3],[0,4],[1,2],[1,4],[2,4],[3,4]]
    # E = [[0,1],[0,3],[0,4],[1,2],[1,4],[2,4],[3,4]]

    # Postion vector stack with each p(i) = [x,y] i.e. x and y coordinate of point p(i)
    # pos_vect = [0,2,1,1,0,0,-1,1,0,1]   
    pos_vect = [0,0,1,1,2,2,3,3,4,4]

    
    rigidity_mat = get_rigidity_mat(E,pos_vect)
    print(rigidity_mat)

    