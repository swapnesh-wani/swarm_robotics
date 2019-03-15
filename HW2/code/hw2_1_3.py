import numpy as np
import getLaplacian as gl
import scipy.linalg as spla
import simulateConsensus as sc

if __name__ == '__main__':
    # Exercise 2 : Figure 1(a)
    E = [[0,1],[0,2],[1,4],[2,3],[2,5],[3,0],[3,4],[3,6],[4,7],[6,3]]
    n_vertices = 8
    x_0 = [20, 10, 15, 12, 30, 12, 15, 16]  # initial conditions

    # # Exercise 2 : Figure 1(b)
    # E = [[0,1],[0,2],[0,3],[1,4],[2,5],[3,6],[4,7]]
    # n_vertices = 8
    # x_0 = [20, 10, 15, 12, 30, 12, 15, 16]  # initial conditions

    # # Exercise 2 : Figure 1(c)
    # E = [[0,1],[0,3],[0,6],[1,0],[2,4],[3,5],[4,2],[4,6],[5,1]]
    # n_vertices = 7
    # x_0 = [20, 10, 15, 12, 30, 12, 15]  # initial conditions

    # # Exercise 2 : Figure 1(d)
    # E = [[0,3],[1,0],[1,2],[2,1],[3,1],[3,5],[4,3],[5,6],[6,7],[7,4]]
    # n_vertices = 8
    # x_0 = [20, 10, 15, 12, 30, 12, 15, 16]  # initial conditions

    laplacian = gl.get_laplacian(E,n_vertices,isDirected = True)
    print("Laplacian:\n",laplacian)

    eigenVals, leftEigenVectors = spla.eig(laplacian, b=None, left=True, right=False)
    
    # Finding the index of least eigen value 
    idx_least_ev = np.where(eigenVals == np.sort(eigenVals)[0])

    # Finding the eigen vector of least eigen value
    leftEV = np.reshape(leftEigenVectors[:,idx_least_ev],(len(x_0),1))

    # Applying proper scaling such that q'1=1 which is essentially the sum of all the elements of q
    leftEV = leftEV / np.sum(leftEV)
    
    
    print("Eigen values are: ",eigenVals)
    print("\nIndex of lowest Eigen Value is: ",idx_least_ev)
    print("\nEigen Vector with corresponding Eigen Value = 0 is: ",leftEV)
    
    T = 20  # simulation time
    print("\nConcensus Value = ",np.matmul(np.reshape(x_0,(1,len(x_0))),leftEV))
    sc.simulate_consensus(x_0, T, laplacian, dt =0.001)