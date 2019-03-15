import numpy as np
import matplotlib.pyplot as plt
import getLaplacian as gl # user created


def simulate_consensus (x_0, T, L, dt =0.001):
    time_arr = np.arange(0,T,dt)
    
    # initialize x
    x = np.zeros((len(x_0),len(time_arr)))
    x[:,0] = x_0     # copy x_0 to 1st col of x
   
    for i in range(0,len(time_arr)-1) :
        x[:,i+1] = (-1)*(np.matmul(L,x[:,i]))*dt + x[:,i]
        
    for j in range(0,len(x_0)):
        plt.plot(time_arr,x[j,:])    
    plt.show() 

if __name__ == '__main__':
    x_0 = [20, 10, 15, 12, 30, 12, 15, 16, 25]  # initial conditions
    T = 20  # simulation time

    # Define graph 
    # E = [[0,2],[1,2],[3,2],[2,4],[2,6],[5,4],[7,4],[4,6],[5,6],[7,6],[8,6]] # for Exercise 5 - 2
    E = [[0,2],[1,2],[3,2],[2,6],[5,4],[7,4],[4,6],[5,6],[8,6]]  # for Exercise 5 - 3
    # E = [[0,2],[1,2],[3,2],[2,4],[2,6],[5,4],[7,4],[4,6],[7,6],[8,6],[1,3],[2,8],] # for Exercise 5 - 4
    # E = [[0,2],[1,2],[3,2],[2,4],[5,4],[7,6],[8,6]] # for Exercise 5 - 5
    n_vertices = 9

    laplacian = gl.get_laplacian(E,n_vertices,isDirected = False)
    eigenVals = np.linalg.eigvals(laplacian)
    print("Eigen values are: ",np.sort(eigenVals))

    simulate_consensus(x_0, T, laplacian, dt =0.001)
    
    

