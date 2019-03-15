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
    plt.ylabel('Robot(s) Reading')
    plt.xlabel('Time')
    plt.show()

if __name__ == '__main__':
    x_0 = [20, 10, 15, 12, 30, 12, 15]  # initial conditions
    T = 20  # simulation time

    # Define graph 
    # E = [[0,2],[1,4],[2,1],[2,3],[2,5],[4,5],[6,3]] # for Figure 3(a)
    # E = [[0,2],[1,4],[2,1],[2,3],[3,6],[5,2],[5,4]] # for Figure 3(b)

    # E = [[1,4],[2,5],[4,5],[6,3]] # for Figure 4(a)
    # E = [[2,1],[5,4],[3,6]] # for Figure 4(b)
    E = [[0,2],[2,3],[5,2]] # for Figure 4(c)
    n_vertices = 7

    laplacian = gl.get_laplacian(E,n_vertices,isDirected = True)
    eigenVals = np.linalg.eigvals(laplacian)
    print("Eigen values are: ",np.sort(eigenVals))

    simulate_consensus(x_0, T, laplacian, dt =0.001)
    
    

