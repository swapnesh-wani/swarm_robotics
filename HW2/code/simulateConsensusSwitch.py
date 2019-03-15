import numpy as np
import matplotlib.pyplot as plt
import getLaplacian as gl # user created


def simulate_consensus_switch (x_0, T, L_list, switch_time, dt =0.001):
    time_arr = np.arange(0,T,dt)
    
    # initialize x
    x = np.zeros((len(x_0),len(time_arr)))
    x[:,0] = x_0     # copy x_0 to 1st col of x
    
    start_time = 0
    end_time = 0
    L_idx = 0
    L = L_list[L_idx]       # Initiaize to first Laplacian

    for i in range(0,len(time_arr)-1) :
        if (round(end_time - start_time,4) == switch_time) :
            L = L_list[L_idx%len(L_list)]
            L_idx = L_idx + 1
            start_time = time_arr[i]
        end_time = time_arr[i+1]
        x[:,i+1] = (-1)*(np.matmul(L,x[:,i]))*dt + x[:,i]

    print("Number of Switches: ",L_idx) 

    for j in range(0,len(x_0)):
        plt.plot(time_arr,x[j,:])    
    plt.ylabel('Robot(s) Reading')
    plt.xlabel('Time')
    plt.show()

if __name__ == '__main__':
    x_0 = [20, 10, 15, 12, 30, 12, 15]  # initial conditions
    T = 300 # simulation time
    switch_time = 2

    # Define graph 
    # E1 = [[0,2],[1,2],[1,4],[2,0],[2,1],[2,3],[2,5],[3,2],[3,6],[4,1],[4,5],[5,2],[5,4],[6,3]] # for Figure 2(a)
    # E2 = [[0,2],[1,2],[1,4],[2,0],[2,1],[2,3],[3,2],[3,6],[4,1],[4,5],[5,4],[6,3]] # for Figure 2(b)
    # E3 = [[0,2],[1,2],[2,0],[2,1],[2,3],[2,5],[3,2],[3,6],[4,5],[5,2],[5,4],[6,3]] # for Figure 2(c)
    # E = [E1,E2,E3]

    # E1 = [[0,2],[1,4],[2,1],[2,3],[2,5],[4,5],[6,3]] # for Figure 3(a)
    # E2 = [[0,2],[1,4],[2,1],[2,3],[3,6],[5,2],[5,4]] # for Figure 3(b)
    # E = [E1,E2]

    E1 = [[1,4],[2,5],[4,5],[6,3]] # for Figure 4(a)
    E2 = [[2,1],[5,4],[3,6]] # for Figure 4(b)
    E3 = [[0,2],[2,3],[5,2]] # for Figure 4(c)
    E = [E1,E2,E3]

    laplacian_list = []
    n_vertices = 7

    for i in range(0,len(E)):
        laplacian_list.append(gl.get_laplacian(E[i],n_vertices,isDirected = True))


    simulate_consensus_switch(x_0, T, laplacian_list, switch_time, dt =0.001)
    
    

