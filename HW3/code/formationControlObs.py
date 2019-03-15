import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time

def show_animation(data, edges, obs,T, dt, n_ver):
    fig = plt.figure()

    ax2 = fig.add_subplot(122, autoscale_on=True,xlim=(-1,15), ylim=(-1,15))
    plt.title("Animation of the Robots")
    plt.xlabel("Position x")
    plt.ylabel("Position y")
    ax2.grid()
    time_text = ax2.text(0.02, .95, '', transform=ax2.transAxes)   

    ########################################
    ax3 = fig.add_subplot(221, autoscale_on=False,xlim=(-1,15), ylim=(-1,15))
    plt.title("X position of the Robots vs Time")
    plt.xlabel("Time")
    plt.ylabel("Position x")
    ax3.grid()
    time_text_x = ax3.text(0.02, .95, '', transform=ax3.transAxes)

    ax4 = fig.add_subplot(223, autoscale_on=False,xlim=(-1,15), ylim=(-1,15))
    plt.title("Y position of the Robots vs Time")
    plt.xlabel("Time")
    plt.ylabel("Position y")
    ax4.grid()
    time_text_y = ax4.text(0.02, .95, '', transform=ax4.transAxes)
    ########################################

    line_list = []

    for i in edges: #add as many lines as there are edges
        line, = ax2.plot([], [], 'o-.', lw=2)
        line_list.append(line)

    for i in range(n_ver): #add as many vertices as there are in the graph
        line_x, = ax3.plot([], [])
        line_list.append(line_x)
        line_y, = ax4.plot([], [])
        line_list.append(line_y)
    
    for obs in obstacles: #as many rounds as there are obstacles
        line, = ax2.plot([obs[0]],[obs[1]],'ko-', lw=15)
        line_list.append(line)

        
   

    line_list.append(time_text)
    line_list.append(time_text_x)
    line_list.append(time_text_y)
    
    # # initialization function: plot the background of each frame
    # def init():
    #     for line in line_list:
    #         line.set_data([], [])
    #     return line_list

    # animation function. This is called sequentially
    def animate(i,data,edges,dt,skip,n_ver,T):
        time_arr = np.linspace(0, T, T/dt)
        time_elaspsed = i*skip*dt
        time_text.set_text('time = %.1fs' % time_elaspsed)
        time_text_x.set_text('time = %.1fs' % time_elaspsed)
        time_text_y.set_text('time = %.1fs' % time_elaspsed)

        # Method 1

        xdata = np.reshape(data[:,i*skip],(2,n_ver))[0,:]
        ydata = np.reshape(data[:,i*skip],(2,n_ver))[1,:]

        for j,edge in enumerate(edges):
            [x1, y1] = [xdata[edge[0]], ydata[edge[0]]]
            [x2, y2] = [xdata[edge[1]], ydata[edge[1]]]
            line_list[j].set_data([x1,x2] ,[y1,y2])

        for j in range(n_ver):
            line_list[j*2+len(edges)].set_data(time_arr[:i*skip],data[j*2,:i*skip])
            line_list[j*2+1+len(edges)].set_data(time_arr[:i*skip],data[j*2+1,:i*skip])
        
        return line_list
    
    skip=10     # factor to skip the no. of frames from previous frame 

    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, fargs = [data, edges, dt, skip,n_ver,T], frames=int(T/dt/skip), init_func = None, interval=10, blit=True, repeat = False)

    anim.save("hw3\movie.mp4")
    plt.show(fig)
    plt.close('all')

def cal_obs_dist(pos,obs):
    return np.sqrt(np.sum(np.square(np.subtract(pos,obs))))

def evaluate_control_law(L,E,pos,vel,zref,zref_vel,goal,obstacles):
    ########################################
    # Coeficients for the control law
    KF = 10
    KT = 5
    KO = 10
    DF = 2*np.sqrt(KF)
    DT = 2*np.sqrt(KT)
    dmax = 1.0
    ########################################
    formation_law = (-KF*np.matmul(L,pos)) + (KF*np.ndarray.flatten(np.matmul(E,zref))) + (-DF*np.matmul(L,vel)) + (DF*np.ndarray.flatten(np.matmul(E,zref_vel)))
    target_law = (KT*np.minimum(goal-pos,1)) - (DT*vel)


    obs_vect = []

    force_obs = np.zeros((2*n_ver,len(obstacles)))
    
    for j,obs in enumerate(obstacles):
        for i in range(n_ver):
            curr_obs_dist = cal_obs_dist([pos[i],pos[i+n_ver]] , obs)
            if(curr_obs_dist<dmax):
                force_obs[i][j] = (-KO*(curr_obs_dist-dmax)/(curr_obs_dist ** 3))*(pos[i]-obs[0])  # for x coordinate
                force_obs[i+n_ver][j] = (-KO*(curr_obs_dist-dmax)/(curr_obs_dist ** 3))*(pos[i+n_ver]-obs[1])  # for y coordinate
            else:
                force_obs[i][j] = 0
                force_obs[i+n_ver][j] = 0

    obstacle_law = np.sum(force_obs,axis = 1)

    return (formation_law + target_law + obstacle_law)

def simulate(p_0,v_0,pref,obs,L,E,goal,n_ver,T = 10, dt = 0.001):  
    p_0 = np.reshape(p_0,(n_ver,2))
    p_0 = np.hstack((p_0[:,0],p_0[:,1]))    # Converting P[x1,y2,x2,y2] to P[x1,x2,y1,y2]

    pref = np.reshape(pref,(n_ver,2))
    pref = np.hstack((pref[:,0],pref[:,1])) # Converting Pref[x1,y2,x2,y2] to Pref[x1,x2,y1,y2]

    # Constructing goal vector [x1, ..., xn, y1, ..., yn]
    goal_vect = []
    goal_vect[0:n_ver-1] = [goal[0] for i in range(n_ver)]  # storing x positions
    goal_vect[n_ver:2*n_ver-1] = [goal[1] for i in range(n_ver)]  # storing y positions

    velocity = np.zeros((n_ver*2,int(T/dt)))
    position = np.zeros((n_ver*2,int(T/dt)))

    velocity[:,0] = v_0
    position[:,0] = p_0

    block_L_mat = np.bmat([[L,np.zeros((n_ver,n_ver))],[np.zeros((n_ver,n_ver)),L]])
    block_E_mat = np.bmat([[E,np.zeros((n_ver,n_ver))],[np.zeros((n_ver,n_ver)),E]])
    block_E_tr_mat = np.bmat([[np.transpose(E),np.zeros((n_ver,n_ver))],[np.zeros((n_ver,n_ver)),np.transpose(E)]])
    zref = np.transpose(np.matmul(block_E_tr_mat,pref))
    zref_vel = np.zeros((2*n_ver,1)) 

    for i in range(int(T/dt)-1):
        velocity[:,i+1] = velocity[:,i] + dt*(evaluate_control_law(block_L_mat,block_E_mat,position[:,i],velocity[:,i],zref,zref_vel,goal_vect,obs))
        position[:,i+1] = position[:,i] + dt*velocity[:,i+1]

    return position
        


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

def get_incidence(edges,numVertices):
    # Initialize Incidence Matrix
    inc_mat = np.zeros((numVertices,len(edges)),dtype=int)
  
    # Compute Incidence Matrix
    for edge in edges:
        i = edge[1]  #head
        j = edge[0]  #tail
        k = edges.index(edge)
        inc_mat[i][k] = 1
        inc_mat[j][k] = -1

    return inc_mat

if __name__ == "__main__":

    T = 15
    dt = 0.001  # step for integration
    
    # Define graph 
    edges = [[0,1],[1,2],[2,3],[3,0]]
    n_edges = len(edges)

    # Number of vertices
    n_ver = np.max(np.reshape(edges,(n_edges,2)))+1
    
    # Initial Postion vector stacked with each p(i) = [x,y] i.e. x and y coordinate of point p(i)
    # init_pos = [0,0,1,1,2,2,3,3]
    init_pos = [0,0,-0.2,-0.2,0.2,0,0,0.1]
    des_pos = [9.5,10.5,9.5,9.5,10.5,9.5,10.5,10.5]
    # des_pos = [0,0,0,1,1,1,1,0]

    # Initial Velocity vector stacked with each v(i) = [v(x),v(y)] i.e. x and y velocities of point p(i)
    init_vel = [0,0,0,0,0,0,0,0]

    goal = [10,10]

    # No obstacles
    obstacles = []

    # # 8 obstacles
    # obstacles = [[4,0],[4,4.7],[4,8],[6,2],[7,6],[6,10],[8,4],[8,12]]
    
    # # 9 obstacles
    # obstacles = [[4,0],[4,4.7],[4,8],[6,2],[7,6],[6,10],[8,4],[8,12],[8.1,5.6]]
    
    # obstacles = [[6,0],[6,0.5],[6,1],[6,1.5],[6,2],[6,2.5],[6,3],[6,3.5],[6,4],[6,4.5],[6,5],[0,6],[0.5,6],[1,6],[1.5,6],[2,6],[2.5,6],[3,6],[3.5,6],[4,6],[4.5,6],[5,6]]


    L = get_laplacian(edges,n_ver)
    E = get_incidence(edges,n_ver)

    data = simulate(init_pos,init_vel,des_pos,obstacles,L,E,goal,n_ver,T,dt)
    show_animation(data, edges, obstacles, T, dt, n_ver)


    