import numpy as np
import matplotlib.pyplot as plt
import getConstraints as gc
import getRigidityMat as rm
from matplotlib import animation

def show_animation(data, E, T, dt, n_ver):
    fig = plt.figure()
    
    ax1 = fig.add_subplot(221, autoscale_on=False,xlim=(-5,5), ylim=(-5,5))
    plt.title("Initial Positions of the Robots")
    ax1.plot(np.reshape(data[:,0],(n_ver,2))[:,0],(np.reshape(data[:,0],(n_ver,2))[:,1]),'bo')
    ax1.grid()

    ax2 = fig.add_subplot(222, autoscale_on=False,xlim=(-5,5), ylim=(-5,5))
    plt.title("Animation of the Robots")
    ax2.grid()
    time_text = ax2.text(0.02, .95, '', transform=ax2.transAxes)   

    ########################################
    ax3 = fig.add_subplot(223, autoscale_on=False,xlim=(0,T), ylim=(-3,3))
    plt.title("X position of the Robots vs Time")
    ax3.grid()
    time_text_x = ax3.text(0.02, .95, '', transform=ax3.transAxes)

    ax4 = fig.add_subplot(224, autoscale_on=False,xlim=(0,T), ylim=(-3,3))
    plt.title("Y position of the Robots vs Time")
    ax4.grid()
    time_text_y = ax4.text(0.02, .95, '', transform=ax4.transAxes)
    ########################################
    
    line_list = []

    for i in E: #add as many lines as there are edges
        line, = ax2.plot([], [], 'o-.', lw=2)
        line_list.append(line)

    for i in range(n_ver): #add as many vertices as there are in the graph
        line_x, = ax3.plot([], [])
        line_list.append(line_x)
        line_y, = ax4.plot([], [])
        line_list.append(line_y)

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
        time = i*skip*dt
        time_text.set_text('time = %.1fs' % time)
        time_text_x.set_text('time = %.1fs' % time)
        time_text_y.set_text('time = %.1fs' % time)
        
        # Method 1

        xdata = np.reshape(data[:,i*skip],(n_ver,2))[:,0]
        ydata = np.reshape(data[:,i*skip],(n_ver,2))[:,1]

        for j,edge in enumerate(edges):
            [x1, y1] = [xdata[edge[0]], ydata[edge[0]]]
            [x2, y2] = [xdata[edge[1]], ydata[edge[1]]]
            line_list[j].set_data([x1,x2] ,[y1,y2])

        for j in range(n_ver):
            line_list[j*2+len(edges)].set_data(time_arr[:i*skip],data[j*2,:i*skip])
            line_list[j*2+1+len(edges)].set_data(time_arr[:i*skip],data[j*2+1,:i*skip])

        # Method 2

        # for j,edge in enumerate(edges):
        #     [x1, y1] = [data[2*edge[0],i*10], data[2*edge[0]+1,i*10]]
        #     [x2, y2] = [data[2*edge[1],i*10], data[2*edge[1]+1,i*10]]
        #     line_list[j].set_data([x1,x2] ,[y1,y2])

        return line_list
    
    skip=10     # factor to skip the no. of frames from previous frame 

    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, fargs = [data, E, dt, skip,n_ver,T], frames=int(T/dt/skip), init_func = None, interval=10, blit=True, repeat = False)

    # anim.save("hw3\movie.mp4")
    plt.show(fig)
    plt.close('all')


def formation_control(p_0, gd, edges, T, dt =0.001):
    time_arr = np.arange(0,T,dt)
    
    # initialize x
    p = np.zeros((len(p_0),len(time_arr)))

    curr_rig_mat = np.transpose(rm.get_rigidity_mat(edges,p_0))
    curr_constraints = gc.get_constraints(edges,p_0)
    p[:,0] = p_0
    
    for i in range(0,len(time_arr)-1) :
        curr_rig_mat = np.transpose(rm.get_rigidity_mat(edges,p[:,i]))
        curr_constraints = gc.get_constraints(edges,p[:,i])
        p[:,i+1] =  (np.matmul(curr_rig_mat, np.subtract(gd,curr_constraints)) * dt) + p[:,i]

    return p


if __name__ == "__main__":

    T = 10  # Total Time for integration
    dt = 0.001  # step for integration

    # Define graph 
    E = [[0,3],[0,4],[1,2],[1,4],[2,3],[2,4],[3,4]]
    n_edges = len(E)

    # Number of vertices
    n_ver = np.max(np.reshape(E,(n_edges,2)))+1
    # print("Number of vertices = {0}".format(n_ver))

    # Initial Postion vector stacked with each p(i) = [x,y] i.e. x and y coordinate of point p(i)
    
    # # Bad Initialization
    # init_pos_vect = [0,1,1,0,2,2,3,2,4,0]
    
    # Good Initialization
    init_pos_vect = [0.3,1.9,0.9,1.2,0.6,1,-2,2.1,-0.5,0.9]
    
    des_pos_vect = [0,2,1,1,0,0,-1,1,0,1] 
    des_constraints = gc.get_constraints(E,des_pos_vect)
    data = formation_control(init_pos_vect, des_constraints, E, T,dt = dt)
    show_animation(data, E, T, dt, n_ver)