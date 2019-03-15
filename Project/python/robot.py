import numpy as np
import pybullet as p
import itertools
import os

class Robot():
    """ 
    The class is the interface to a single robot
    """
    def __init__(self, init_pos, robot_id, dt):
        self.id = robot_id
        self.dt = dt
        self.pybullet_id = p.loadSDF(os.path.join(os.path.dirname(__file__),"../models/robot.sdf"))[0]
        self.joint_ids = list(range(p.getNumJoints(self.pybullet_id)))
        self.initial_position = init_pos
        self.reset()
        self.time_elapsed = 0
        self.switch_time = 10
        self.is_goal = 0
        self.goal = []

        self.gx = None
        self.gy = None

        # No friction between bbody and surface.
        p.changeDynamics(self.pybullet_id, -1, lateralFriction=5., rollingFriction=0.)

        # Friction between joint links and surface.
        for i in range(p.getNumJoints(self.pybullet_id)):
            p.changeDynamics(self.pybullet_id, i, lateralFriction=5., rollingFriction=0.)
            
        self.messages_received = []
        self.messages_to_send = []
        self.neighbors = []
        

    def reset(self):
        p.resetBasePositionAndOrientation(self.pybullet_id, self.initial_position, (0., 0., 0., 1.))
            
    def set_wheel_velocity(self, vel):
        """ 
        Sets the wheel velocity,expects an array containing two numbers (left and right wheel vel) 
        """
        assert len(vel) == 2, "Expect velocity to be array of size two"
        p.setJointMotorControlArray(self.pybullet_id, self.joint_ids, p.VELOCITY_CONTROL,
            targetVelocities=vel)

    def get_pos_and_orientation(self):
        """
        Returns the position and orientation (as Yaw angle) of the robot.
        """
        pos, rot = p.getBasePositionAndOrientation(self.pybullet_id)
        euler = p.getEulerFromQuaternion(rot)
        return np.array(pos), euler[2]
    
    def get_messages(self):
        return self.messages_received
        
    def send_message(self, robot_id, message):
        self.messages_to_send.append([robot_id, message])
        
    def get_neighbors(self):
        return self.neighbors
    
    def compute_controller(self):
        """ 
        function that will be called each control cycle which implements the control law
        TO BE MODIFIED
        
        we expect this function to read sensors (built-in functions from the class)
        and at the end to call set_wheel_velocity to set the appropriate velocity of the robots
        """
        # here we implement an example for a consensus algorithm
        neig = self.get_neighbors()
        messages = self.get_messages()
        pos, rot = self.get_pos_and_orientation()
        
        #send message of positions to all neighbors indicating our position

    
        
        for n in neig:
            self.send_message(n, pos)

        
        self.time_elapsed += self.dt

        

        if self.time_elapsed < self.switch_time:
            # Form Square
            self.goal = [[0.5,-1.5],[0.5,0.5],[1.5,-1.5],[2.5,-0.5],[2.5,-1.5],[2.5,0.5]]
            self.gx,self.gy = np.average(self.goal,axis=0)    

        elif self.time_elapsed > self.switch_time and self.time_elapsed < 2*self.switch_time:
            # Form L
            self.goal = [[0.5,-1.5],[2.5,1.5],[1.5,-1.5],[2.5,-0.5],[2.5,-1.5],[2.5,0.5]]
            self.gx,self.gy = np.average(self.goal,axis=0)  
        elif self.time_elapsed > 2*self.switch_time and self.time_elapsed < 6*self.switch_time:
             # Form verticle line
            self.goal = [[2.5,3.5],[2.5,7.5],[2.5,2.5],[2.5,5.5],[2.5,4.5],[2.5,6.5]]
            self.gx,self.gy = np.average(self.goal,axis=0)  
        elif self.time_elapsed > 6.5*self.switch_time and self.time_elapsed < 8*self.switch_time:
            # Form horizontal line
            self.goal = [[0.5,5.5],[1.5,5.5],[2.5,5.5],[3.5,5.5],[4.5,5.5],[5.5,5.5]]
            self.gx,self.gy = np.average(self.goal,axis=0)  
        elif self.time_elapsed > 9*self.switch_time and self.time_elapsed < 11*self.switch_time:
            # Form horizontal line in main arena
            self.goal = [[0.5-8,5.5],[1.5-8,5.5],[2.5-8,5.5],[3.5-8,5.5],[4.5-8,5.5],[5.5-8,5.5]]
            self.gx,self.gy = np.average(self.goal,axis=0)  
        elif self.time_elapsed > 11*self.switch_time and self.time_elapsed < 18*self.switch_time:
            # Form triangle
            self.goal = [[-6.5,9.5],[-5.5,10.5],[-4.5,11.5],[-3.5,10.5],[-4.5,9.5],[-2.5,9.5]]
            self.gx,self.gy = np.average(self.goal,axis=0)  
       
        # self.goal = [[0.5+5,-1.5+5],[2.5+5,1.5+5],[1.5+5,-1.5+5],[2.5+5,0.5+5],[2.5+5,-1.5+5],[2.5+5,-0.5+5]]

        # # Form straight line
        # self.goal = [[2,-1.5],[2,1],[2,-1],[2,0.5],[2,-0.5],[2,0]]
        
        # check if we received the position of our neighbors and compute desired change in position
        # as a function of the neighbors (message is composed of [neighbors id, position])
        dx = 0.
        dy = 0.

        ax=0
        ay=0
        dmax = 1
        KF = 10
        KT = 2.5
        DT = np.sqrt(KT)
        KO = 7

        if messages:
            for m in messages:
                curr_obs_dist = np.linalg.norm(pos-m[1][0])

                if(curr_obs_dist<dmax):
                    ax += (-KO*(curr_obs_dist-dmax)/(curr_obs_dist ** 3))*(pos[0]-m[1][0])  # for x coordinate
                    ay += (-KO*(curr_obs_dist-dmax)/(curr_obs_dist ** 3))*(pos[1]-m[1][1])  # for y coordinate
                else:
                    ax += 0
                    ay += 0

        if messages:
            for m in messages:
                dx += KF*((m[1][0]-pos[0]) - (self.goal[m[0]][0]-self.goal[self.id][0])) + KT*np.minimum(self.goal[self.id][0]-pos[0],1) - DT*(p.getBaseVelocity(self.pybullet_id)[0][0]) + self.dt*ax + p.getBaseVelocity(self.pybullet_id)[0][0]
                dy += KF*((m[1][1]-pos[1]) - (self.goal[m[0]][1]-self.goal[self.id][1])) + KT*np.minimum(self.goal[self.id][1]-pos[1],1) - DT*(p.getBaseVelocity(self.pybullet_id)[0][1]) + self.dt*ay + p.getBaseVelocity(self.pybullet_id)[0][1]
            # # integrate
            # des_pos_x = pos[0] + self.dt * dx
            # des_pos_y = pos[1] + self.dt * dy

            #compute velocity change for the wheels
            vel_norm = np.linalg.norm([dx, dy]) #norm of desired velocity
            if vel_norm < 0.01:
                vel_norm = 0.01
            des_theta = np.arctan2(dy/vel_norm, dx/vel_norm)
            right_wheel = np.sin(des_theta-rot)*vel_norm + np.cos(des_theta-rot)*vel_norm
            left_wheel = -np.sin(des_theta-rot)*vel_norm + np.cos(des_theta-rot)*vel_norm
            self.set_wheel_velocity([left_wheel, right_wheel])