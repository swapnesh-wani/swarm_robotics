import numpy as np

def get_constraints(edges,pos_vect):
    ''' In this function constraints are calculated as per the edges passed
    '''
    # Initialize empty contraints
    constraints = []
    pos_vect = np.reshape(pos_vect,(5,2))
    for edge in edges:
        constraints.append(np.sum(np.square(np.subtract(pos_vect[edge[0]],pos_vect[edge[1]]))))    

    return constraints

    
   
if __name__ == "__main__":
    # Define graph 
    # E = [[0,3],[0,4],[1,2],[1,4],[2,4],[3,4]]
    E = [[0,1],[0,3],[0,4],[1,2],[1,4],[2,4],[3,4]]
    
    # Postion vector stack with each p(i) = [x,y] i.e. x and y coordinate of point p(i)
    pos_vect = [0,2,1,1,0,0,-1,1,0,1]  
    # pos_vect = [1.42621215,1.54369821,1.43217611,1.63151863,2.53752869,2.51367263,2.44970826,2.51963658,2.42592939,1.51991934] 
    constraints = get_constraints(E,pos_vect)
    print(constraints)
    
    # print(np.sum(np.square(np.subtract(pos_vect[0],pos_vect[3]))))