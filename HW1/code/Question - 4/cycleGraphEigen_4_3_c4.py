import numpy as np
import  getLaplacian as gl   # user created
import  cycleGraph  as cg    # user created


def main():
    n_vertices = 4
    edges = cg.cycle_graph(n_vertices)

    laplacian = gl.get_laplacian(edges,n_vertices,isDirected = False)

    print("Laplacian Matrix: \n",laplacian)

    eigenVals = np.linalg.eigvals(laplacian)
    eigenVals = np.sort(eigenVals)

    print("Two smallest eigen values are = ",eigenVals[:2])
    print("Two largest eigen values are = ",eigenVals[::-1][:2])

if __name__ == '__main__':
    main()


