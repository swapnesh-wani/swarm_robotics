def cycle_graph(num_vertices):
    edges = []
    for i in range(0,num_vertices):
        edge = [i%num_vertices,(i+1)%num_vertices]   # for cyclic order
        edges.append(edge)

    return edges

if __name__ == '__main__':
    n_vertices = 4
    edges = cycle_graph(n_vertices)
    print(edges)