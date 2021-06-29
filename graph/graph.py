class Graph(object):
    """Base class for Graph.
    This class defines the structure or graph of the system.
    In this case to graph structure our quantum date and give a description on how it is to carry out its
    quantum machine learning using the quantum neural netowrks"""
    def __init__(self):
        self.adj_list = None

    def set_adj_list(self, adj_list):
        self.adj_list = adj_list

