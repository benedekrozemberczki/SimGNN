import networkx as nx
import random
import json
from tqdm import tqdm

def transfomer(graph_1, graph_2, vals, index):
    graph_1.remove_nodes_from(nx.isolates(graph_1))
    graph_2.remove_nodes_from(nx.isolates(graph_2)) 
    edges_1 = [[edge[0], edge[1]] for edge in graph_1.edges()]
    nodes_2 = graph_2.nodes()
    random.shuffle(nodes_2)
    mapper = {node:i for i, node in enumerate(nodes_2)}
    edges_2 = [[mapper[edge[0]], mapper[edge[1]]] for edge in graph_2.edges()]

    graph_1 = nx.from_edgelist(edges_1)
    graph_2 = nx.from_edgelist(edges_2)
    graph_1.remove_nodes_from(nx.isolates(graph_1))
    graph_2.remove_nodes_from(nx.isolates(graph_2))
    edges_1 = [[edge[0], edge[1]] for edge in graph_1.edges()]
    edges_2 = [[edge[0], edge[1]] for edge in graph_2.edges()]
    data = dict()
    data["graph_1"] = edges_1
    data["graph_2"] = edges_2
    data["labels_1"] = [str(graph_1.degree(node)) for node in graph_1]
    data["labels_2"] = [str(graph_2.degree(node))  for node in graph_2]
    data["ged"] = vals

    if len(data["labels_1"]) == len(nx.nodes(graph_1)) and len(data["labels_2"]) == len(nx.nodes(graph_2)) :
        if len(nx.nodes(graph_1)) == max(graph_1.nodes())+1 and len(nx.nodes(graph_2)) == max(graph_2.nodes())+1:
            with open("./test/"+str(index)+".json","w") as f:
                json.dump(data,f)
            z = index + 1
    else:
        z = index
    return z


 



index = 0
while index <1000:
    print(index)
    graph = nx.erdos_renyi_graph(int(random.uniform(10,20)),random.uniform(0.4,0.7))
    if nx.is_connected(graph):
        nodes = graph.nodes()
        clone = nx.from_edgelist(graph.edges())
        counter = 0
        vals = int(abs(random.uniform(5,35)))
        while counter < vals:
            x = random.uniform(0,1)

            if x>0.5:
                
                node_1 ,node_2 = random.choice(clone.edges())
                counter = counter + 1
                if graph.has_edge(node_1,node_2):
                    clone.remove_edge(node_1,node_2)
            else:
                node_1 = random.choice(clone.nodes())
                node_2 = random.choice(clone.nodes())
                if node_1!=node_2 and not clone.has_edge(node_1,node_2) and not graph.has_edge(node_1,node_2):
                    clone.add_edge(node_1,node_2)
                    counter = counter + 1
    try:
        clone.remove_nodes_from(nx.isolates(clone))
        graph.remove_nodes_from(nx.isolates(graph)) 
        if nx.is_connected(clone) and nx.is_connected(graph):
            index = transfomer(graph, clone, vals, index)
    except:
        pass
                    


