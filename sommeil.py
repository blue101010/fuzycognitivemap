import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Definition des concepts
concepts = ['Stress', 'Sommeil', 'Cafe', 'Productivite']

# Matrice de poids (lignes=cibles, colonnes=sources)
W = np.array([
    [0,    0,     0,    0],      # Stress (pas d'entrees)
    [-0.7, 0,    -0.6,  0],      # Sommeil
    [0.4,  0,     0,    0],      # Cafe
    [0,    0.8,   0.5,  0]       # Productivite
])

# Creation du graphe dirige
G = nx.DiGraph()
for i, target in enumerate(concepts):
    G.add_node(target)
    for j, source in enumerate(concepts):
        if W[i,j] != 0:
            G.add_edge(source, target, weight=W[i,j])

# Visualisation
pos = nx.spring_layout(G, seed=42)
edges = G.edges(data=True)
colors = ['green' if d['weight']>0 else 'red' for _,_,d in edges]

plt.figure(figsize=(10, 8))
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2000)
nx.draw_networkx_labels(G, pos, font_size=10)
nx.draw_networkx_edges(G, pos, edge_color=colors, arrows=True)
edge_labels = {(u,v): f"{d['weight']:.1f}" for u,v,d in edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels)
plt.title("FCM: Facteurs de productivite")
plt.axis('off')
plt.show()