# Graph Neural Networks




Neural networks have been adapted to leverage the structure and properties of graphs. We explore the components needed for building a graph neural network - and motivate the design choices behind them.


A graph represent the relations (edges) between a collection of entities (nodes).
Information in the form of scalars or embeddings can be stored at each graph node or edge.
Graphs can also be specialized by edge directionality:
- Directed graphs have edges with a source node and a destination node, so information flows from the source to the destination.
- Undirected graphs have edges without a source or destination, so information can flow in both directions.
A single undirected edge is equivalent to two directed edges in opposite directions.


You’re probably already familiar with some types of graph data, such as social networks. However, graphs are an extremely powerful and general representation of data, we can also discuss how images and textes can be modeled as graphs 

## images as graphs 

Images can be represented as graphs where pixels are nodes and edges connect adjacent pixels. Each node stores the pixel's color information (RGB values or embeddings). This graph representation enables operations like convolution and enables graph neural networks to process images by treating spatial relationships as graph structure.

## text as graphs

We can digitize text by associating indices to each character, word, or token, and representing text as a sequence of these indices. This creates a simple directed graph, where each character or index is a node and is connected via an edge to the node that follows it.

---

Of course, in practice, this is not usually how text and images are encoded: these graph representations are redundant since all images and all text will have very regular structures. For instance, images have a banded structure in their adjacency matrix because all nodes (pixels) are connected in a grid. The adjacency matrix for text is just a diagonal line, because each word only connects to the prior word, and to the next one.

## Graph-valued data in the wild
Graphs are a useful tool to describe data you might already be familiar with. Let’s move on to data which is more heterogeneously structured. In these examples, the number of neighbors to each node is variable (as opposed to the fixed neighborhood size of images and text). This data is hard to phrase in any other way besides a graph.


### Molecules as graphs.
Molecules can be represented as graphs where atoms are nodes and chemical bonds are edges. Each node stores information about the atom (e.g., element type, charge), and edges represent the type of bond (single, double, etc.). This graph structure allows graph neural networks to predict molecular properties like solubility or reactivity by learning from the atomic interactions.

### Social networks as graphs
Social networks are tools to study patterns in collective behaviour of people, institutions and organizations. We can build a graph representing groups of people by modelling individuals as nodes, and their relationships as edges.

### citation networks as graphs
Citation networks represent papers or patents as nodes and citation links as directed edges from citing documents to cited documents. Node features can include metadata such as publication year, venue, and topic area, while edges capture influence and information flow. Graph neural networks on citation graphs can learn to classify papers, recommend related work, or predict emerging research trends by aggregating information from referenced and citing neighbors.

## Problems that have graph structured data
Graph-structured prediction tasks are usually grouped into three levels:

- Graph-level tasks
    - The model takes an entire graph as one example and predicts a single output for the graph.
    - Examples: molecular property prediction, graph classification, whole-network anomaly scoring.
    - In practice, the GNN computes node representations and then pools them into a global graph representation before the final prediction.

- Node-level tasks
    - The model predicts labels or scores for individual nodes inside a graph.
    - Examples: classifying papers in a citation network, predicting the role of a user in a social graph, labeling proteins in a biological network.
    - Node-level prediction depends on both local features and information aggregated from neighbors, which is the core strength of GNNs.

- Edge-level tasks
    - The model predicts the existence, strength, or type of relationships between pairs of nodes.
    - Examples: link prediction in recommender systems, edge classification in knowledge graphs, predicting new chemical bonds.
    - Edge-level problems often require comparing or combining representations of two nodes and their shared context.

These three levels capture the main ways graph data is used in practice. The same underlying message-passing mechanism can be adapted to each task by changing the readout: a graph readout for graph-level problems, a node readout for node-level problems, and a pairwise or edge readout for edge-level problems.
