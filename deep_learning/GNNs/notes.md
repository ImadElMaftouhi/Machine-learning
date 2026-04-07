# Graph Neural Networks




# Graph Neural Networks

Neural networks have been adapted to leverage the structure and properties of graphs. We explore the components needed to build a graph neural network and motivate the design choices behind them.

A graph represents the relationships (edges) between a collection of entities (nodes). Scalars or embeddings can be stored at nodes and edges. Graphs can also be specialized by edge directionality:

- Directed graphs have edges with a source node and a destination node, so information flows from source to destination.
- Undirected graphs have edges without an assigned direction, so information can flow in both directions.

A single undirected edge is equivalent to two directed edges in opposite directions.

You are probably already familiar with some types of graph data, such as social networks. Graphs are an extremely general representation of data, and we can also model images and text as graphs.

## Images as graphs

Images can be represented as graphs where pixels are nodes and edges connect adjacent pixels. Each node stores color information (RGB values or embeddings). This representation enables graph neural networks to process images by treating spatial relationships as graph structure.

## Text as graphs

Text can be represented as a graph by treating each character, word, or token as a node and connecting each item to the one that follows it. This yields a simple directed graph in which the edges capture sequence order.

---

In practice, this is not usually how text and images are encoded. These graph representations are redundant because images and text already have very regular structure. For example, images produce a banded adjacency matrix because each pixel connects to nearby pixels in a grid. Text produces an adjacency matrix that looks like a diagonal line because each token only connects to its predecessor and successor.

## Graph-valued data in the wild

Graphs are a useful way to describe data with heterogeneous structure. In these examples, the number of neighbors per node can vary, unlike the fixed neighborhood size of images and text. This kind of data is hard to express in any other format.

### Molecules as graphs

Molecules can be represented as graphs where atoms are nodes and chemical bonds are edges. Each node can store information about the atom (for example, element type or charge), and each edge can encode bond type (single, double, etc.). This graph structure allows GNNs to predict molecular properties such as solubility or reactivity by learning from atomic interactions.

### Social networks as graphs

Social networks can be represented by modeling individuals as nodes and relationships as edges. Graphs make it easy to study patterns in collective behavior among people, organizations, and institutions.

### Citation networks as graphs

Citation networks represent papers or patents as nodes and citations as directed edges from citing documents to cited documents. Node features can include metadata such as publication year, venue, and topic area, while edges capture influence and information flow. GNNs on citation graphs can learn to classify papers, recommend related work, or predict emerging research trends by aggregating information from referenced and citing neighbors.

## Problems that have graph-structured data

Graph-structured prediction tasks are usually grouped into three levels:

- Graph-level tasks
    - The model takes an entire graph as one example and predicts a single output for the graph.
    - Examples: molecular property prediction, graph classification, whole-network anomaly scoring.
    - In practice, the GNN computes node representations and then pools them into a global graph representation before the final prediction.
    > This is analogous to image classification problems with MNIST and CIFAR, where a label is assigned to an entire image. With text, a similar problem is sentiment analysis, where the goal is to identify the mood of an entire sentence.

- Node-level tasks
    - The model predicts labels or scores for individual nodes inside a graph.
    - Examples: classifying papers in a citation network, predicting a user’s role in a social graph, labeling proteins in a biological network.
    - Node-level prediction depends on both local features and information aggregated from neighbors, which is the core strength of GNNs.

- Edge-level tasks
    - The model predicts the existence, strength, or type of relationships between pairs of nodes.
    - Examples: link prediction in recommender systems, edge classification in knowledge graphs, predicting new chemical bonds.
    - Edge-level problems often compare or combine representations of two nodes and their shared context.

These three levels capture the main ways graph data is used in practice. The same underlying message-passing mechanism can be adapted to each task by changing the readout: graph readout for graph-level problems, node readout for node-level problems, and pairwise or edge readout for edge-level problems.

These problems can be solved with a single model class, the GNN.

## Using graphs in machine learning

The first step in solving different graph tasks is to decide how to represent graphs for compatibility with neural networks. Machine learning models typically take rectangular arrays as input.

Graphs can contain up to four kinds of information: node features, edge features, global context, and connectivity. The first three are straightforward: for example, node features can be stored in a node feature matrix by assigning each node an index and storing its features in the corresponding row. While these matrices may have a variable number of rows, they can still be processed by standard neural network components.

## Representing graph connectivity

Graph connectivity can be expressed in several ways, but each representation has trade-offs.

### Adjacency matrix

- Stores connectivity as a square matrix with one row and one column per node.
- Entry `(i, j)` indicates whether there is an edge from node `i` to node `j`.

Difficulties:
- Memory cost grows quadratically with the number of nodes, even for sparse graphs.
- Sparse graphs waste most of the matrix entries, so the representation becomes inefficient.
- Fixed size makes batching graphs of different sizes awkward.
- Permuting node indices changes the matrix, even though the graph is the same, so the representation is not naturally permutation invariant.
- Another problem is that there are many adjacency matrices that can encode the same connectivity and there is no guarantee that these different matrices would produce the same result in a deep neural network.

### Edge lists

An edge list stores connectivity as a collection of edge pairs, where each pair lists the source and destination node for one edge.

Difficulties:
- Edge lists require explicit ordering of edges, which can introduce arbitrary structure and make it harder to preserve permutation invariance.

### Adjacency lists

An adjacency list stores, for each node, the list of its neighbors. This is a compact way to represent sparse graphs when node degrees vary.

Difficulties:
- Adjacency lists and neighbor sets vary in length per node, complicating tensor-based batching and parallel processing.

### Sparse tensor formats

Sparse tensor formats represent graph connectivity using index tensors and value tensors, which are efficient for large sparse graphs.

Difficulties:
- Sparse formats must manage index tensors and values, which adds implementation complexity.
- Some graph frameworks require both node and edge indexing schemes, so converting between representations can be error-prone.

Overall, the choice of connectivity representation balances memory efficiency, ease of batching, and support for permutation-invariant graph operations.

## Graph Neural Networks

A GNN is an optimizable transformation on all attributes of the graph (nodes, edges, global-context) that preserves graph symmetries (permutation invariances). 

The simplest GNN architecture is one where we learn new embeddings for all graph attributes (nodes, edges, global) but where we do not yet use the connectivity of the graph.

This GNN uses a separate multilayer perceptron (MLP) on each component of a graph; called a GNN layer.

### Message Passing in GNNs

To incorporate graph connectivity, GNNs use a message-passing mechanism. In each layer, nodes aggregate information from their neighbors via edges. This process updates node representations by combining local features with aggregated neighbor information.

A basic message-passing step can be formalized as:

1. **Message Computation**: For each edge, compute a message based on the source node and edge features.
2. **Aggregation**: For each node, aggregate messages from its incoming edges (e.g., sum, mean, or max).
3. **Update**: Update the node's representation using the aggregated messages and its current features, often via an MLP.

This allows GNNs to capture local graph structure and propagate information across the graph.

### Variants of GNNs

Several popular GNN variants build on this foundation:

- **Graph Convolutional Networks (GCNs)**: Use spectral graph convolutions, aggregating neighbor features with normalized adjacency.
- **GraphSAGE**: Samples neighbors for scalability, using mean or LSTM aggregation.
- **GAT (Graph Attention Networks)**: Applies attention mechanisms to weigh neighbor importance.
- **GIN (Graph Isomorphism Networks)**: Designed to be as powerful as the Weisfeiler-Lehman graph isomorphism test.

### Training GNNs

GNNs are trained end-to-end using backpropagation, similar to other neural networks. Loss functions depend on the task: cross-entropy for classification, mean squared error for regression. Challenges include overfitting on small graphs and handling variable sizes.

### Applications and Libraries

GNNs excel in domains like drug discovery, social network analysis, and recommendation systems. Popular libraries include PyTorch Geometric, DGL, and TensorFlow GNN, providing efficient implementations for message passing and graph operations.