A graph represent the relations (edges) between a collection of entities (nodes).
Information in the form of scalars or embeddings can be stored at each graph node or edge.
Graphs can also be specialized by edge directionality:
- Directed graphs have edges with a source node and a destination node, so information flows from the source to the destination.
- Undirected graphs have edges without a source or destination, so information can flow in both directions.
A single undirected edge is equivalent to two directed edges in opposite directions.


You’re probably already familiar with some types of graph data, such as social networks. However, graphs are an extremely powerful and general representation of data, we can also discuss how images and textes can be modeled as graphs 

## images as graphs 

Images can be represented as graphs where pixels are nodes and edges connect adjacent pixels. Each node stores the pixel's color information (RGB values or embeddings). This graph representation enables operations like convolution and enables graph neural networks to process images by treating spatial relationships as graph structure.