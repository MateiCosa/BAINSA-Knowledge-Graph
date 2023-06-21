# BAINSA-Knowledge-Graph
This repository contains the work of a Bocconi AI and Neuroscience Association team tasked with building a knowledge graph of the English Wikipedia.

<h3> Introduction </h3>

A ’knowledge graph’ is a network of entities that captures information on the
relationships between different topics. In the case of Wikipedia, the vast body
of articles spanning countless different domains makes it suitable for a good
representation of human knowledge that can yield interesting results. Our
project attempts to construct the structure of the graph and use it to build models that leverage the topology of the graph.

<h3> Contents </h3>

The notebooks and scripts containted in this repo deal with processing and cleaning data from the Wiki dumps. Our implementation of the general graph class and Dijkstra's algorithm are also included. For the modelling part, we first use a heuristic approach that uses the shorthest paths between a node and some predefined pages to create node embeddings. This embeddings are then updated iteratively by taking linear combinations of the current embedding and the embeddings of the neighboring nodes. Finally, we implement GraphSage, a graph neural network algorithm for learning inductive node representations. 

