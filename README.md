# **Maximally Feedforward Neuron Ordering using JAX**
https://codex.flywire.ai/app/mfas_challenge

### **Overview**
This project focuses on optimizing the ordering of neurons from the **FlyWire Connectome** to maximize "feedforward-ness," providing insights into the direction of information flow in the brain. We model the connectome as a directed graph, where the neurons (nodes) are connected by weighted synaptic connections (edges). The goal of this project is to sort the neurons such that the total weight of edges pointing forward is maximized, effectively minimizing the weight of backward edges.

We utilize **JAX** for efficient automatic differentiation and high-performance GPU/TPU computing, alongside **Optax** for gradient-based optimization of neuron positions.

---

## **Project Structure**
```
├── connectome_graph.csv # Input data: The directed graph representing neuron connections - download from Codex
├── script.py # Main optimization script 
├── functions.py # Utility functions for optimization and evaluation 
├── README.md # Project documentation 
├── requirements.txt # Python dependencies 
└── ordered_nodes_0.csv # Example output: optimized neuron ordering
```

## **Usage**

Once installed, you can run the optimization using the following command:

python script.py <run_idx>

## **Optimization Details**

Each neuron is assigned a scalar position, initialized randomly.

The loss function is the negative feedforward edge weight, penalizing backward edges

Optimizer: Adam optimizer from Optax with gradient clipping to handle potential exploding gradients.

Neuron positions are updated iteratively by minimizing the loss using JAX's automatic differentiation.

The model evaluates the percentage of feedforward edge weight after each update.

## **Performance Evaluation**
The performance of the model is evaluated by calculating the percentage of forward edge weight (i.e., the proportion of edges that point forward in the optimized neuron ordering).

Metrics:
Total Feedforward Edge Weight: The sum of weights for all edges where Source Position < Target Position.

Percentage of Forward Edge Weight: The forward edge weight as a percentage of the total edge weight.
