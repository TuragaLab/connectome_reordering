import jax
import jax.numpy as jnp
import optax
from jax import random, jit


def normalize_positions(positions):
    # Normalize positions to have zero mean and unit variance
    mean = jnp.mean(positions)
    std = jnp.std(positions) + 1e-8  # Avoid division by zero
    positions = (positions - mean) / std
    return positions


def calculate_node_forward(source_orders, target_orders, edge_weights):
    forward_edges = source_orders < target_orders

    # Calculate total forward edge weight
    forward_edge_weight = jnp.sum(edge_weights * forward_edges)

    # Calculate total edge weight (for normalization)
    total_edge_weight = jnp.sum(edge_weights)

    # Calculate percentage of forward edge weight
    percentage_forward = 100 * (forward_edge_weight / total_edge_weight)

    print(f"Percentage of forward edge weight: {percentage_forward:.2f}%")
    return percentage_forward


def calculate_metric(
    positions, w, num_nodes, source_indices, target_indices, edge_weights
):
    # Get final positions
    final_positions = jnp.dot(positions, w)

    # Sort node indices based on positions
    sorted_indices = jnp.argsort(final_positions)
    node_order = jnp.zeros(num_nodes)
    node_order = node_order.at[sorted_indices].set(jnp.arange(num_nodes))

    source_order = node_order[source_indices]
    target_order = node_order[target_indices]
    return calculate_node_forward(source_order, target_order, edge_weights)


@jax.jit
def objective_function(positions, w, source_indices, target_indices, edge_weights):
    # Project each neuron embedding onto the learnable direction w
    projections = jnp.dot(
        positions, w
    )  # positions: (num_nodes, embedding_dim), w: (embedding_dim,)

    # Get the scalar projections for the source and target neurons
    proj_source = projections[source_indices]
    proj_target = projections[target_indices]

    # distances = jnp.linalg.norm(proj_target - proj_source, axis=-1)
    # beta = 5.0 / (1.0 + distances)
    # Use the difference between source and target projections
    delta = proj_target - proj_source

    # Sigmoid to encourage proj_source < proj_target
    beta = 1
    sigmoid = jax.nn.hard_tanh(beta * delta)

    # Compute the total forward weight
    total_forward_weight = jnp.sum(edge_weights * sigmoid)

    return -total_forward_weight


# Function to compute total forward edge weight given an ordering
def compute_total_forward_weight(
    ordering, source_indices, target_indices, edge_weights_normalized
):
    node_ranks = jnp.zeros_like(ordering)
    node_ranks = node_ranks.at[ordering].set(jnp.arange(len(ordering)))
    edge_directions = node_ranks[target_indices] - node_ranks[source_indices]
    forward_edges = edge_directions > 0
    total_forward_weight = jnp.sum(edge_weights_normalized * forward_edges)
    return total_forward_weight


