import jax
import jax.numpy as jnp
import optax


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


def calculate_metric(
    positions, num_nodes, source_indices, target_indices, edge_weights
):
    # Get final positions
    final_positions = positions

    # Sort node indices based on positions
    sorted_indices = jnp.argsort(final_positions)
    node_order = jnp.zeros(num_nodes)
    node_order = node_order.at[sorted_indices].set(jnp.arange(num_nodes))

    source_order = node_order[source_indices]
    target_order = node_order[target_indices]
    calculate_node_forward(source_order, target_order, edge_weights)


# @jax.jit
def objective_function(positions, source_indices, target_indices, edge_weights):
    # Get positions of source and target nodes
    pos_source = positions[source_indices]
    pos_target = positions[target_indices]

    delta = pos_target - pos_source
    beta = 5.0  # Adjusted beta for stability
    sigmoid = jax.nn.sigmoid(beta * delta)

    # Compute the weighted sum
    total_forward_weight = jnp.sum(edge_weights * sigmoid)

    # Add L2 regularization term
    regularization_strength = 0.0001
    regularization = regularization_strength * jnp.sum(positions**2)

    return -total_forward_weight + regularization
