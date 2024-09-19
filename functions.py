import jax
import jax.numpy as jnp


def normalize_positions(positions):
    # Normalize positions to have zero mean and unit variance
    mean = jnp.mean(positions)
    std = jnp.std(positions) + 1e-8  # Add epsilon to avoid division by zero
    positions = (positions - mean) / std
    return positions


def calculate_metric(
    positions, num_nodes, source_indices, target_indices, edge_weights
):
    # Get final positions
    final_positions = positions

    # Sort node indices based on positions
    sorted_indices = jnp.argsort(final_positions)

    # Map back to node IDs
    # ordered_node_ids = [index_to_node_id[int(idx)] for idx in sorted_indices]

    # Create a mapping from node index to order in the final sequence
    node_order = jnp.zeros(num_nodes)
    node_order = node_order.at[sorted_indices].set(jnp.arange(num_nodes))

    # Compute the direction of each edge in the final ordering
    edge_directions = node_order[target_indices] - node_order[source_indices]

    # Edges pointing forward have positive edge_directions
    forward_edges = edge_directions > 0

    # Compute the total forward edge weight
    total_forward_weight = jnp.sum(edge_weights * forward_edges)
    total_edge_weight = jnp.sum(edge_weights)

    print(f"Total Forward Edge Weight: {total_forward_weight}")
    print(
        f"Percentage of Forward Edge Weight: {100 * float(total_forward_weight) / float(total_edge_weight):.2f}%"
    )


@jax.jit
def objective_function(positions, source_indices, target_indices, edge_weights, epoch):
    # Get positions of source and target nodes
    pos_source = positions[source_indices]
    pos_target = positions[target_indices]

    delta = pos_target - pos_source
    delta /= jnp.std(delta)
    beta = 2.0
    sigmoid = jax.nn.sigmoid(beta * delta)
    # Compute the weighted sum
    total_forward_weight = jnp.sum(edge_weights * sigmoid)

    return -total_forward_weight


def safe_sigmoid(x):
    return jnp.where(x >= 0, 1 / (1 + jnp.exp(-x)), jnp.exp(x) / (1 + jnp.exp(x)))
