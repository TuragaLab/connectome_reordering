import polars as pl
import numpy as np
import functions
import jax.numpy as jnp
import jax
from jax import random
import optax


def run(run_idx):
    # Load the data
    df = pl.read_csv("./connectome_graph.csv")

    # Extract arrays
    source_nodes = df[df.columns[0]].to_numpy().astype(np.int64)
    target_nodes = df[df.columns[1]].to_numpy().astype(np.int64)
    edge_weights = (
        df[df.columns[2]].to_numpy().astype(np.int64)
    )  # Compute the direction of each edge in the initial ordering

    # Get unique node IDs and map to indices
    unique_nodes = np.unique(np.concatenate((source_nodes, target_nodes)))
    node_id_to_index = {node_id: idx for idx, node_id in enumerate(unique_nodes)}
    index_to_node_id = {idx: node_id for node_id, idx in node_id_to_index.items()}

    # Map node IDs to indices in edge lists
    source_indices = np.array([node_id_to_index[node_id] for node_id in source_nodes])
    target_indices = np.array([node_id_to_index[node_id] for node_id in target_nodes])

    # Convert to JAX arrays
    source_indices = jnp.array(source_indices)
    target_indices = jnp.array(target_indices)
    edge_weights = jnp.array(edge_weights)

    # Compute maximum edge weight
    total_edge_weight = jnp.sum(edge_weights)

    # Normalize edge weights
    edge_weights = edge_weights / total_edge_weight

    num_nodes = len(unique_nodes)
    key = random.PRNGKey(0)
    positions = random.uniform(key, shape=(num_nodes,))

    sorted_indices = jnp.argsort(positions)

    # Create a mapping from node index to order in the sequence
    node_order = jnp.zeros(num_nodes, dtype=int)
    node_order = node_order.at[sorted_indices].set(jnp.arange(num_nodes))

    edge_directions = node_order[target_indices] - node_order[source_indices]

    forward_edges = edge_directions > 0

    total_forward_weight_initial = jnp.sum(edge_weights * forward_edges)

    total_edge_weight = jnp.sum(edge_weights)
    original_total_edge_weights = total_edge_weight
    # Compute the percentage of forward edge weight
    percentage_forward_initial = (
        100 * float(total_forward_weight_initial) / (total_edge_weight)
    )

    # Create the gradient function
    objective_grad = jax.grad(functions.objective_function)

    # Define the optimizer
    optimizer = optax.adam(learning_rate=0.005)

    # Initialize optimizer state
    opt_state = optimizer.init(positions)

    # Print the results
    print(f"Total Forward Edge Weight (Initial): {total_forward_weight_initial}")
    print(
        f"Percentage of Forward Edge Weight (Initial): {percentage_forward_initial:.2f}%"
    )

    num_epochs = 10000

    for epoch in range(num_epochs):
        # Compute gradients
        loss, grads = jax.value_and_grad(functions.objective_function)(
            positions, source_indices, target_indices, edge_weights, epoch
        )

        # Update positions
        updates, opt_state = optimizer.update(grads, opt_state)
        positions = optax.apply_updates(positions, updates)
        positions = functions.normalize_positions(positions)
        # Optional: Print progress every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {-loss}")
            functions.calculate_metric(
                positions, num_nodes, source_indices, target_indices, edge_weights
            )
    jnp.save(f"positions_{run_idx}.npy", positions)


if __name__ == "__main__":
    import sys

    # read kwargs from json file
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)

    itr = sys.argv[1]
    print("STARTING RUN", itr)
    run(itr)
