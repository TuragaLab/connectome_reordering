import jax
import jax.numpy as jnp
from jax import random, jit
import optax
import polars as pl
import numpy as np
import functions

# Enable 64-bit precision and debug options
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)


def run(run_idx):
    # Load the data
    df = pl.read_csv("./connectome_graph.csv")

    # Extract arrays with appropriate data types
    source_nodes = df[df.columns[0]].to_numpy().astype(np.int64)
    target_nodes = df[df.columns[1]].to_numpy().astype(np.int64)
    edge_weights = df[df.columns[2]].to_numpy().astype(np.float64)

    # Get unique node IDs and map to indices
    unique_nodes = np.unique(np.concatenate((source_nodes, target_nodes)))
    node_id_to_index = {node_id: idx for idx, node_id in enumerate(unique_nodes)}
    index_to_node_id = {idx: node_id for node_id, idx in node_id_to_index.items()}
    source_indices = jnp.array([node_id_to_index[node_id] for node_id in source_nodes])
    target_indices = jnp.array([node_id_to_index[node_id] for node_id in target_nodes])
    edge_weights = jnp.array(edge_weights)

    # Compute total edge weight
    max_edge_weight = jnp.max(edge_weights)

    # Normalize edge weights
    edge_weights = edge_weights / max_edge_weight

    num_nodes = len(unique_nodes)
    key = random.PRNGKey(int(run_idx))
    embedding_dim = 5  # Adjust the embedding dimensionality
    positions = random.uniform(
        key, shape=(num_nodes, embedding_dim), minval=-0.1, maxval=0.1
    )
    key, subkey = random.split(key)
    w = random.uniform(subkey, shape=(embedding_dim,))
    # Define the optimizer with gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(learning_rate=0.001)
    )

    # Initialize optimizer state
    opt_state = optimizer.init((positions, w))

    num_epochs = 10000
    best_metric = 0

    @jax.jit
    def optimization_step(
        positions, w, opt_state, source_indices, target_indices, edge_weights
    ):
        # Compute loss and gradients for both positions and w
        loss, grads = jax.value_and_grad(functions.objective_function, argnums=(0, 1))(
            positions, w, source_indices, target_indices, edge_weights
        )

        # Update positions and w
        updates, opt_state = optimizer.update(grads, opt_state)
        positions, w = optax.apply_updates((positions, w), updates)

        return positions, w, opt_state, loss

    for epoch in range(num_epochs):
        positions, w, opt_state, loss = optimization_step(
            positions, w, opt_state, source_indices, target_indices, edge_weights
        )

        if jnp.isnan(loss) or jnp.isinf(loss):
            print("NaN or Inf detected in loss.")
            break
        if jnp.any(jnp.isnan(positions)) or jnp.any(jnp.isinf(positions)):
            print("NaN or Inf detected in positions.")
            break

        # Optional: Print progress every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {-loss}")
            metric = functions.calculate_metric(
                positions, w, num_nodes, source_indices, target_indices, edge_weights
            )
            if metric > best_metric:
                best_metric = metric
                print(f"New best metric: {best_metric:.2f}")

    # Map back to original node IDs and save the ordering
    sorted_indices = jnp.argsort(np.dot(positions, w))
    ordered_node_ids = [index_to_node_id[int(idx)] for idx in sorted_indices]

    # Save the ordering to a CSV file
    import pandas as pd

    ordered_nodes_df = pd.DataFrame({"Node ID": ordered_node_ids})
    ordered_nodes_df.to_csv(f"ordered_nodes_{best_metric}_{run_idx}.csv", index=False)

    best_ordering, best_weight = functions.simulated_annealing(
        positions,
        w,
        source_indices,
        target_indices,
        edge_weights,
        initial_temp=1.0,
        final_temp=0.001,
        max_iter=100000,
    )
    node_order = jnp.zeros(num_nodes)
    node_order = node_order.at[best_ordering].set(jnp.arange(num_nodes))
    source_order = node_order[source_indices]
    target_order = node_order[target_indices]
    print("Annealing: ")
    functions.calculate_node_forward(source_order, target_order, edge_weights)

    ordered_node_ids = [index_to_node_id[int(idx)] for idx in best_ordering]

    ordered_nodes_df = pd.DataFrame({"Node ID": ordered_node_ids})
    ordered_nodes_df.to_csv(
        f"annealing_ordered_nodes_{best_metric}_{run_idx}.csv", index=False
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python script.py <run_idx>")
        sys.exit(1)

    run_idx = sys.argv[1]
    print("STARTING RUN", run_idx)
    run(run_idx)
