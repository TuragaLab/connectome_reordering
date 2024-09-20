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
    positions = random.uniform(key, shape=(num_nodes,), minval=-0.1, maxval=0.1)

    # Define the optimizer with gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(learning_rate=0.001)
    )

    # Initialize optimizer state
    opt_state = optimizer.init(positions)

    num_epochs = 10000
    best_metric = 0
    for epoch in range(num_epochs):
        positions, opt_state, loss = functions.optimization_step(
            positions,
            opt_state,
            source_indices,
            target_indices,
            edge_weights,
            epoch,
            optimizer,
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
                positions, num_nodes, source_indices, target_indices, edge_weights
            )
            if metric > best_metric:
                best_metric = metric
                print(f"New best metric: {best_metric:.2f}")

    # Map back to original node IDs and save the ordering
    sorted_indices = jnp.argsort(positions)
    ordered_node_ids = [index_to_node_id[int(idx)] for idx in sorted_indices]

    # Save the ordering to a CSV file
    import pandas as pd

    ordered_nodes_df = pd.DataFrame({"Node ID": ordered_node_ids})
    ordered_nodes_df.to_csv(f"ordered_nodes_{best_metric}_{run_idx}.csv", index=False)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python script.py <run_idx>")
        sys.exit(1)

    run_idx = sys.argv[1]
    print("STARTING RUN", run_idx)
    run(run_idx)
