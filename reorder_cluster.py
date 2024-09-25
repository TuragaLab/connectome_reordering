import jax
import jax.numpy as jnp
from jax import random, jit
import optax
import polars as pl
import numpy as np
import functions

# Enable 64-bit precision and debug options
jax.config.update("jax_enable_x64", True)

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
    key = random.PRNGKey(int(run_idx)+52345235)
    embedding_dim = 15  # Adjust the embedding dimensionality
    positions = random.uniform(
        key, shape=(num_nodes, embedding_dim), minval=-1, maxval=1
    )
    key, subkey = random.split(key)
    w = random.uniform(subkey, shape=(embedding_dim,))

    # Define the optimizer with gradient clipping)
#    positions = jnp.load("./checkpoints/positions_82.63022402029044_4.npy")
#    w = jnp.load("./checkpoints/weights_82.63022402029044_4.npy")

    num_epochs = 100000
    beta = jnp.logspace(-1, 1, num=num_epochs)
    exponential_decay_scheduler = optax.exponential_decay(init_value=0.0005, transition_steps=num_epochs,
                                                      decay_rate=0.995, transition_begin=int(num_epochs*0.5),
                                                      staircase=False)
    #exponential_decay_scheduler = optax.cosine_decay_schedule(init_value=0.0005, decay_steps=num_epochs)

    optimizer = optax.adam(learning_rate=exponential_decay_scheduler)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=exponential_decay_scheduler)
    )
    # Initialize optimizer state
    opt_state = optimizer.init((positions, w))

    best_metric = 0

    @jax.jit
    def optimization_step(
        positions, w,beta, opt_state, source_indices, target_indices, edge_weights
    ):
        # Compute loss and gradients for both positions and w
        loss, grads = jax.value_and_grad(functions.objective_function, argnums=(0, 1))(
            positions, w, beta, source_indices, target_indices, edge_weights
        )

        # Update positions and w
        updates, opt_state = optimizer.update(grads, opt_state)
        positions, w = optax.apply_updates((positions, w), updates)

        return positions, w, opt_state, loss
    for epoch in range(num_epochs):
        positions, w, opt_state, loss = optimization_step(
            positions, w, beta[epoch], opt_state, source_indices, target_indices, edge_weights
        )

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
    ordered_nodes_df = pd.DataFrame({"Node ID": ordered_node_ids, "Order": jnp.arange(num_nodes)})
    ordered_nodes_df.to_csv(f"./checkpoints/ordered_nodes_{best_metric}_{run_idx}.csv", index=False)
    # Save weights
    jnp.save(f'./checkpoints/weights_{best_metric}_{run_idx}.npy', w)
    jnp.save(f'./checkpoints/positions_{best_metric}_{run_idx}.npy', positions)

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python script.py <run_idx>")
        sys.exit(1)

    run_idx = sys.argv[1]
    print("STARTING RUN", run_idx)
    run(run_idx)
