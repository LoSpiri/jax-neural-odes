import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load data
data = jnp.load("spirals.npz")
xy_train = data["xy_train"]  # (10000, 100, 2)
xy_test = data["xy_test"]     # (10000, 100, 2)
alpha_train = data["alpha_train"]  # (10000, 1)

# Normalize data
alpha_mean = jnp.mean(alpha_train)
alpha_std = jnp.std(alpha_train)
alpha_normalized = (alpha_train - alpha_mean) / alpha_std

xy_mean = jnp.mean(xy_train, axis=(0, 1), keepdims=True)
xy_std = jnp.std(xy_train, axis=(0, 1), keepdims=True)
xy_normalized = (xy_train - xy_mean) / xy_std

# GRU Cell implementation
class GRUCell(eqx.Module):
    """GRU Cell following Algorithm 1 from the paper"""
    Wz: jnp.ndarray  # Update gate weights
    Wr: jnp.ndarray  # Reset gate weights
    Wh: jnp.ndarray  # Candidate hidden state weights
    
    def __init__(self, input_size, hidden_size, key):
        key_z, key_r, key_h = jrandom.split(key, 3)
        
        # Initialize weights (input + hidden -> gate)
        scale = 1.0 / jnp.sqrt(hidden_size)
        self.Wz = jrandom.normal(key_z, (hidden_size + input_size, hidden_size)) * scale
        self.Wr = jrandom.normal(key_r, (hidden_size + input_size, hidden_size)) * scale
        self.Wh = jrandom.normal(key_h, (hidden_size + input_size, hidden_size)) * scale
    
    def __call__(self, x, h_prev):
        """
        GRU update following Algorithm 1:
        z = σ(fz([h_prev; x]))  - Update gate
        r = σ(fr([h_prev; x]))  - Reset gate
        h' = g([r * h_prev; x]) - Candidate hidden state
        h = (1 - z) * h' + z * h_prev  - New hidden state
        """
        # Concatenate previous hidden state and input
        combined = jnp.concatenate([h_prev, x], axis=-1)
        
        # Update gate
        z = jax.nn.sigmoid(combined @ self.Wz)
        
        # Reset gate
        r = jax.nn.sigmoid(combined @ self.Wr)
        
        # Candidate hidden state (with reset applied)
        combined_reset = jnp.concatenate([r * h_prev, x], axis=-1)
        h_prime = jnp.tanh(combined_reset @ self.Wh)
        
        # Final hidden state
        h = (1 - z) * h_prime + z * h_prev
        
        return h


class GRURNN(eqx.Module):
    """GRU-RNN model for sequence prediction"""
    gru_cell: GRUCell
    output_layer: eqx.nn.Linear
    hidden_size: int
    
    def __init__(self, input_size, hidden_size, output_size, key):
        key_gru, key_out = jrandom.split(key)
        self.gru_cell = GRUCell(input_size, hidden_size, key_gru)
        self.output_layer = eqx.nn.Linear(hidden_size, output_size, key=key_out)
        self.hidden_size = hidden_size
    
    def __call__(self, xs):
        """
        Process sequence of inputs
        xs: (seq_len, input_size)
        Returns: (seq_len, output_size)
        """
        # Initialize hidden state
        h = jnp.zeros(self.hidden_size)
        
        # Process each timestep
        hiddens = []
        for x in xs:
            h = self.gru_cell(x, h)
            hiddens.append(h)
        
        # Convert to predictions
        hiddens = jnp.stack(hiddens)  # (seq_len, hidden_size)
        outputs = jax.vmap(self.output_layer)(hiddens)
        
        return outputs


# Loss function
def loss_fn(model, x, y):
    """MSE loss for sequence prediction"""
    pred = model(x)
    return jnp.mean((pred - y) ** 2)


# Training step
@eqx.filter_jit
def train_step(model, opt_state, x, y, optimizer):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


# Initialize model
key = jrandom.PRNGKey(42)
input_size = 2  # (x, y) coordinates
hidden_size = 16
output_size = 2  # predict next (x, y)

model = GRURNN(input_size, hidden_size, output_size, key)

# Setup optimizer
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

# Training loop
num_epochs = 3
batch_size = 64
num_samples = xy_normalized.shape[0]

print("Starting training...")
for epoch in range(num_epochs):
    # Shuffle training data
    key, subkey = jrandom.split(key)
    perm = jrandom.permutation(subkey, num_samples)
    
    epoch_loss = 0.0
    num_batches = 0
    
    for i in tqdm(range(0, num_samples, batch_size)):
        batch_idx = perm[i:i+batch_size]
        
        # Get batch
        batch_x = xy_normalized[batch_idx]  # (batch, 100, 2)
        
        # For each sequence in batch, use t:t+1 as input:output
        batch_loss = 0.0
        for j in range(len(batch_idx)):
            x_seq = batch_x[j, :-1]  # (99, 2) - input sequence
            y_seq = batch_x[j, 1:]   # (99, 2) - target sequence
            
            model, opt_state, step_loss = train_step(
                model, opt_state, x_seq, y_seq, optimizer
            )
            batch_loss += step_loss
        
        batch_loss /= len(batch_idx)
        epoch_loss += batch_loss
        num_batches += 1
    
    avg_loss = epoch_loss / num_batches
    
    print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

print("Training complete!")

# Test prediction
test_idx = 0
test_seq = xy_normalized[test_idx, :-1]  # (99, 2)
test_target = xy_normalized[test_idx, 1:]  # (99, 2)

predictions = model(test_seq)

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original trajectory
axes[0].plot(xy_normalized[test_idx, :, 0], 
             xy_normalized[test_idx, :, 1], 'b-', label='True', alpha=0.7)
axes[0].set_title('Original Trajectory')
axes[0].axis('equal')
axes[0].legend()

# Predicted trajectory
axes[1].plot(predictions[:, 0], predictions[:, 1], 'r-', label='Predicted', alpha=0.7)
axes[1].set_title('Predicted Trajectory')
axes[1].axis('equal')
axes[1].legend()

# Comparison
axes[2].plot(xy_normalized[test_idx, 1:, 0], 
             xy_normalized[test_idx, 1:, 1], 'b-', label='True', alpha=0.5)
axes[2].plot(predictions[:, 0], predictions[:, 1], 'r--', label='Predicted', alpha=0.5)
axes[2].set_title('Comparison')
axes[2].axis('equal')
axes[2].legend()

plt.tight_layout()
plt.show()

# Calculate test error
test_mse = jnp.mean((predictions - test_target) ** 2)
print(f"Test MSE: {test_mse:.6f}")