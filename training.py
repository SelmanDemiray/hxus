import numpy as np
import time
import os
from neural_network import EncoderDecoderNN

def train_model(encoder_inputs, decoder_inputs, decoder_targets, 
                model, epochs=10, batch_size=32, learning_rate=0.01,
                save_dir='models', save_every=5):
    """
    Train the encoder-decoder model
    
    Parameters:
    -----------
    encoder_inputs : numpy array
        Encoder input sequences [num_samples, seq_len]
    decoder_inputs : numpy array
        Decoder input sequences [num_samples, seq_len]
    decoder_targets : numpy array
        Decoder target sequences [num_samples, seq_len]
    model : EncoderDecoderNN
        The neural network model
    epochs : int
        Number of training epochs
    batch_size : int
        Size of each training batch
    learning_rate : float
        Learning rate for gradient descent
    save_dir : str
        Directory to save model checkpoints
    save_every : int
        Save model every N epochs
    """
    num_samples = encoder_inputs.shape[0]
    num_batches = int(np.ceil(num_samples / batch_size))
    
    # Create directory for saving models
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        
        # Shuffle data
        indices = np.random.permutation(num_samples)
        shuffled_encoder_inputs = encoder_inputs[indices]
        shuffled_decoder_inputs = decoder_inputs[indices]
        shuffled_decoder_targets = decoder_targets[indices]
        
        for batch in range(num_batches):
            # Get batch data
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, num_samples)
            batch_encoder_inputs = shuffled_encoder_inputs[start_idx:end_idx]
            batch_decoder_inputs = shuffled_decoder_inputs[start_idx:end_idx]
            batch_decoder_targets = shuffled_decoder_targets[start_idx:end_idx]
            
            # Forward pass
            encoder_hidden, decoder_hidden, decoder_outputs = model.forward_pass(
                batch_encoder_inputs, batch_decoder_inputs)
            
            # Backward pass
            batch_loss = model.backward_pass(
                batch_encoder_inputs, batch_decoder_inputs, batch_decoder_targets,
                decoder_outputs, encoder_hidden, decoder_hidden, learning_rate)
            
            total_loss += batch_loss * (end_idx - start_idx)
            
            # Print progress
            if batch % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch+1}/{num_batches}, Loss: {batch_loss:.4f}")
        
        # Calculate average loss
        avg_loss = total_loss / num_samples
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}")
        
        # Save model checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_dir = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_model(checkpoint_dir)
            print(f"Model checkpoint saved at epoch {epoch+1}")
    
    # Save final model
    model.save_model(save_dir)
    print(f"Training completed. Final model saved in {save_dir}")
    
    return model
