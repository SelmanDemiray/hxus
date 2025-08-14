import numpy as np
import time
import os
import uuid
import matplotlib.pyplot as plt
from datetime import datetime
from neural_network import EncoderDecoderNN

def generate_model_name():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    rand = uuid.uuid4().hex[:6]
    return f"model_{timestamp}_{rand}"

def train_model(encoder_inputs, decoder_inputs, decoder_targets, 
                model, epochs=10, batch_size=32, learning_rate=0.01,
                save_dir='models', save_every=5):
    """
    Train the encoder-decoder model
    """
    num_samples = encoder_inputs.shape[0]
    num_batches = int(np.ceil(num_samples / batch_size))

    # Generate unique model name and directories
    model_name = generate_model_name()
    model_dir = os.path.join(save_dir, model_name)
    vis_dir = os.path.join('visualizations', model_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    epoch_losses = []
    batch_losses = []

    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        batch_epoch_losses = []

        # Shuffle data
        indices = np.random.permutation(num_samples)
        shuffled_encoder_inputs = encoder_inputs[indices]
        shuffled_decoder_inputs = decoder_inputs[indices]
        shuffled_decoder_targets = decoder_targets[indices]

        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, num_samples)
            batch_encoder_inputs = shuffled_encoder_inputs[start_idx:end_idx]
            batch_decoder_inputs = shuffled_decoder_inputs[start_idx:end_idx]
            batch_decoder_targets = shuffled_decoder_targets[start_idx:end_idx]

            encoder_hidden, decoder_hidden, decoder_outputs = model.forward_pass(
                batch_encoder_inputs, batch_decoder_inputs)

            batch_loss = model.backward_pass(
                batch_encoder_inputs, batch_decoder_inputs, batch_decoder_targets,
                decoder_outputs, encoder_hidden, decoder_hidden, learning_rate)

            total_loss += batch_loss * (end_idx - start_idx)
            batch_epoch_losses.append(batch_loss)

            if batch % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch+1}/{num_batches}, Loss: {batch_loss:.4f}")

        avg_loss = total_loss / num_samples
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}")

        epoch_losses.append(avg_loss)
        batch_losses.append(batch_epoch_losses)

    # Save a single visualization for the entire training progress
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_losses, marker='o', label='Epoch Loss')
    for i, batch_epoch_losses in enumerate(batch_losses):
        plt.plot([i + (j / len(batch_epoch_losses)) for j in range(len(batch_epoch_losses))], batch_epoch_losses, alpha=0.3, color='gray')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress: Epoch and Batch Loss')
    plt.legend()
    plt.savefig(os.path.join(vis_dir, 'training_progress.png'))
    plt.close()

    # Save raw loss data
    np.save(os.path.join(vis_dir, 'batch_losses.npy'), np.array(batch_losses))
    np.save(os.path.join(vis_dir, 'epoch_losses.npy'), np.array(epoch_losses))

    # Save final model
    model.save_model(model_dir)
    print(f"Training completed. Final model saved in {model_dir}")
    print(f"Final visualization saved in {vis_dir}")

    return model_name, model
