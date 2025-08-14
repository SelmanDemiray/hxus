import numpy as np
import pickle
import os
import json
import h5py

class EncoderDecoderNN:
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, xp=np):
        self.xp = xp
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Use xp for arrays
        self.encoder_embed = xp.random.randn(vocab_size, embedding_dim) * 0.1
        self.encoder_Wxh = xp.random.randn(embedding_dim, hidden_dim) * 0.1
        self.encoder_Whh = xp.random.randn(hidden_dim, hidden_dim) * 0.1
        self.encoder_bh = xp.zeros((1, hidden_dim))

        self.decoder_embed = xp.random.randn(vocab_size, embedding_dim) * 0.1
        self.decoder_Wxh = xp.random.randn(embedding_dim, hidden_dim) * 0.1
        self.decoder_Whh = xp.random.randn(hidden_dim, hidden_dim) * 0.1
        self.decoder_bh = xp.zeros((1, hidden_dim))
        self.decoder_Why = xp.random.randn(hidden_dim, vocab_size) * 0.1
        self.decoder_by = xp.zeros((1, vocab_size))
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + self.xp.exp(-self.xp.clip(x, -15, 15)))
    
    def softmax(self, x):
        """Softmax activation function"""
        exp_x = self.xp.exp(x - self.xp.max(x, axis=1, keepdims=True))
        return exp_x / self.xp.sum(exp_x, axis=1, keepdims=True)
    
    def forward_pass(self, encoder_inputs, decoder_inputs):
        batch_size = encoder_inputs.shape[0]
        encoder_seq_len = encoder_inputs.shape[1]
        decoder_seq_len = decoder_inputs.shape[1]
        
        # Initialize hidden states and outputs
        encoder_hidden = self.xp.zeros((batch_size, self.hidden_dim))
        decoder_hidden = self.xp.zeros((batch_size, self.hidden_dim))
        decoder_outputs = self.xp.zeros((batch_size, decoder_seq_len, self.vocab_size))
        
        # Encoder forward pass
        for t in range(encoder_seq_len):
            # One-hot encode inputs
            x_t = self.xp.zeros((batch_size, self.vocab_size))
            for i in range(batch_size):
                idx = int(encoder_inputs[i, t])
                if idx > 0:  # Skip padding
                    x_t[i, idx] = 1
            # Embedding lookup
            embed_t = x_t @ self.encoder_embed
            # Update hidden state
            encoder_hidden = self.sigmoid(
                embed_t @ self.encoder_Wxh + 
                encoder_hidden @ self.encoder_Whh + 
                self.encoder_bh
            )
        
        # Transfer encoder final state to decoder initial state
        decoder_hidden = encoder_hidden
        
        # Decoder forward pass
        for t in range(decoder_seq_len):
            # One-hot encode inputs
            x_t = self.xp.zeros((batch_size, self.vocab_size))
            for i in range(batch_size):
                idx = int(decoder_inputs[i, t])
                if idx > 0:  # Skip padding
                    x_t[i, idx] = 1
            
            # Embedding lookup
            embed_t = x_t @ self.decoder_embed
            
            # Update hidden state
            decoder_hidden = self.sigmoid(
                embed_t @ self.decoder_Wxh + 
                decoder_hidden @ self.decoder_Whh + 
                self.decoder_bh
            )
            
            # Compute output
            decoder_outputs[:, t, :] = self.softmax(
                decoder_hidden @ self.decoder_Why + self.decoder_by
            )
        
        return encoder_hidden, decoder_hidden, decoder_outputs
    
    def backward_pass(self, encoder_inputs, decoder_inputs, decoder_targets, decoder_outputs, 
                      encoder_hidden, decoder_hidden, learning_rate=0.01):
        xp = self.xp
        batch_size = encoder_inputs.shape[0]
        encoder_seq_len = encoder_inputs.shape[1]
        decoder_seq_len = decoder_inputs.shape[1]
        
        # Initialize gradients
        dencoder_embed = xp.zeros_like(self.encoder_embed)
        dencoder_Wxh = xp.zeros_like(self.encoder_Wxh)
        dencoder_Whh = xp.zeros_like(self.encoder_Whh)
        dencoder_bh = xp.zeros_like(self.encoder_bh)
        
        ddecoder_embed = xp.zeros_like(self.decoder_embed)
        ddecoder_Wxh = xp.zeros_like(self.decoder_Wxh)
        ddecoder_Whh = xp.zeros_like(self.decoder_Whh)
        ddecoder_bh = xp.zeros_like(self.decoder_bh)
        ddecoder_Why = xp.zeros_like(self.decoder_Why)
        ddecoder_by = xp.zeros_like(self.decoder_by)
        
        # Compute loss
        loss = 0
        for t in range(decoder_seq_len):
            for i in range(batch_size):
                if decoder_targets[i, t] > 0:  # Skip padding
                    loss -= xp.log(decoder_outputs[i, t, int(decoder_targets[i, t])] + 1e-10)
        loss /= batch_size
        
        # Backward pass through decoder
        dh_next = xp.zeros((batch_size, self.hidden_dim))
        
        for t in reversed(range(decoder_seq_len)):
            # Gradient of the softmax output
            dy = decoder_outputs[:, t, :].copy()
            for i in range(batch_size):
                if decoder_targets[i, t] > 0:  # Skip padding
                    dy[i, decoder_targets[i, t]] -= 1

            # Gradient of Why and by
            ddecoder_Why += dh_next.T @ dy
            ddecoder_by += xp.sum(dy, axis=0, keepdims=True)

            # Gradient of hidden state
            dh = dy @ self.decoder_Why.T + dh_next

            # Gate gradients
            dh_raw = (1 - decoder_hidden) * decoder_hidden * dh

            # Gradient of Whh, Wxh, and bh
            ddecoder_bh += xp.sum(dh_raw, axis=0, keepdims=True)

            # One-hot encode inputs
            x_t = xp.zeros((batch_size, self.vocab_size))
            for i in range(batch_size):
                if decoder_inputs[i, t] > 0:  # Skip padding
                    x_t[i, decoder_inputs[i, t]] = 1

            # Embedding lookup
            embed_t = x_t @ self.decoder_embed

            ddecoder_Wxh += embed_t.T @ dh_raw
            ddecoder_Whh += decoder_hidden.T @ dh_raw

            # Gradient of embedding
            dembed = dh_raw @ self.decoder_Wxh.T
            for i in range(batch_size):
                if decoder_inputs[i, t] > 0:  # Skip padding
                    ddecoder_embed[decoder_inputs[i, t]] += dembed[i]

            # Next hidden state gradient
            if t > 0:
                dh_next = dh_raw @ self.decoder_Whh.T
        
        # Update weights with gradient descent
        self.encoder_embed -= learning_rate * dencoder_embed
        self.encoder_Wxh -= learning_rate * dencoder_Wxh
        self.encoder_Whh -= learning_rate * dencoder_Whh
        self.encoder_bh -= learning_rate * dencoder_bh
        
        self.decoder_embed -= learning_rate * ddecoder_embed
        self.decoder_Wxh -= learning_rate * ddecoder_Wxh
        self.decoder_Whh -= learning_rate * ddecoder_Whh
        self.decoder_bh -= learning_rate * ddecoder_bh
        self.decoder_Why -= learning_rate * ddecoder_Why
        self.decoder_by -= learning_rate * ddecoder_by
        
        return loss
    
    def predict(self, encoder_input, data_processor, max_length=20):
        """Generate a response from an input sequence"""
        # Convert input to sequence and reshape for batch size 1
        if isinstance(encoder_input, str):
            encoder_input = data_processor.text_to_sequence(encoder_input)
            encoder_input = encoder_input[:data_processor.max_sequence_length]
            encoder_input = encoder_input + [0] * max(0, data_processor.max_sequence_length - len(encoder_input))
            encoder_input = np.array([encoder_input])
        
        batch_size = encoder_input.shape[0]
        encoder_seq_len = encoder_input.shape[1]
        
        # Initialize hidden states
        encoder_hidden = np.zeros((batch_size, self.hidden_dim))
        
        # Encoder forward pass
        for t in range(encoder_seq_len):
            # One-hot encode inputs
            x_t = np.zeros((batch_size, self.vocab_size))
            for i in range(batch_size):
                if encoder_input[i, t] > 0:  # Skip padding
                    x_t[i, encoder_input[i, t]] = 1
            
            # Embedding lookup
            embed_t = x_t @ self.encoder_embed
            
            # Update hidden state
            encoder_hidden = self.sigmoid(
                embed_t @ self.encoder_Wxh + 
                encoder_hidden @ self.encoder_Whh + 
                self.encoder_bh
            )
        
        # Generate sequence with decoder
        decoder_hidden = encoder_hidden
        decoder_input = np.array([[data_processor.word_to_idx['<START>']]])
        generated_sequence = []
        
        for _ in range(max_length):
            # One-hot encode inputs
            x_t = np.zeros((batch_size, self.vocab_size))
            for i in range(batch_size):
                x_t[i, decoder_input[i, 0]] = 1
            
            # Embedding lookup
            embed_t = x_t @ self.decoder_embed
            
            # Update hidden state
            decoder_hidden = self.sigmoid(
                embed_t @ self.decoder_Wxh + 
                decoder_hidden @ self.decoder_Whh + 
                self.decoder_bh
            )
            
            # Compute output
            output = self.softmax(decoder_hidden @ self.decoder_Why + self.decoder_by)
            
            # Get next token
            next_token = np.argmax(output, axis=1)
            
            # Stop if <END> token
            if next_token[0] == data_processor.word_to_idx['<END>']:
                break
                
            generated_sequence.append(next_token[0])
            decoder_input = next_token.reshape(-1, 1)
        
        return data_processor.sequence_to_text(generated_sequence)
    
    def save_model(self, directory='models'):
        """Save the model in multiple formats"""
        os.makedirs(directory, exist_ok=True)
        
        # Save as pickle
        with open(os.path.join(directory, 'model.pkl'), 'wb') as f:
            pickle.dump(self.__dict__, f)
        
        # Save as JSON
        json_dict = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.__dict__.items()}
        with open(os.path.join(directory, 'model.json'), 'w') as f:
            json.dump(json_dict, f)
        
        # Save as HDF5
        with h5py.File(os.path.join(directory, 'model.h5'), 'w') as f:
            for k, v in self.__dict__.items():
                if isinstance(v, np.ndarray):
                    f.create_dataset(k, data=v)
                else:
                    f.attrs[k] = v
        
        # Save as NumPy
        np.savez(os.path.join(directory, 'model.npz'), **{k: v for k, v in self.__dict__.items() if isinstance(v, np.ndarray)})
        
        print(f"Model saved in {directory} in multiple formats")
    
    @classmethod
    def load_model(cls, path):
        """Load model from file"""
        if path.endswith('.pkl'):
            with open(path, 'rb') as f:
                model_dict = pickle.load(f)
        elif path.endswith('.json'):
            with open(path, 'r') as f:
                model_dict_str = json.load(f)
                model_dict = {k: np.array(v) if isinstance(v, list) else v for k, v in model_dict_str.items()}
        elif path.endswith('.h5'):
            model_dict = {}
            with h5py.File(path, 'r') as f:
                for k in f.keys():
                    model_dict[k] = f[k][()]
                for k in f.attrs:
                    model_dict[k] = f.attrs[k]
        elif path.endswith('.npz'):
            model_arrays = np.load(path)
            model_dict = {k: model_arrays[k] for k in model_arrays.files}
        else:
            raise ValueError(f"Unsupported file format: {path}")
        
        # Create new model instance
        model = cls(model_dict['vocab_size'], 
                    model_dict.get('embedding_dim', 128), 
                    model_dict.get('hidden_dim', 256))
        
        # Load weights
        for k, v in model_dict.items():
            setattr(model, k, v)
        
        print(f"Model loaded from {path}")
        return model
