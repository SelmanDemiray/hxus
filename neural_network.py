import numpy as np
import pickle
import os
import json
import h5py
import cupy

class TransformerEncoderDecoder:
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_heads=4, xp=np):
        self.xp = xp
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.head_dim = hidden_dim // num_heads

        # Embeddings
        self.encoder_embed = xp.random.randn(vocab_size, embedding_dim) * 0.1
        self.decoder_embed = xp.random.randn(vocab_size, embedding_dim) * 0.1

        # Positional encoding
        self.encoder_pos = self._positional_encoding(100, embedding_dim)
        self.decoder_pos = self._positional_encoding(100, embedding_dim)

        # Encoder self-attention weights
        self.enc_attn_Wq = xp.random.randn(num_heads, embedding_dim, self.head_dim) * 0.1
        self.enc_attn_Wk = xp.random.randn(num_heads, embedding_dim, self.head_dim) * 0.1
        self.enc_attn_Wv = xp.random.randn(num_heads, embedding_dim, self.head_dim) * 0.1
        self.enc_attn_Wo = xp.random.randn(num_heads * self.head_dim, embedding_dim) * 0.1

        # Encoder FFN
        self.enc_ffn_W1 = xp.random.randn(embedding_dim, hidden_dim) * 0.1
        self.enc_ffn_b1 = xp.zeros((1, hidden_dim))
        self.enc_ffn_W2 = xp.random.randn(hidden_dim, embedding_dim) * 0.1
        self.enc_ffn_b2 = xp.zeros((1, embedding_dim))

        # Decoder self-attention weights
        self.dec_attn_Wq = xp.random.randn(num_heads, embedding_dim, self.head_dim) * 0.1
        self.dec_attn_Wk = xp.random.randn(num_heads, embedding_dim, self.head_dim) * 0.1
        self.dec_attn_Wv = xp.random.randn(num_heads, embedding_dim, self.head_dim) * 0.1
        self.dec_attn_Wo = xp.random.randn(num_heads * self.head_dim, embedding_dim) * 0.1

        # Decoder cross-attention weights
        self.cross_attn_Wq = xp.random.randn(num_heads, embedding_dim, self.head_dim) * 0.1
        self.cross_attn_Wk = xp.random.randn(num_heads, embedding_dim, self.head_dim) * 0.1
        self.cross_attn_Wv = xp.random.randn(num_heads, embedding_dim, self.head_dim) * 0.1
        self.cross_attn_Wo = xp.random.randn(num_heads * self.head_dim, embedding_dim) * 0.1

        # Decoder FFN
        self.dec_ffn_W1 = xp.random.randn(embedding_dim, hidden_dim) * 0.1
        self.dec_ffn_b1 = xp.zeros((1, hidden_dim))
        self.dec_ffn_W2 = xp.random.randn(hidden_dim, embedding_dim) * 0.1
        self.dec_ffn_b2 = xp.zeros((1, embedding_dim))

        # Output layer
        self.out_W = xp.random.randn(embedding_dim, vocab_size) * 0.1
        self.out_b = xp.zeros((1, vocab_size))

    def _positional_encoding(self, seq_len, dim):
        xp = self.xp
        pos = xp.arange(seq_len)[:, None]
        i = xp.arange(dim)[None, :]
        angle_rates = 1 / xp.power(10000, (2 * (i // 2)) / xp.float32(dim))
        angle_rads = pos * angle_rates
        pos_encoding = xp.zeros((seq_len, dim))
        pos_encoding[:, 0::2] = xp.sin(angle_rads[:, 0::2])
        pos_encoding[:, 1::2] = xp.cos(angle_rads[:, 1::2])
        return pos_encoding

    def softmax(self, x, axis=-1):
        exp_x = self.xp.exp(x - self.xp.max(x, axis=axis, keepdims=True))
        return exp_x / self.xp.sum(exp_x, axis=axis, keepdims=True)

    def multi_head_attention_forward(self, q, k, v, Wq, Wk, Wv, Wo, mask=None):
        """Forward pass for multi-head attention with cached values for backward pass"""
        batch_size, seq_len, embed_dim = q.shape
        
        # Store inputs for backward pass
        cache = {
            'q': q, 'k': k, 'v': v,
            'Wq': Wq, 'Wk': Wk, 'Wv': Wv, 'Wo': Wo,
            'mask': mask
        }
        
        heads = []
        head_outputs = []
        
        for h in range(self.num_heads):
            q_h = q @ Wq[h]  # (batch, seq_len, head_dim)
            k_h = k @ Wk[h]
            v_h = v @ Wv[h]
            
            scores = self.xp.matmul(q_h, k_h.transpose(0, 2, 1)) / self.xp.sqrt(self.head_dim)
            
            if mask is not None:
                scores = scores + mask
                
            weights = self.softmax(scores, axis=-1)
            attn = self.xp.matmul(weights, v_h)
            
            # Store intermediate values for backward pass
            head_outputs.append({
                'q_h': q_h, 'k_h': k_h, 'v_h': v_h,
                'scores': scores, 'weights': weights, 'attn': attn
            })
            heads.append(attn)
        
        concat = self.xp.concatenate(heads, axis=-1)
        out = concat @ Wo
        
        cache['head_outputs'] = head_outputs
        cache['concat'] = concat
        
        return out, cache

    def multi_head_attention_backward(self, grad_out, cache):
        """Backward pass for multi-head attention"""
        xp = self.xp
        q, k, v = cache['q'], cache['k'], cache['v']
        Wq, Wk, Wv, Wo = cache['Wq'], cache['Wk'], cache['Wv'], cache['Wo']
        head_outputs = cache['head_outputs']
        concat = cache['concat']
        
        batch_size, seq_len, embed_dim = q.shape
        
        # Gradients for output projection
        dWo = concat.transpose(0, 2, 1) @ grad_out
        dWo = dWo.sum(axis=0)  # Sum over batch dimension
        dconcat = grad_out @ Wo.T
        
        # Split gradient for each head
        head_grads = xp.split(dconcat, self.num_heads, axis=-1)
        
        # Initialize gradients
        dWq = xp.zeros_like(Wq)
        dWk = xp.zeros_like(Wk)
        dWv = xp.zeros_like(Wv)
        dq = xp.zeros_like(q)
        dk = xp.zeros_like(k)
        dv = xp.zeros_like(v)
        
        for h in range(self.num_heads):
            head_cache = head_outputs[h]
            q_h, k_h, v_h = head_cache['q_h'], head_cache['k_h'], head_cache['v_h']
            weights, scores = head_cache['weights'], head_cache['scores']
            
            dattn = head_grads[h]  # Gradient w.r.t. attention output
            
            # Gradient w.r.t. attention weights and values
            dweights = dattn @ v_h.transpose(0, 2, 1)
            dv_h = weights.transpose(0, 2, 1) @ dattn
            
            # Gradient w.r.t. attention scores (softmax backward)
            dscores = weights * (dweights - xp.sum(weights * dweights, axis=-1, keepdims=True))
            dscores = dscores / xp.sqrt(self.head_dim)
            
            # Gradient w.r.t. queries and keys
            dq_h = dscores @ k_h
            dk_h = dscores.transpose(0, 2, 1) @ q_h
            
            # Gradient w.r.t. weight matrices
            dWq_h = q.transpose(0, 2, 1) @ dq_h
            dWq[h] = dWq_h.sum(axis=0)
            dWk_h = k.transpose(0, 2, 1) @ dk_h
            dWk[h] = dWk_h.sum(axis=0)
            dWv_h = v.transpose(0, 2, 1) @ dv_h
            dWv[h] = dWv_h.sum(axis=0)
            
            # Gradient w.r.t. input queries, keys, values
            dq += dq_h @ Wq[h].T
            dk += dk_h @ Wk[h].T
            dv += dv_h @ Wv[h].T
        
        return dq, dk, dv, dWq, dWk, dWv, dWo

    def feed_forward_backward(self, x, grad_out, W1, b1, W2, b2):
        """Backward pass for feed-forward network"""
        xp = self.xp
        
        # Forward pass (recreate intermediate values)
        h1 = x @ W1 + b1
        h1_relu = xp.maximum(0, h1)
        
        # Backward pass
        dW2 = h1_relu.transpose(0, 2, 1) @ grad_out
        dW2 = dW2.sum(axis=0)
        db2 = grad_out.sum(axis=(0, 1), keepdims=True)
        
        dh1_relu = grad_out @ W2.T
        dh1 = dh1_relu * (h1 > 0)  # ReLU derivative
        
        dW1 = x.transpose(0, 2, 1) @ dh1
        dW1 = dW1.sum(axis=0)
        db1 = dh1.sum(axis=(0, 1), keepdims=True)
        dx = dh1 @ W1.T
        
        return dx, dW1, db1, dW2, db2

    def forward_pass(self, encoder_inputs, decoder_inputs):
        """Enhanced forward pass that stores intermediate values for backward pass"""
        batch_size = encoder_inputs.shape[0]
        encoder_seq_len = encoder_inputs.shape[1]
        decoder_seq_len = decoder_inputs.shape[1]

        # Store all intermediate values for backward pass
        self.forward_cache = {}

        # Encoder embedding + positional encoding
        encoder_emb = self.encoder_embed[encoder_inputs] + self.encoder_pos[:encoder_seq_len]
        self.forward_cache['encoder_emb'] = encoder_emb
        self.forward_cache['encoder_inputs'] = encoder_inputs
        
        # Encoder self-attention
        enc_attn, enc_attn_cache = self.multi_head_attention_forward(
            encoder_emb, encoder_emb, encoder_emb,
            self.enc_attn_Wq, self.enc_attn_Wk, self.enc_attn_Wv, self.enc_attn_Wo
        )
        self.forward_cache['enc_attn_cache'] = enc_attn_cache
        
        # Encoder FFN
        enc_ffn_input = enc_attn  # Residual connection input
        h1 = enc_ffn_input @ self.enc_ffn_W1 + self.enc_ffn_b1
        h1_relu = self.xp.maximum(0, h1)
        enc_ffn = h1_relu @ self.enc_ffn_W2 + self.enc_ffn_b2
        
        encoder_out = enc_attn + enc_ffn  # Residual connection
        self.forward_cache['enc_ffn_input'] = enc_ffn_input
        self.forward_cache['enc_h1'] = h1
        self.forward_cache['enc_h1_relu'] = h1_relu
        self.forward_cache['encoder_out'] = encoder_out

        # Decoder embedding + positional encoding
        decoder_emb = self.decoder_embed[decoder_inputs] + self.decoder_pos[:decoder_seq_len]
        self.forward_cache['decoder_emb'] = decoder_emb
        self.forward_cache['decoder_inputs'] = decoder_inputs
        
        # Decoder self-attention (mask future positions)
        xp = self.xp
        mask = xp.triu(xp.ones((decoder_seq_len, decoder_seq_len)) * -1e9, k=1)
        mask = xp.broadcast_to(mask, (batch_size, decoder_seq_len, decoder_seq_len))
        
        dec_attn, dec_attn_cache = self.multi_head_attention_forward(
            decoder_emb, decoder_emb, decoder_emb,
            self.dec_attn_Wq, self.dec_attn_Wk, self.dec_attn_Wv, self.dec_attn_Wo,
            mask=mask
        )
        self.forward_cache['dec_attn_cache'] = dec_attn_cache
        
        # Decoder cross-attention (attend to encoder outputs)
        cross_attn, cross_attn_cache = self.multi_head_attention_forward(
            dec_attn, encoder_out, encoder_out,
            self.cross_attn_Wq, self.cross_attn_Wk, self.cross_attn_Wv, self.cross_attn_Wo
        )
        self.forward_cache['cross_attn_cache'] = cross_attn_cache
        
        # Decoder FFN
        dec_ffn_input = cross_attn
        h1_dec = dec_ffn_input @ self.dec_ffn_W1 + self.dec_ffn_b1
        h1_dec_relu = self.xp.maximum(0, h1_dec)
        dec_ffn = h1_dec_relu @ self.dec_ffn_W2 + self.dec_ffn_b2
        
        decoder_out = cross_attn + dec_ffn  # Residual connection
        self.forward_cache['dec_ffn_input'] = dec_ffn_input
        self.forward_cache['dec_h1'] = h1_dec
        self.forward_cache['dec_h1_relu'] = h1_dec_relu
        self.forward_cache['decoder_out'] = decoder_out

        # Output logits
        logits = decoder_out @ self.out_W + self.out_b
        outputs = self.softmax(logits, axis=-1)
        self.forward_cache['logits'] = logits
        self.forward_cache['outputs'] = outputs
        
        return encoder_out, decoder_out, outputs

    def backward_pass(self, encoder_inputs, decoder_inputs, decoder_targets, 
                      decoder_outputs=None, encoder_hidden=None, decoder_hidden=None, learning_rate=0.01):
        """Proper backward pass for the transformer - compatible with existing training code"""
        xp = self.xp
        
        # Use cached outputs from forward pass if decoder_outputs not provided
        if decoder_outputs is None:
            decoder_outputs = self.forward_cache['outputs']
        
        batch_size, decoder_seq_len, vocab_size = decoder_outputs.shape
        
        # Compute loss and initial gradient
        loss = 0
        
        # Cross-entropy loss computation
        for t in range(decoder_seq_len):
            for i in range(batch_size):
                if decoder_targets[i, t] > 0:  # Skip padding
                    target_idx = int(decoder_targets[i, t])
                    loss -= xp.log(decoder_outputs[i, t, target_idx] + 1e-10)
        
        loss /= batch_size
        
        # Convert to softmax gradient
        doutputs = decoder_outputs.copy()
        for t in range(decoder_seq_len):
            for i in range(batch_size):
                if decoder_targets[i, t] > 0:
                    target_idx = int(decoder_targets[i, t])
                    doutputs[i, t, target_idx] -= 1
        doutputs /= batch_size
        
        # Gradient w.r.t. output layer
        decoder_out = self.forward_cache['decoder_out']
        dout_W = decoder_out.transpose(0, 2, 1) @ doutputs
        dout_W = dout_W.sum(axis=0)
        dout_b = doutputs.sum(axis=(0, 1), keepdims=True)
        ddecoder_out = doutputs @ self.out_W.T
        
        # Decoder FFN backward
        dec_ffn_input = self.forward_cache['dec_ffn_input']
        ddec_ffn_input, ddec_ffn_W1, ddec_ffn_b1, ddec_ffn_W2, ddec_ffn_b2 = \
            self.feed_forward_backward(dec_ffn_input, ddecoder_out, 
                                     self.dec_ffn_W1, self.dec_ffn_b1, 
                                     self.dec_ffn_W2, self.dec_ffn_b2)
        
        # Residual connection gradient
        dcross_attn = ddec_ffn_input
        
        # Cross-attention backward
        cross_attn_cache = self.forward_cache['cross_attn_cache']
        ddec_attn, dencoder_out_cross, _, dcross_attn_Wq, dcross_attn_Wk, dcross_attn_Wv, dcross_attn_Wo = \
            self.multi_head_attention_backward(dcross_attn, cross_attn_cache)
        
        # Decoder self-attention backward
        dec_attn_cache = self.forward_cache['dec_attn_cache']
        ddecoder_emb, _, _, ddec_attn_Wq, ddec_attn_Wk, ddec_attn_Wv, ddec_attn_Wo = \
            self.multi_head_attention_backward(ddec_attn, dec_attn_cache)
        
        # Decoder embedding gradients
        ddecoder_embed = xp.zeros_like(self.decoder_embed)
        decoder_inputs = self.forward_cache['decoder_inputs']
        for i in range(batch_size):
            for t in range(decoder_seq_len):
                if decoder_inputs[i, t] > 0:
                    ddecoder_embed[decoder_inputs[i, t]] += ddecoder_emb[i, t]
        
        # Encoder backward pass
        dencoder_out = dencoder_out_cross
        
        # Encoder FFN backward
        enc_ffn_input = self.forward_cache['enc_ffn_input']
        denc_attn, denc_ffn_W1, denc_ffn_b1, denc_ffn_W2, denc_ffn_b2 = \
            self.feed_forward_backward(enc_ffn_input, dencoder_out,
                                     self.enc_ffn_W1, self.enc_ffn_b1,
                                     self.enc_ffn_W2, self.enc_ffn_b2)
        
        # Encoder self-attention backward
        enc_attn_cache = self.forward_cache['enc_attn_cache']
        dencoder_emb, _, _, denc_attn_Wq, denc_attn_Wk, denc_attn_Wv, denc_attn_Wo = \
            self.multi_head_attention_backward(denc_attn, enc_attn_cache)
        
        # Encoder embedding gradients
        dencoder_embed = xp.zeros_like(self.encoder_embed)
        encoder_inputs = self.forward_cache['encoder_inputs']
        for i in range(batch_size):
            for t in range(encoder_inputs.shape[1]):
                if encoder_inputs[i, t] > 0:
                    dencoder_embed[encoder_inputs[i, t]] += dencoder_emb[i, t]
        
        # Update parameters
        self.encoder_embed -= learning_rate * xp.clip(dencoder_embed, -1, 1)
        self.decoder_embed -= learning_rate * xp.clip(ddecoder_embed, -1, 1)
        
        self.enc_attn_Wq -= learning_rate * xp.clip(denc_attn_Wq, -1, 1)
        self.enc_attn_Wk -= learning_rate * xp.clip(denc_attn_Wk, -1, 1)
        self.enc_attn_Wv -= learning_rate * xp.clip(denc_attn_Wv, -1, 1)
        self.enc_attn_Wo -= learning_rate * xp.clip(denc_attn_Wo, -1, 1)
        
        self.dec_attn_Wq -= learning_rate * xp.clip(ddec_attn_Wq, -1, 1)
        self.dec_attn_Wk -= learning_rate * xp.clip(ddec_attn_Wk, -1, 1)
        self.dec_attn_Wv -= learning_rate * xp.clip(ddec_attn_Wv, -1, 1)
        self.dec_attn_Wo -= learning_rate * xp.clip(ddec_attn_Wo, -1, 1)
        
        self.cross_attn_Wq -= learning_rate * xp.clip(dcross_attn_Wq, -1, 1)
        self.cross_attn_Wk -= learning_rate * xp.clip(dcross_attn_Wk, -1, 1)
        self.cross_attn_Wv -= learning_rate * xp.clip(dcross_attn_Wv, -1, 1)
        self.cross_attn_Wo -= learning_rate * xp.clip(dcross_attn_Wo, -1, 1)
        
        self.enc_ffn_W1 -= learning_rate * xp.clip(denc_ffn_W1, -1, 1)
        self.enc_ffn_b1 -= learning_rate * xp.clip(denc_ffn_b1.squeeze(), -1, 1)
        self.enc_ffn_W2 -= learning_rate * xp.clip(denc_ffn_W2, -1, 1)
        self.enc_ffn_b2 -= learning_rate * xp.clip(denc_ffn_b2.squeeze(), -1, 1)
        
        self.dec_ffn_W1 -= learning_rate * xp.clip(ddec_ffn_W1, -1, 1)
        self.dec_ffn_b1 -= learning_rate * xp.clip(ddec_ffn_b1.squeeze(), -1, 1)
        self.dec_ffn_W2 -= learning_rate * xp.clip(ddec_ffn_W2, -1, 1)
        self.dec_ffn_b2 -= learning_rate * xp.clip(ddec_ffn_b2.squeeze(), -1, 1)
        
        self.out_W -= learning_rate * dout_W
        self.out_b      -= learning_rate * dout_b.squeeze()
        
        return loss

    def train_step(self, encoder_inputs, decoder_inputs, decoder_targets, learning_rate=0.01):
        """Complete training step: forward + backward pass"""
        # Forward pass
        encoder_out, decoder_out, outputs = self.forward_pass(encoder_inputs, decoder_inputs)
        
        # Backward pass - now compatible with training.py expectations
        loss = self.backward_pass(encoder_inputs, decoder_inputs, decoder_targets, 
                                outputs, encoder_out, decoder_out, learning_rate)
        
        return loss, outputs
    
    # Keep the original methods for compatibility
    def multi_head_attention(self, q, k, v, Wq, Wk, Wv, Wo, mask=None):
        out, _ = self.multi_head_attention_forward(q, k, v, Wq, Wk, Wv, Wo, mask)
        return out

    def feed_forward(self, x, W1, b1, W2, b2):
        h = self.xp.maximum(0, x @ W1 + b1)
        out = h @ W2 + b2
        return out
    
    def predict(self, encoder_input, data_processor, max_length=20):
        """Generate a response from an input sequence (simplified for inference)"""
        # This would need to be updated to work with the transformer architecture
        # For now, keeping the original structure but noting it needs revision
        raise NotImplementedError("Predict method needs to be updated for transformer architecture")
    
    def save_model(self, directory='models'):
        """Save the model in multiple formats"""
        os.makedirs(directory, exist_ok=True)

        # Convert CuPy arrays to NumPy for saving
        model_dict = {}
        for k, v in self.__dict__.items():
            if k == "xp" or k == "forward_cache":
                continue  # Do not save the xp module or forward_cache
            if type(v).__module__ == "cupy.core.core":
                model_dict[k] = v.get()
            else:
                model_dict[k] = v

        # Save as pickle
        with open(os.path.join(directory, 'model.pkl'), 'wb') as f:
            pickle.dump(model_dict, f)

        # Save as JSON (skip arrays that can't be converted)
        json_dict = {}
        for k, v in model_dict.items():
            if isinstance(v, np.ndarray):
                json_dict[k] = v.tolist()
            elif isinstance(v, (int, float, str, list, dict)):
                json_dict[k] = v
            # skip other types

        with open(os.path.join(directory, 'model.json'), 'w') as f:
            json.dump(json_dict, f)

        # Save as HDF5
        with h5py.File(os.path.join(directory, 'model.h5'), 'w') as f:
            for k, v in model_dict.items():
                if isinstance(v, np.ndarray):
                    f.create_dataset(k, data=v)
                else:
                    # Convert CuPy scalars to NumPy scalars for attributes
                    if hasattr(v, 'get'):
                        v = v.get()
                    # Convert CuPy scalars (e.g., cupy.int32) to Python scalars
                    if hasattr(v, 'item'):
                        # Only call .item() if v is a scalar
                        if hasattr(v, 'shape') and v.shape == () or (hasattr(v, 'size') and v.size == 1):
                            v = v.item()
                    f.attrs[k] = v

        # Save as NumPy
        np.savez(os.path.join(directory, 'model.npz'), **{k: v for k, v in model_dict.items() if isinstance(v, np.ndarray)})

        print(f"Model saved in {directory} in multiple formats")
    
    @classmethod
    def load_model(cls, path, xp=np):
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
                    model_dict.get('hidden_dim', 256),
                    model_dict.get('num_heads', 4),
                    xp=xp)
        
        # Load weights and convert to appropriate array type
        for k, v in model_dict.items():
            if isinstance(v, np.ndarray) and xp.__name__ == "cupy":
                setattr(model, k, xp.array(v))  # Convert to CuPy
            else:
                setattr(model, k, v)
        
        print(f"Model loaded from {path}")
        return model