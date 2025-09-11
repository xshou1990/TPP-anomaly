import torch
import torch.nn as nn
import torch.nn.functional as F

class JJJ_RMTPP(nn.Module):
    def __init__(self, num_event_types, embed_dim=32, hidden_dim=64, 
                 time_embed_size=16, num_layers=1, num_heads=2,
                 mc_num_sample_per_step=20, loss_integral_num_sample_per_step=20,
                 use_padding=False):
        super(JJJ_RMTPP, self).__init__()
        self.num_event_types = num_event_types
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.time_embed_size = time_embed_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mc_num_sample_per_step = mc_num_sample_per_step
        self.loss_integral_num_sample_per_step = loss_integral_num_sample_per_step
        self.use_padding = use_padding
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.2,
            batch_first=True
        )

        # Event type embedding (handle 1-based indexing if no padding)
        padding_idx = 0 if use_padding else None
        self.type_embed = nn.Embedding(
            num_event_types + (0 if use_padding else 1),  # +1 for 1-based indexing
            embed_dim,
            padding_idx=padding_idx
        )

        # Enhanced time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_size),
            nn.ReLU(),
            nn.Linear(time_embed_size, embed_dim)
          )
        
        # Changed from GRU to simple RNN
        self.rnn = nn.RNN(  # <--- This is the key change
            input_size=embed_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='relu'  # Can choose 'tanh' (default) or 'relu'
        )

        # Intensity parameters
        self.w_t = nn.Parameter(torch.zeros(1, num_event_types))
        self.b_t = nn.Parameter(torch.zeros(1, num_event_types))

        # Output layers with dropout
        self.dropout = nn.Dropout(0.1)
        self.type_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            self.dropout,
            nn.Linear(hidden_dim, num_event_types)
        )

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.w_t)
        nn.init.xavier_uniform_(self.b_t)
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, dts, types):
        # Create mask (handle both padded and non-padded cases)
        if self.use_padding:
            mask = (types != 0).unsqueeze(-1).float()
        else:
            mask = (types >= 1).unsqueeze(-1).float()  # For 1-based indexing

        # Embeddings with masking
        type_emb = self.type_embed(types) * mask
        time_emb = self.time_embed(dts.unsqueeze(-1)) * mask
        
        # Combine features
        combined = torch.cat([type_emb, time_emb], dim=-1)
        combined = self.dropout(combined)

        # Process sequence with RNN
        lengths = mask.squeeze(-1).sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            combined, lengths, batch_first=True, enforce_sorted=False)
        
        packed_output, _ = self.rnn(packed)  # <-- Now using simple RNN
        hiddens, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True)
        
        return hiddens

    def compute_intensity(self, hiddens, dts, types=None):
        """
        Compute intensity with proper dimension handling
        Args:
            hiddens: [batch_size, seq_len, hidden_dim]
            dts: [batch_size, seq_len]
            types: [batch_size, seq_len] or None
        """
        # Compute base intensity
        past_influence = self.type_output(hiddens)  # [batch_size, seq_len, num_event_types]
        exponent = past_influence + self.w_t * dts.unsqueeze(-1) + self.b_t
        exponent = torch.clamp(exponent, min=-50, max=50)
        intensity = torch.exp(exponent)  # [batch_size, seq_len, num_event_types]
        
        if types is not None:
            # Ensure types has correct shape for gathering
            if types.dim() == 2:
                types = types.unsqueeze(-1)  # [batch_size, seq_len, 1]
            elif types.dim() == 1:
                types = types.unsqueeze(-1).unsqueeze(-1)  # [batch_size, 1, 1]
                
            # Gather intensities for actual event types
            true_intensity = torch.gather(intensity, -1, types).squeeze(-1)
            return intensity, true_intensity
        return intensity

    def compute_loss(self, dts, types, mask=None):
        """Handle both padded and non-padded sequences"""
        if mask is None:
            mask = (types != 0).float() if self.use_padding else (types >= 1).float()

        batch_size, seq_len = types.size()
        hiddens = self.forward(dts[:, :-1], types[:, :-1])
        
        # Type prediction loss
        type_logits = self.type_output(hiddens)  # [batch_size, seq_len-1, num_event_types]
        type_loss = F.cross_entropy(
            type_logits.reshape(-1, self.num_event_types),
            types[:, 1:].reshape(-1),
            reduction='none',
            ignore_index=0 if self.use_padding else -100
        ).view(batch_size, seq_len-1)

        # Time loss calculation - ensure proper dimensions
        intensity, true_intensity = self.compute_intensity(
            hiddens, 
            dts[:, 1:], 
            types[:, 1:]  # Pass types with proper dimensions
        )
        
        log_true_intensity = torch.log(true_intensity + 1e-7)
        total_intensity = intensity.sum(-1)
        time_loss = -log_true_intensity + total_intensity

        # Combine losses
        valid_mask = mask[:, 1:]
        num_valid = valid_mask.sum()
        total_loss = (
            (type_loss * valid_mask).sum() +
            (time_loss * valid_mask).sum()
        ) / max(num_valid, 1)

        return total_loss

    def predict_next_event(self, dts, types):
        """Unchanged prediction method"""
        with torch.no_grad():
            hiddens = self.forward(dts, types)
            last_hidden = hiddens[:, -1, :]

            # Predict next type
            type_logits = self.type_output(last_hidden)
            pred_types = torch.argmax(type_logits, dim=-1)

            # Predict next time using the most probable event type
            type_probs = F.softmax(type_logits, dim=-1)
            most_probable = torch.argmax(type_probs, dim=-1)
            w_selected = self.w_t.squeeze(0).gather(0, most_probable)
            b_selected = self.b_t.squeeze(0).gather(0, most_probable)

            intensity = torch.exp(type_logits.gather(1, most_probable.unsqueeze(1)) + b_selected.unsqueeze(1))
            pred_times = dts[:, -1] + 1.0 / (intensity.squeeze() + 1e-7)

            return pred_types, pred_times

    def prepare_batch(self, batch_sequences):
        """Convert taxi-style sequences to model inputs"""
        max_len = max(len(seq) for seq in batch_sequences)
        batch_size = len(batch_sequences)
        
        times = torch.zeros(batch_size, max_len)
        dts = torch.zeros(batch_size, max_len)
        types = torch.zeros(batch_size, max_len, dtype=torch.long)
        
        for i, seq in enumerate(batch_sequences):
            seq_len = len(seq)
            seq_times = [event['time_since_start'] for event in seq]
            seq_types = [event['type_event'] for event in seq]
            
            times[i, :seq_len] = torch.tensor(seq_times)
            types[i, :seq_len] = torch.tensor(seq_types)
            
            if seq_len > 1:
                dts[i, 1:seq_len] = times[i, 1:seq_len] - times[i, :seq_len-1]
                
        return times, dts, types
    
    def predict_with_thinning(self, dts, types, num_sample=10, look_ahead=10):
        """Predict next event using thinning algorithm"""
        with torch.no_grad():
            hiddens = self.forward(dts, types)
            last_hidden = hiddens[:, -1:, :]
            
            # Generate candidate times
            max_intensity = torch.exp(self.type_output(last_hidden)).sum(-1) * 1.5
            candidate_times = torch.rand(
                (dts.size(0), num_sample),
                device=dts.device
            ) * look_ahead
            
            # Evaluate intensities at candidate times
            expanded_hidden = last_hidden.expand(-1, num_sample, -1)
            intensities = self.compute_intensity(
                expanded_hidden, 
                candidate_times.unsqueeze(-1)
            ).sum(-1)
            
            # Acceptance/rejection
            uniforms = torch.rand_like(candidate_times)
            accepted = uniforms * max_intensity <= intensities
            accepted_times = candidate_times[accepted]
            
            if accepted_times.numel() == 0:
                # Fallback to simple prediction if no samples accepted
                return self.predict_next_event(dts, types)
            
            # Select earliest accepted time for each sequence
            pred_times, _ = accepted_times.min(dim=1)
            
            # Predict type at the selected time
            type_probs = F.softmax(self.type_output(last_hidden), dim=-1)
            pred_types = torch.argmax(type_probs, dim=-1)
            
            return pred_types, pred_times