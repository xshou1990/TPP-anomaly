import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

class JJJ_RMTPP(nn.Module):
    def __init__(self, num_event_types, embed_dim, hidden_dim):
        super(JJJ_RMTPP, self).__init__()
        self.num_event_types = num_event_types
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Event type embedding with padding_idx=0
        self.type_embed = nn.Embedding(num_event_types, embed_dim, padding_idx=0)

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU()
        )

        # Use GRU instead of RNN for better gradient flow
        self.rnn = nn.GRU(input_size=embed_dim * 2,
                          hidden_size=hidden_dim,
                          batch_first=True)

        # Intensity parameters
        self.w_t = nn.Parameter(torch.zeros(1, num_event_types))
        self.b_t = nn.Parameter(torch.zeros(1, num_event_types))

        # Output layer for type prediction
        self.type_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_event_types)
        )

        # Initialize parameters
        nn.init.xavier_uniform_(self.w_t)
        nn.init.xavier_uniform_(self.b_t)
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)

    def forward(self, dts, types):
        # Create mask for embedding
        mask = (types != 0).unsqueeze(-1).float()

        # Embed event types with masking
        type_emb = self.type_embed(types)
        type_emb = type_emb * mask

        # Embed times
        time_emb = self.time_embed(dts.unsqueeze(-1))
        time_emb = time_emb * mask

        # Combine embeddings
        combined = torch.cat([type_emb, time_emb], dim=-1)

        # Pack padded sequence
        lengths = mask.squeeze(-1).sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            combined, lengths, batch_first=True, enforce_sorted=False)

        # Process through GRU
        packed_output, _ = self.rnn(packed)
        hiddens, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True)

        return hiddens

    def compute_intensity(self, hiddens, dts):
        past_influence = self.type_output(hiddens)
        exponent = past_influence + self.w_t * dts.unsqueeze(-1) + self.b_t
        exponent = torch.clamp(exponent, min=-50, max=50)
        return torch.exp(exponent)

    def compute_loss(self, dts, types, mask):
        device = next(self.parameters()).device
        dts, types, mask = dts.to(device), types.to(device), mask.to(device)

        batch_size, seq_len = types.size()

        # Get hidden states for all but last event
        hiddens = self.forward(dts[:, :-1], types[:, :-1])
        past_influence = self.type_output(hiddens)

        # Compute intensity
        intensity = self.compute_intensity(hiddens, dts[:, 1:])

        # Event type prediction loss
        type_loss = F.cross_entropy(
            past_influence.reshape(-1, self.num_event_types),
            types[:, 1:].reshape(-1),
            reduction='none',
            ignore_index=0  # Ignore padding
        ).view(batch_size, seq_len-1)

        # --- Exact integral calculation ---
        w = self.w_t.squeeze(0)
        b = self.b_t.squeeze(0)
        dt_next = dts[:, 1:]

        base_terms = past_influence + b
        base_terms = torch.clamp(base_terms, min=-50, max=50)
        shifted_terms = base_terms + w * dt_next.unsqueeze(-1)
        shifted_terms = torch.clamp(shifted_terms, min=-50, max=50)

        exp_base = torch.exp(base_terms)
        exp_shifted = torch.exp(shifted_terms)

        # Handle different cases for w
        mask_nonzero = (w != 0).view(1, 1, -1)
        integral_per_type = torch.where(
            mask_nonzero,
            (exp_shifted - exp_base) / w,
            dt_next.unsqueeze(-1) * exp_base
        )
        total_integral = integral_per_type.sum(dim=-1)

        # --- Time loss calculation ---
        true_indices = types[:, 1:].unsqueeze(-1)
        true_intensity = torch.gather(intensity, -1, true_indices).squeeze(-1)

        # Mask out padding events
        valid_mask = mask[:, 1:] * (types[:, 1:] != 0).float()
        log_true_intensity = torch.log(true_intensity + 1e-7)
        time_loss = - (log_true_intensity - total_integral)

        # Apply valid mask
        num_valid = valid_mask.sum()
        total_loss = (
            (type_loss * valid_mask).sum() +
            (time_loss * valid_mask).sum()
        ) / num_valid

        return total_loss

    def predict_next_event(self, dts, types):
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

            # Intensity for most probable event
            intensity = torch.exp(type_logits.gather(1, most_probable.unsqueeze(1)) + b_selected.unsqueeze(1))
            pred_times = dts[:, -1] + 1.0 / (intensity.squeeze() + 1e-7)

            return pred_types, pred_times
        
    def train_epoch(self, loader, optimizer):
        """Train the model for one epoch"""
        self.train()
        total_loss = 0
        for times, dts, types, mask in loader:
            optimizer.zero_grad()
            loss = self.compute_loss(dts, types, mask)
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * types.size(0)
        return total_loss / len(loader.dataset)

    def validate(self, loader):
        """Validate the model"""
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for times, dts, types, mask in loader:
                loss = self.compute_loss(dts, types, mask)
                total_loss += loss.item() * types.size(0)
        return total_loss / len(loader.dataset)

    def predict_sequences(self, sequences):
        """
        Generate predictions with proper device handling
        Returns DataFrame with columns:
            sequence_id, event_idx, pred_event_type, pred_time_since_last,
            true_event_type, true_time_since_last
        """
        self.eval()
        all_predictions = []
        device = next(self.parameters()).device  # Get model's device

        for seq_id, (raw_times, raw_types) in enumerate(sequences):

            # Convert to tensors and move to model's device
            times = torch.tensor(raw_times, dtype=torch.float32, device=device)
            types = torch.tensor(raw_types, dtype=torch.long, device=device)

            # Compute inter-event times on same device
            dts = torch.zeros_like(times, device=device)
            if len(times) > 1:
                dts[1:] = times[1:] - times[:-1]

            # Process sequence step-by-step
            sequence_predictions = []
            history_dts = torch.zeros(0, dtype=torch.float32, device=device)
            history_types = torch.zeros(0, dtype=torch.long, device=device)

            for event_idx in range(1, len(times)):  # Start from 1 since we need history
                # Add current event to history
                if event_idx > 0:
                    history_dts = torch.cat([history_dts, dts[event_idx].unsqueeze(0)])
                    history_types = torch.cat([history_types, types[event_idx].unsqueeze(0)])

                # Skip prediction if not enough history
                if event_idx < 1:
                    continue

                # Predict next event (already on correct device)
                with torch.no_grad():
                    # Format inputs for model (add batch dimension)
                    batch_dts = history_dts.unsqueeze(0)  # Already on correct device
                    batch_types = history_types.unsqueeze(0)

                    # Get prediction
                    pred_types, pred_times = self.predict_next_event(batch_dts, batch_types)

                    # Store results (moving to CPU if needed)
                    sequence_predictions.append({
                        'event_idx': event_idx,
                        'pred_event_type': pred_types[0].item() - 1,
                        'pred_time_since_last': pred_times[0].item(),
                        'true_event_type': types[event_idx].item() - 1,
                        'true_time_since_last': dts[event_idx].item()
                    })

            # Add sequence predictions to main list
            for pred in sequence_predictions:
                all_predictions.append({
                    'sequence_id': seq_id + 1,
                    **pred
                })

        return pd.DataFrame(all_predictions)