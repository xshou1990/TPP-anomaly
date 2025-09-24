import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_thinning import EventSampler
import math


class CJ_RMTPP(nn.Module):
    def __init__(self, num_event_types, embed_dim=32, hidden_dim=64,
                 time_embed_size=16, num_layers=1, num_heads=2,
                 mc_num_sample_per_step=20, loss_integral_num_sample_per_step=20,
                 use_padding=False, thinning_num_sample=1, thinning_num_exp=500,
                 thinning_over_sample_rate=5, thinning_num_samples_boundary=5,
                 thinning_dtime_max=5, thinning_patience_counter=5):

        super(CJ_RMTPP, self).__init__()
        self.num_event_types = num_event_types
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.time_embed_size = time_embed_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mc_num_sample_per_step = mc_num_sample_per_step
        self.loss_integral_num_sample_per_step = loss_integral_num_sample_per_step
        self.use_padding = use_padding

        # Add thinning sampler parameters
        self.thinning_num_sample = thinning_num_sample
        self.thinning_num_exp = thinning_num_exp
        self.thinning_over_sample_rate = thinning_over_sample_rate
        self.thinning_num_samples_boundary = thinning_num_samples_boundary
        self.thinning_dtime_max = thinning_dtime_max
        self.thinning_patience_counter = thinning_patience_counter

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.2,
            batch_first=True
        )

        # Event type embedding (handle 1-based indexing if no padding)
        padding_idx = 0 if use_padding else None

        self.type_embed = nn.Embedding(
            num_event_types + (1 if not use_padding else 0),  # +1 for 1-based indexing
            self.embed_dim,  # Changed from hidden_dim to embed_dim
            padding_idx=padding_idx
        )

        # Enhanced time embedding
        self.time_embed = nn.Linear(1, self.embed_dim)  # Changed from hidden_dim to embed_dim

        self.input_projection = nn.Linear(embed_dim * 2, hidden_dim)

        # Changed from GRU to simple RNN
        self.rnn = nn.RNN(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
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

        # Linear layer for hidden_to_intensity_logits
        self.hidden_to_intensity_logits = nn.Linear(hidden_dim, num_event_types)

        # Initialize parameters
        self._init_weights()

    def init_thinning_sampler(self, device):
        """Initialize the thinning sampler"""
        self.thinning_sampler = EventSampler(
            num_sample=self.thinning_num_sample,
            num_exp=self.thinning_num_exp,
            over_sample_rate=self.thinning_over_sample_rate,
            num_samples_boundary=self.thinning_num_samples_boundary,
            dtime_max=self.thinning_dtime_max,
            patience_counter=self.thinning_patience_counter,
            device=device
        )

    def intensity_fn_wrapper(self, time_seq, time_delta_seq, event_seq, dtime_samples,
                             max_steps=None, compute_last_step_only=False):
        """
        Wrapper function to compute intensities for the thinning sampler.
        This converts the sampler's interface to match your model's intensity computation.

        Args:
            time_seq: [batch_size, seq_len], absolute times
            time_delta_seq: [batch_size, seq_len], time intervals
            event_seq: [batch_size, seq_len], event types
            dtime_samples: [batch_size, seq_len, num_samples], time deltas to sample at
            max_steps: maximum sequence length to process
            compute_last_step_only: whether to compute only the last step

        Returns:
            intensities: [batch_size, seq_len, num_samples, num_event_types]
        """
        batch_size, seq_len, num_samples = dtime_samples.shape

        # Reshape for parallel computation
        # [batch_size * seq_len * num_samples]
        dtime_samples_flat = dtime_samples.reshape(-1)

        # Repeat hidden states for each sample
        if compute_last_step_only:
            # Use only the last hidden state
            last_hidden = self.get_last_hidden_state(time_seq, time_delta_seq, event_seq)
            # [batch_size, 1, hidden_dim] -> [batch_size * num_samples, hidden_dim]
            hidden_repeated = last_hidden.repeat_interleave(num_samples, dim=0)
        else:
            # Get all hidden states
            hiddens = self.forward(time_seq, time_delta_seq, event_seq)
            # [batch_size, seq_len, hidden_dim] -> [batch_size * seq_len * num_samples, hidden_dim]
            hidden_repeated = hiddens.unsqueeze(2).repeat(1, 1, num_samples, 1).reshape(-1, self.hidden_dim)

        # Compute intensities for all samples
        # [batch_size * seq_len * num_samples, num_event_types]
        intensities_flat = self.compute_intensity_from_hidden(hidden_repeated, dtime_samples_flat)

        # Reshape back to [batch_size, seq_len, num_samples, num_event_types]
        intensities = intensities_flat.reshape(batch_size, seq_len, num_samples, self.num_event_types)

        return intensities

    def compute_intensity_from_hidden(self, hiddens, dtimes):
        """
        Compute intensity from hidden states and time deltas.

        Args:
            hiddens: [*, hidden_dim] hidden states
            dtimes: [*] time deltas

        Returns:
            intensities: [*, num_event_types]
        """
        # Ensure proper dimensions
        if hiddens.dim() == 1:
            hiddens = hiddens.unsqueeze(0)
        if dtimes.dim() == 0:
            dtimes = dtimes.unsqueeze(0)

        # Expand dtimes to match hidden dimensions if needed
        if hiddens.shape[0] != dtimes.shape[0]:
            dtimes = dtimes.expand(hiddens.shape[0])

        # Compute intensity using your existing method
        past_influence = self.hidden_to_intensity_logits(hiddens)
        exponent = past_influence + self.w_t * dtimes.unsqueeze(-1) + self.b_t
        exponent = torch.clamp(exponent, min=-50, max=50)
        intensity = torch.exp(exponent)

        return intensity

    def get_last_hidden_state(self, time_seq, time_delta_seq, event_seq):
        """
        Get the last hidden state for the sequence.
        """
        hiddens = self.forward(time_seq, time_delta_seq, event_seq)
        # Get the last non-padded hidden state
        if self.use_padding:
            mask = (event_seq != 0)
        else:
            mask = (event_seq >= 1)

        # Get indices of last events
        seq_lens = mask.sum(dim=1)
        last_indices = seq_lens - 1

        # Gather last hidden states
        batch_indices = torch.arange(hiddens.size(0), device=hiddens.device)
        last_hidden = hiddens[batch_indices, last_indices]

        return last_hidden.unsqueeze(1)  # [batch_size, 1, hidden_dim]

    def predict_with_thinning(self, dts, types, num_sample=10, look_ahead=10):
        """Predict next event using the thinning algorithm"""
        # Initialize sampler if not already done
        if not hasattr(self, 'thinning_sampler'):
            self.init_thinning_sampler(dts.device)

        with torch.no_grad():
            # Convert dts to absolute times
            times = torch.cumsum(dts, dim=1)

            # Get sequence lengths
            if self.use_padding:
                mask = (types != 0)
            else:
                mask = (types >= 1)
            seq_lens = mask.sum(dim=1)

            # Prepare inputs for thinning
            time_seq = times
            time_delta_seq = dts
            event_seq = types

            # Compute next event times using thinning
            next_times, weights = self.thinning_sampler.draw_next_time_one_step(
                time_seq=time_seq,
                time_delta_seq=time_delta_seq,
                event_seq=event_seq,
                dtime_boundary=None,  # Will be computed internally
                intensity_fn=self.intensity_fn_wrapper,
                compute_last_step_only=True  # Only predict from last event
            )

            # next_times: [batch_size, seq_len, num_sample]
            # We only want predictions from the last event
            batch_size = dts.size(0)
            batch_indices = torch.arange(batch_size, device=dts.device)
            last_event_indices = seq_lens - 1

            # Get samples for last event
            last_event_samples = next_times[batch_indices, last_event_indices]  # [batch_size, num_sample]

            # Weighted average of samples
            pred_dtimes = (last_event_samples * weights[batch_indices, last_event_indices]).sum(dim=1)

            # Predict event type at the sampled time
            last_hidden = self.get_last_hidden_state(time_seq, time_delta_seq, event_seq)

            # Use average predicted time for type prediction
            type_logits = self.type_output(last_hidden.squeeze(1))
            pred_types = torch.argmax(type_logits, dim=-1)

            return pred_types, pred_dtimes

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
        combined = self.input_projection(combined)  # Project to hidden_dim
        combined = self.dropout(combined)

        # Process sequence with RNN
        lengths = mask.squeeze(-1).sum(1).cpu().int()  # Convert to int
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
        past_influence = self.hidden_to_intensity_logits(hiddens)  # Changed to use the correct layer
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
        ).view(batch_size, seq_len - 1)

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

            # Fixed intensity calculation
            intensity_input = self.hidden_to_intensity_logits(last_hidden)
            intensity = torch.exp(intensity_input.gather(1, most_probable.unsqueeze(1)) +
                                  w_selected.unsqueeze(1) * 0.1 +  # Small dt assumption
                                  b_selected.unsqueeze(1))
            pred_times = dts[:, -1] + 1.0 / (intensity.squeeze() + 1e-7)

            return pred_types, pred_times

    def predict_with_thinning(self, dts, types, num_sample=10, look_ahead=10):
        """Predict next event using thinning algorithm"""
        with torch.no_grad():
            hiddens = self.forward(dts, types)
            last_hidden = hiddens[:, -1:, :]

            # Generate candidate times
            max_intensity = torch.exp(self.hidden_to_intensity_logits(last_hidden)).sum(-1) * 1.5
            candidate_times = torch.rand(
                (dts.size(0), num_sample),
                device=dts.device
            ) * look_ahead

            # Evaluate intensities at candidate times
            expanded_hidden = last_hidden.expand(-1, num_sample, -1)
            intensities = self.compute_intensity(
                expanded_hidden,
                candidate_times
            ).sum(-1)

            # Acceptance/rejection
            uniforms = torch.rand_like(candidate_times)
            accepted = uniforms * max_intensity <= intensities

            # Handle case where no samples are accepted
            pred_times = torch.zeros(dts.size(0), device=dts.device)

            for i in range(dts.size(0)):
                accepted_times_i = candidate_times[i][accepted[i]]
                if accepted_times_i.numel() > 0:
                    # Take the earliest accepted time
                    pred_times[i] = accepted_times_i.min()
                else:
                    # Fallback: use simple prediction if no samples accepted
                    # Compute intensity for this sequence
                    intensity_input = self.hidden_to_intensity_logits(last_hidden[i])  # [1, num_event_types]

                    # Use the most probable event type for fallback
                    type_logits_i = self.type_output(last_hidden[i])  # [1, num_event_types]
                    most_probable_i = torch.argmax(type_logits_i)  # [1]

                    # Gather the correct intensity value
                    # intensity_input shape: [1, num_event_types]
                    # most_probable_i shape: [1] -> needs to be [1, 1] for gather
                    intensity_value = intensity_input.gather(1, most_probable_i.unsqueeze(0).unsqueeze(1))  # [1, 1]

                    w_selected_i = self.w_t.squeeze(0).gather(0, most_probable_i)  # [1]
                    b_selected_i = self.b_t.squeeze(0).gather(0, most_probable_i)  # [1]

                    # Compute expected time based on intensity
                    intensity_i = torch.exp(intensity_value.squeeze() +
                                            w_selected_i * 0.1 +  # Small dt assumption
                                            b_selected_i)
                    pred_times[i] = 1.0 / (intensity_i + 1e-7)

            # Predict type at the selected time
            type_probs = F.softmax(self.type_output(last_hidden.squeeze(1)), dim=-1)
            pred_types = torch.argmax(type_probs, dim=-1)

            return pred_types, pred_times
