import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender


class Sets2Sets(SequentialRecommender):
    """PyTorch implementation of Sets2Sets next-basket recommendation.

    The model consumes basket-level histories exported by ``NextBasketDataset`` and predicts
    multi-label targets for the next basket.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # configuration
        self.embedding_size = config["embedding_size"]
        self.hidden_size = (
            config["hidden_size"] if "hidden_size" in config else self.embedding_size
        )
        self.dropout = config["dropout"] if "dropout" in config else 0.0
        self.set_loss_weight = (
            config["SETS2SETS_SET_LOSS_WEIGHT"]
            if "SETS2SETS_SET_LOSS_WEIGHT" in config
            else 10.0
        )
        # Always use ALL negative items (matching original implementation)
        # Original uses: zeros_idx_set = (target == 0).nonzero() to get all negatives
        self.history_bias = (
            float(config["SETS2SETS_HISTORY_BIAS"])
            if "SETS2SETS_HISTORY_BIAS" in config
            else 1.0
        )

        # dataset specific field names
        self.history_items_field = (
            config["SETS2SETS_HISTORY_ITEMS_FIELD"]
            if "SETS2SETS_HISTORY_ITEMS_FIELD" in config
            else "history_item_matrix"
        )
        self.history_length_field = (
            config["SETS2SETS_HISTORY_LENGTH_FIELD"]
            if "SETS2SETS_HISTORY_LENGTH_FIELD" in config
            else "history_basket_length"
        )
        self.history_item_len_field = (
            config["SETS2SETS_HISTORY_ITEM_LENGTH_FIELD"]
            if "SETS2SETS_HISTORY_ITEM_LENGTH_FIELD" in config
            else "history_item_length_per_basket"
        )
        self.target_items_field = (
            config["SETS2SETS_TARGET_ITEMS_FIELD"]
            if "SETS2SETS_TARGET_ITEMS_FIELD" in config
            else "target_item_list"
        )
        self.target_length_field = (
            config["SETS2SETS_TARGET_LENGTH_FIELD"]
            if "SETS2SETS_TARGET_LENGTH_FIELD" in config
            else "target_item_length"
        )

        self.max_history_baskets = (
            int(config["MAX_NBR_HISTORY_BASKETS"])
            if "MAX_NBR_HISTORY_BASKETS" in config
            else 50
        )
        self.max_basket_items = (
            int(config["MAX_NBR_BASKET_SIZE"])
            if "MAX_NBR_BASKET_SIZE" in config
            else 100
        )

        # layers
        # Encoder and decoder have separate embeddings (matching original)
        self.encoder_item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )
        self.decoder_item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )
        self.encoder_gru = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            batch_first=True,
        )
        # Matching original decoder architecture:
        # - attn: nn.Linear(self.hidden_size * 2, self.max_length) - attention over encoder outputs
        # - attn_combine: nn.Linear(self.hidden_size * 2, self.hidden_size) - combines decoder input and attention
        # - gru: nn.GRU(hidden_size, hidden_size, num_layers)
        # - out: nn.Linear(self.hidden_size, self.output_size) - output layer (only takes GRU output)
        self.decoder_gru = nn.GRU(
            input_size=self.hidden_size,  # After attn_combine, input is hidden_size
            hidden_size=self.hidden_size,
            batch_first=True,
        )
        # Original: self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        # Takes concatenation of decoder input embedding and hidden state
        self.attn = nn.Linear(self.hidden_size * 2, self.max_history_baskets)
        # Original: self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # Combines decoder input and attention-applied encoder outputs
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # Original: self.out = nn.Linear(self.hidden_size, self.output_size)
        # Only takes GRU output, not concatenated features
        self.output_layer = nn.Linear(self.hidden_size, self.n_items)
        # Matching original: attn_combine5 maps history_context to per-item weights
        # Original (line 118): self.attn_combine5 = nn.Linear(self.output_size, self.output_size)
        # Original doesn't specify bias=False, so bias=True by default (will be initialized to zeros)
        # Original (line 188): value = torch.sigmoid(self.attn_combine5(history_context).unsqueeze(0))
        # Outputs [1, output_size] where each element is a weight for that item
        self.history_weight_layer = nn.Linear(self.n_items, self.n_items)  # bias=True by default (matching original)
        self.dropout_layer = nn.Dropout(self.dropout)
        # Note: Original uses MSE loss with inverse frequency weights, not BCE
        # We'll compute inverse frequency weights during training
        
        self.apply(self._init_weights)
        
        # Register inverse frequency weights as buffer (computed from training data)
        # Original: weights[idx] = max_freq / codes_freq[idx] if codes_freq[idx] > 0 else 0
        self.register_buffer('inverse_freq_weights', torch.ones(self.n_items))
        self._weights_computed = False
        # For accumulating frequencies across batches
        self.register_buffer('_accumulated_freq', torch.zeros(self.n_items))
    
    def _accumulate_freq_from_batch(self, interaction):
        """Accumulate item frequencies from a batch (for computing weights from all training data)."""
        history_items = interaction[self.history_items_field]
        device = self._accumulated_freq.device
        flat_items = history_items.view(-1)
        mask = flat_items.gt(0)
        indices = flat_items[mask].to(device)  # Move to accumulated_freq's device
        
        if indices.size(0) > 0:
            self._accumulated_freq.scatter_add_(0, indices.long(), torch.ones_like(indices, dtype=torch.float, device=device))
    
    def _finalize_weights_from_accumulated(self):
        """Compute inverse frequency weights from accumulated frequencies."""
        codes_freq = self._accumulated_freq.clone()
        codes_freq[0] = 0.0  # Exclude padding token
        
        max_freq = codes_freq.max()
        if max_freq > 0:
            self.inverse_freq_weights = torch.where(
                codes_freq > 0,
                max_freq / codes_freq,
                torch.zeros_like(codes_freq)
            )
        else:
            self.inverse_freq_weights = torch.ones(self.n_items, device=self.inverse_freq_weights.device)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def _aggregate_baskets(self, history_items, per_basket_len):
        """Convert set of items per basket into averaged embeddings."""

        device = history_items.device
        batch_size, num_baskets, num_items = history_items.shape

        item_emb = self.encoder_item_embedding(history_items)
        length_mask = (
            torch.arange(num_items, device=device)
            .view(1, 1, -1)
            .expand(batch_size, num_baskets, -1)
        ) < per_basket_len.unsqueeze(-1)
        valid_mask = length_mask & history_items.gt(0)

        mask_float = valid_mask.float()
        # Matching original encoder averaging (lines 65-72):
        # Original divides by length = [1] * hidden_size (all ones), effectively no averaging
        # So we just sum embeddings without dividing by count
        basket_emb = (item_emb * mask_float.unsqueeze(-1)).sum(dim=2)
        # Note: Dropout is NOT applied here (matching original - encoder doesn't use dropout)
        # Dropout is only applied to decoder input conditionally (see _compute_logits)
        return basket_emb

    def _encode_history(self, basket_emb, history_len, per_basket_len):
        """Encode history baskets sequentially matching original implementation.
        
        Original (lines 260-264):
        for ei in range(input_length - 1):
            if ei == 0:
                continue  # Skip padding basket [-1]
            encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
            encoder_outputs[ei - 1] = encoder_output[0][0]
        
        Processes baskets one at a time sequentially, skipping padding baskets entirely.
        Padding baskets are [-1] baskets (per_basket_len == 0 or all items are -1).
        """
        batch_size = basket_emb.size(0)
        device = basket_emb.device
        max_baskets = basket_emb.size(1)
        
        # Initialize hidden state (matching original: encoder.initHidden())
        # Original: Variable(torch.zeros(num_layers, 1, self.hidden_size))
        hidden = torch.zeros(1, batch_size, self.hidden_size, device=device)  # [num_layers=1, batch_size, hidden_size]
        
        # Initialize encoder outputs (matching original: Variable(torch.zeros(max_length, encoder.hidden_size)))
        encoder_outputs = torch.zeros(batch_size, max_baskets, self.hidden_size, device=device)
        
        # Create mask for valid baskets (non-padding baskets)
        # Dataset structure matches original: [[-1], basket1, ..., basketN, [-1]]
        # history_len includes padding: history_len = num_hist + 2
        # Valid baskets are at positions 1 to history_len - 2 (skipping padding at 0 and history_len - 1)
        # Matching original: for ei in range(input_length - 1): if ei == 0: continue
        basket_mask = (
            torch.arange(max_baskets, device=device)
            .view(1, -1)
            .expand(batch_size, -1)
        )
        # Valid baskets: positions > 0 AND < history_len - 1 AND have items
        valid_positions = (basket_mask > 0) & (basket_mask < (history_len.unsqueeze(1) - 1))
        has_items_mask = per_basket_len.gt(0)  # [batch_size, max_baskets] - True if basket has items
        basket_mask = valid_positions & has_items_mask  # Both conditions must be true
        
        # Process baskets sequentially (matching original: for ei in range(input_length - 1))
        # Original skips ei==0 (first [-1] padding) and processes ei=1 to input_length-2
        # We skip padding baskets entirely (where basket_mask is False)
        for ei in range(max_baskets):
            # Check which samples have valid (non-padding) baskets at this position
            valid_mask = basket_mask[:, ei]  # [batch_size] - True if this position is a valid basket
            
            if not valid_mask.any():  # Skip if no samples have valid baskets here (matching original continue)
                continue
            
            # Get current basket embedding: [batch_size, hidden_size]
            # This already has -1 items masked out from _aggregate_baskets
            current_basket = basket_emb[:, ei, :]  # [batch_size, hidden_size]
            
            # Process one basket at a time: [batch_size, hidden_size] -> [batch_size, hidden_size]
            # Original: encoder(input_variable[ei], encoder_hidden)
            # With batch_first=True: input is [batch_size, seq_len=1, hidden_size]
            current_basket_seq = current_basket.unsqueeze(1)  # [batch_size, 1, hidden_size]
            
            # Process through GRU: hidden is [num_layers, batch_size, hidden_size]
            # Output: [batch_size, 1, hidden_size], hidden: [num_layers, batch_size, hidden_size]
            # Hidden state accumulates across baskets (matching original sequential processing)
            output, new_hidden = self.encoder_gru(current_basket_seq, hidden)
            
            # Only update hidden state for samples with valid baskets at this position
            # For padding samples, keep previous hidden state (matching original skip behavior)
            # valid_mask: [batch_size] -> [1, batch_size, 1] for broadcasting
            update_mask = valid_mask.float().view(1, -1, 1)  # [1, batch_size, 1]
            hidden = hidden * (1 - update_mask) + new_hidden * update_mask
            
            # Store output at position ei-1 (matching original indexing)
            # Original: encoder_outputs[ei - 1] = encoder_output[0][0]
            # Since we skip ei==0, we store at ei-1 to match original positions
            # But we need to handle the case where ei==0 (shouldn't happen due to mask, but safety)
            if ei > 0:
                encoder_outputs[:, ei - 1, :] = output.squeeze(1)  # [batch_size, hidden_size]
        
        # Create mask for encoder_outputs positions (stored at ei-1, so mask positions 0 to history_len-3)
        # encoder_outputs_mask[i] corresponds to basket at position i+1
        encoder_outputs_mask = (
            torch.arange(max_baskets, device=device)
            .view(1, -1)
            .expand(batch_size, -1)
        ) < (history_len.unsqueeze(1) - 2)  # Positions 0 to history_len-3
        encoder_outputs = encoder_outputs * encoder_outputs_mask.float().unsqueeze(-1)
        
        return encoder_outputs, hidden, basket_mask

    def _history_frequency(self, history_items, history_len):
        """Compute history frequency matching original: history_record[ele] += 1.0 / (input_length - 2)
        
        Original: input_length is the total number of baskets for EACH USER (including padding).
        History frequency normalizes by (input_length - 2) per user, which is the number of baskets used
        for frequency computation (excluding padding at position 0 and decoder input at position input_length-1).
        
        In our implementation:
        - history_len: [batch_size] tensor with total length including padding (equals input_length per user)
        - Actual basket count = history_len - 2 (excluding padding at front and back)
        - Normalization is applied per-user: each user's frequency is divided by (history_len - 2)
        """
        batch_size = history_items.size(0)
        device = history_items.device
        
        # Flatten all items from all baskets
        flat_items = history_items.view(batch_size, -1)
        mask = flat_items.gt(0)
        indices = flat_items.clone()
        indices[~mask] = 0
        
        # Count occurrences per item
        freq = torch.zeros(batch_size, self.n_items, device=device)
        freq.scatter_add_(1, indices, mask.float())
        freq[:, 0] = 0.0  # Exclude padding token
        
        # Normalize by number of baskets per user (matching original: 1.0 / (input_length - 2))
        # history_len includes padding: history_len = num_hist + 2
        # So actual basket count = history_len - 2
        basket_counts = (history_len - 2).float().clamp(min=1.0)  # [batch_size] - per-user basket counts (excluding padding)
        freq = freq / basket_counts.unsqueeze(-1)  # [batch_size, n_items] - broadcast division per user
        
        return freq

    def _prepare_targets(self, interaction):
        target_items = interaction[self.target_items_field]
        target_len = interaction[self.target_length_field].long()

        batch_size, max_items = target_items.shape
        device = target_items.device

        # Create mask for valid positions (within length and not padding)
        # Functionally equivalent to original target_variable[1] (skips front [-1] padding)
        # We achieve same by filtering >= 0 (which skips all -1 padding)
        length_mask = (
            torch.arange(max_items, device=device)
            .view(1, -1)
            .expand(batch_size, -1)
        ) < target_len.unsqueeze(1)
        # Filter out padding (-1) values: only keep valid item IDs (>= 0)
        valid_mask = target_items >= 0
        # Combined mask: must be within length AND not padding
        combined_mask = length_mask & valid_mask
        
        valid_items = target_items.clone()
        # Set invalid positions (padding or beyond length) to 0
        valid_items[~combined_mask] = 0
        # Clamp item IDs to valid range [0, n_items-1] to avoid out-of-bounds errors
        valid_items = torch.clamp(valid_items, 0, self.n_items - 1)

        target = torch.zeros(batch_size, self.n_items, device=device)
        # Only scatter valid items (those that pass both length and padding checks)
        target.scatter_add_(1, valid_items, combined_mask.float())
        target[:, 0] = 0.0  # Ensure padding token (0) is never a target
        target = target.clamp(max=1.0)
        return target, combined_mask.float()

    def _attention(self, decoder_input_emb, hidden, encoder_outputs, history_len):
        """Compute attention matching original implementation.
        
        Original (lines 172-175):
        attn_weights = F.softmax(
            self.attn(torch.cat((droped_ave_embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        
        Note: encoder_outputs are stored at positions ei-1, so encoder_outputs[i] corresponds to basket at position i+1.
        We need to mask attention scores for invalid encoder_outputs positions.
        """
        # Concatenate decoder input embedding and hidden state
        # decoder_input_emb: [batch_size, hidden_size]
        # hidden: [num_layers, batch_size, hidden_size] -> take last layer: hidden[-1]: [batch_size, hidden_size]
        hidden_last = hidden[-1]  # [batch_size, hidden_size]
        combined = torch.cat([decoder_input_emb, hidden_last], dim=1)  # [batch_size, hidden_size * 2]
        
        # Apply attention layer: [batch_size, hidden_size * 2] -> [batch_size, max_length]
        attn_scores = self.attn(combined)  # [batch_size, max_history_baskets]
        
        # Create mask for encoder_outputs positions (encoder_outputs[i] corresponds to basket at position i+1)
        # encoder_outputs are at positions 0 to history_len-3 (for baskets at positions 1 to history_len-2)
        batch_size = encoder_outputs.size(0)
        max_baskets = encoder_outputs.size(1)
        device = encoder_outputs.device
        encoder_outputs_mask = (
            torch.arange(max_baskets, device=device)
            .view(1, -1)
            .expand(batch_size, -1)
        ) < (history_len.unsqueeze(1) - 2)  # Positions 0 to history_len-3
        
        # Mask invalid encoder_outputs positions
        attn_scores = attn_scores.masked_fill(~encoder_outputs_mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1)  # [batch_size, max_history_baskets]
        
        # Apply attention to encoder outputs: [batch_size, 1, max_history_baskets] @ [batch_size, max_history_baskets, hidden_size]
        attn_applied = torch.bmm(
            attn_weights.unsqueeze(1),  # [batch_size, 1, max_history_baskets]
            encoder_outputs  # [batch_size, max_history_baskets, hidden_size]
        )  # [batch_size, 1, hidden_size]
        attn_applied = attn_applied.squeeze(1)  # [batch_size, hidden_size]
        
        return attn_applied, attn_weights

    def _compute_logits(self, interaction):
        history_items = interaction[self.history_items_field]
        history_len = interaction[self.history_length_field].long()
        per_basket_len = interaction[self.history_item_len_field].long()

        basket_emb = self._aggregate_baskets(history_items, per_basket_len)
        encoder_outputs, hidden, basket_mask = self._encode_history(
            basket_emb, history_len, per_basket_len
        )
        history_freq = self._history_frequency(history_items, history_len)

        batch_size = history_items.size(0)
        device = history_items.device
        # Matching original: last_input = input_variable[input_length - 2]
        # With padding structure [[-1], basket1, ..., basketN, [-1]]:
        # input_length = num_hist + 2, so input_length - 2 = num_hist (last valid basket)
        # In our structure: baskets are at positions 1 to history_len - 2
        # So last basket is at position history_len - 2
        last_index = (history_len - 2).clamp(min=1)  # Position history_len - 2, but at least 1
        batch_ids = torch.arange(batch_size, device=device)
        
        # Get last basket items (need to re-embed using decoder's embedding)
        # Original decoder (lines 128-150): uses decoder's own embedding to embed last basket items
        last_basket_items = history_items[batch_ids, last_index]  # [batch_size, max_basket_items]
        last_basket_len = per_basket_len[batch_ids, last_index]  # [batch_size]
        
        # Re-embed using decoder's embedding (matching original line 137: embedded = self.embedding(ele))
        last_basket_item_emb = self.decoder_item_embedding(last_basket_items)  # [batch_size, max_basket_items, embedding_size]
        
        # Average embeddings (matching original lines 136-148)
        # Create mask for valid items
        max_basket_items = last_basket_items.size(1)
        length_mask = (
            torch.arange(max_basket_items, device=device)
            .view(1, -1)
            .expand(batch_size, -1)
        ) < last_basket_len.unsqueeze(1)
        valid_mask = length_mask & last_basket_items.gt(0)
        mask_float = valid_mask.float()
        
        # Sum embeddings (matching original: average_embedding = tmp + embedded, then divide by length=[1]*hidden_size)
        decoder_input_emb = (last_basket_item_emb * mask_float.unsqueeze(-1)).sum(dim=1)  # [batch_size, embedding_size]
        
        # Matching original decoder forward (lines 127-199):
        # 1. Decoder input: droped_ave_embedded (averaged embedding of last basket items)
        #    Original (lines 168-171): if use_dropout: droped_ave_embedded = self.dropout(embedding)
        
        # Apply dropout conditionally to decoder input (matching original: if use_dropout)
        if self.dropout > 0:
            decoder_input_emb = self.dropout_layer(decoder_input_emb)
        
        # 2. Attention: attn_weights = F.softmax(self.attn(torch.cat((droped_ave_embedded[0], hidden[0]), 1)), dim=1)
        #    attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        attn_applied, attn_weights = self._attention(decoder_input_emb, hidden, encoder_outputs, history_len)
        # attn_applied: [batch_size, hidden_size]
        
        # 3. Combine decoder input and attention: output = torch.cat((droped_ave_embedded[0], attn_applied[0]), 1)
        combined = torch.cat([decoder_input_emb, attn_applied], dim=1)  # [batch_size, hidden_size * 2]
        
        # 4. Apply attn_combine: output = self.attn_combine(output).unsqueeze(0)
        output = self.attn_combine(combined)  # [batch_size, hidden_size]
        output = output.unsqueeze(1)  # [batch_size, 1, hidden_size] to match GRU input format
        
        # 5. ReLU: output = F.relu(output)
        output = F.relu(output)
        
        # 6. GRU: output, hidden = self.gru(output, hidden)
        decoder_output, _ = self.decoder_gru(output, hidden)  # decoder_output: [batch_size, 1, hidden_size]
        decoder_state = decoder_output.squeeze(1)  # [batch_size, hidden_size]
        
        # 7. Linear output: linear_output = self.out(output[0])
        #    Original only uses GRU output, not concatenated features
        linear_output = self.output_layer(decoder_state)  # [batch_size, n_items]
        
        # Matching original implementation (line 197):
        # output = F.softmax(linear_output * (one_vec - res * value[0]) + history_context * value[0], dim=1)
        # where value = sigmoid(attn_combine5(history_context))
        # and res is binary mask where history_context != 0
        
        if self.history_bias > 0:
            # Compute per-item weights from history_freq (matching original: value = sigmoid(attn_combine5(history_context)))
            # Original: value = torch.sigmoid(self.attn_combine5(history_context).unsqueeze(0))
            # value shape: [1, output_size] where each element is a weight for that item
            # In batch: [batch_size, n_items] -> [batch_size, n_items]
            history_weight = torch.sigmoid(self.history_weight_layer(history_freq))  # [batch_size, n_items]
            
            # Create binary mask: res[history_context != 0] = 1
            history_mask = (history_freq > 0).float()  # [batch_size, n_items]
            
            # one_vec = ones of size n_items
            one_vec = torch.ones(self.n_items, device=linear_output.device)  # [n_items]
            one_vec_expanded = one_vec.unsqueeze(0)  # [1, n_items]
            
            # Apply original formula: linear_output * (one_vec - res * value) + history_context * value
            # Original (line 197): output = F.softmax(linear_output * (one_vec - res * value[0]) + history_context * value[0], dim=1)
            # For batch: [batch_size, n_items] * ([1, n_items] - [batch_size, n_items] * [batch_size, n_items]) + [batch_size, n_items] * [batch_size, n_items]
            mask_weighted = history_mask * history_weight  # [batch_size, n_items]
            logits = linear_output * (one_vec_expanded - mask_weighted) + history_freq * history_weight
        else:
            # If history_bias is 0, don't use history (matching original when disabled)
            logits = linear_output
        
        return logits

    def forward(self, interaction):
        return self._compute_logits(interaction)

    def _compute_set_loss(self, pred, target):
        """Compute set loss matching original implementation exactly.
        
        Original (lines 219-231) processes single sample (batch_size=1):
        - ones_idx_set = (target == 1).nonzero()  # [num_positives, 2] where col 0 is batch_idx (always 0), col 1 is item_idx
        - zeros_idx_set = (target == 0).nonzero()  # [num_negatives, 2]
        - ones_set = torch.index_select(pred, 1, ones_idx_set[:, 1])  # [1, num_positives]
        - zeros_set = torch.index_select(pred, 1, zeros_idx_set[:, 1])  # [1, num_negatives]
        - repeat_ones = ones_set.repeat(1, zeros_set.shape[1])  # [1, num_positives * num_negatives]
        - repeat_zeros_set = torch.transpose(zeros_set.repeat(ones_set.shape[1], 1), 0, 1).clone()  # [1, num_positives * num_negatives]
        - repeat_zeros = repeat_zeros_set.reshape(1, -1)  # [1, num_positives * num_negatives]
        - difference_val = -(repeat_ones - repeat_zeros)  # [1, num_positives * num_negatives]
        - exp_val = torch.exp(difference_val)
        - exp_loss = torch.sum(exp_val)
        - normalized_loss = exp_loss / (zeros_set.shape[1] * ones_set.shape[1])
        """
        # Convert to data (detach gradients) matching original
        pred_data = pred.data
        target_data = target.data
        
        batch_size = pred_data.size(0)
        set_losses = []
        
        # Process each sample in batch (original processes one at a time)
        for b in range(batch_size):
            # Get single sample predictions and target
            pred_sample = pred_data[b:b+1]  # [1, n_items] to match original shape
            target_sample = target_data[b:b+1]  # [1, n_items]
            
            # Get indices of positive and negative items (matching original)
            ones_idx_set = (target_sample == 1).nonzero(as_tuple=False)  # [num_positives, 2]
            zeros_idx_set = (target_sample == 0).nonzero(as_tuple=False)  # [num_negatives, 2]
            
            if ones_idx_set.size(0) == 0 or zeros_idx_set.size(0) == 0:
                set_losses.append(torch.tensor(0.0, device=pred.device, requires_grad=True))
                continue
            
            # Extract predictions matching original: torch.index_select(pred, 1, ones_idx_set[:, 1])
            ones_set = torch.index_select(pred_sample, 1, ones_idx_set[:, 1])  # [1, num_positives]
            zeros_set = torch.index_select(pred_sample, 1, zeros_idx_set[:, 1])  # [1, num_negatives]
            
            # Create pairwise differences matching original exactly
            # repeat_ones = ones_set.repeat(1, zeros_set.shape[1])
            repeat_ones = ones_set.repeat(1, zeros_set.size(1))  # [1, num_positives * num_negatives]
            # repeat_zeros_set = torch.transpose(zeros_set.repeat(ones_set.shape[1], 1), 0, 1).clone()
            repeat_zeros_set = torch.transpose(zeros_set.repeat(ones_set.size(1), 1), 0, 1).clone()  # [1, num_positives * num_negatives]
            repeat_zeros = repeat_zeros_set.reshape(1, -1)  # [1, num_positives * num_negatives]
            
            # Compute differences: -(repeat_ones - repeat_zeros)
            difference_val = -(repeat_ones - repeat_zeros)  # [1, num_positives * num_negatives]
            exp_val = torch.exp(difference_val)
            exp_loss = torch.sum(exp_val)
            normalized_loss = exp_loss / (zeros_set.size(1) * ones_set.size(1))
            
            set_losses.append(normalized_loss)
        
        # Average across batch (original processes one sample, so this is equivalent)
        set_loss = torch.stack(set_losses).mean()
        return set_loss

    def compute_inverse_freq_weights_from_dataset(self, train_data):
        """Compute inverse frequency weights from ALL training data (matching original).
        
        Original (lines 759-767) computes from all training users:
        codes_freq = get_codes_frequency_no_vector(history_data, input_size, future_data.keys())
        max_freq = max(codes_freq)
        for idx in range(len(codes_freq)):
            if codes_freq[idx] > 0:
                weights[idx] = max_freq / codes_freq[idx]
            else:
                weights[idx] = 0
        
        Call this method before training starts with train_data from data_preparation.
        This is the preferred method for exact match with original implementation.
        """
        # Access the full dataset from train_data
        dataset = train_data.dataset
        
        # Get all history items from training data
        # After _change_feat_format, history_items_field is a tensor [num_samples, max_history_baskets, max_basket_items]
        history_items = dataset.inter_feat[self.history_items_field]
        
        # Get device - dataset tensors are typically on CPU, but we want to compute on model's device
        device = self.inverse_freq_weights.device
        
        # Flatten all items from all users and baskets
        flat_items = history_items.view(-1)  # [num_samples * max_history_baskets * max_basket_items]
        
        # Filter out padding and invalid items (0 is padding, -1 is also padding)
        mask = flat_items.gt(0)  # Only keep items > 0
        indices = flat_items[mask].to(device)  # Move to model's device
        
        # Count occurrences
        codes_freq = torch.zeros(self.n_items, device=device)
        if indices.size(0) > 0:
            codes_freq.scatter_add_(0, indices.long(), torch.ones_like(indices, dtype=torch.float, device=device))
        codes_freq[0] = 0.0  # Exclude padding token
        
        # Compute inverse frequency weights
        max_freq = codes_freq.max()
        if max_freq > 0:
            # weights[idx] = max_freq / codes_freq[idx] if codes_freq[idx] > 0 else 0
            self.inverse_freq_weights = torch.where(
                codes_freq > 0,
                max_freq / codes_freq,
                torch.zeros_like(codes_freq)
            )
        else:
            self.inverse_freq_weights = torch.ones(self.n_items, device=self.inverse_freq_weights.device)
        
        self._weights_computed = True
    
    def finalize_weights_from_accumulation(self):
        """Finalize weights from accumulated frequencies (call after first epoch completes).
        
        This is called automatically if weights weren't computed from full dataset.
        For exact match, prefer compute_inverse_freq_weights_from_dataset(train_data).
        """
        if not self._weights_computed:
            self._finalize_weights_from_accumulated()
            self._weights_computed = True
    
    def calculate_loss(self, interaction):
        """Calculate loss matching original: MSE with inverse frequency weights + set loss."""
        # Accumulate frequencies from batches if weights not computed from full dataset
        # This allows computing weights from all training data by accumulating across batches
        # Weights will be finalized after first epoch completes (see trainer hook) or when compute_inverse_freq_weights_from_dataset is called
        if not self._weights_computed:
            self._accumulate_freq_from_batch(interaction)
        
        logits = self.forward(interaction)
        target, target_mask = self._prepare_targets(interaction)
        
        # Apply softmax to get probabilities (matching original decoder output)
        # Original applies softmax in decoder forward pass (line 197)
        pred = F.softmax(logits, dim=1)
        
        # MSE loss with inverse frequency weights (matching original line 215)
        # Original: mseloss = torch.sum(weights * torch.pow((pred - target), 2))
        # For batch: pred and target are [batch_size, n_items], weights is [1, n_items]
        # Original processes one sample at a time, so we sum over items for each sample, then average over batch
        weights = self.inverse_freq_weights.unsqueeze(0)  # [1, n_items]
        # Sum over items for each sample: [batch_size, n_items] -> [batch_size]
        mse_per_sample = torch.sum(weights * torch.pow((pred - target), 2), dim=1)
        # Average over batch (matching RecBole convention, original has batch_size=1 so sum=mean)
        mse_loss = mse_per_sample.mean()
        
        loss = mse_loss
        if self.set_loss_weight > 0:
            # Compute set loss using original pairwise difference method
            set_loss = self._compute_set_loss(pred, target)
            loss = loss + self.set_loss_weight * set_loss
        
        # Debug: Log loss components periodically to diagnose issues
        # This helps identify if MSE or set loss is dominating
        if hasattr(self, '_loss_debug_counter'):
            self._loss_debug_counter += 1
        else:
            self._loss_debug_counter = 0
        
        if self._loss_debug_counter % 1000 == 0:
            import logging
            logger = logging.getLogger('recbole')
            logger.debug(f"[Loss Debug] MSE: {mse_loss.item():.2f}, Set Loss: {set_loss.item() if self.set_loss_weight > 0 else 0:.2f}, "
                        f"Total: {loss.item():.2f}, Pred max/min: {pred.max().item():.4f}/{pred.min().item():.4f}, "
                        f"Target sum: {target.sum().item():.1f}")
        
        return loss

    def predict(self, interaction):
        """Predict scores for specific items.
        
        Returns probabilities (after softmax) matching original decoder output.
        """
        logits = self.forward(interaction)
        # Apply softmax to get probabilities (matching original decoder output)
        probs = F.softmax(logits, dim=1)
        item_ids = interaction[self.ITEM_ID]
        scores = probs.gather(1, item_ids.view(-1, 1)).squeeze(1)
        return scores

    def full_sort_predict(self, interaction):
        """Predict scores for all items.
        
        Returns probabilities (after softmax) matching original decoder output.
        """
        logits = self.forward(interaction)
        # Apply softmax to get probabilities (matching original decoder output)
        probs = F.softmax(logits, dim=1)
        return probs

