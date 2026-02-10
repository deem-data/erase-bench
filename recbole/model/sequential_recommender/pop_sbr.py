# -*- coding: utf-8 -*-
"""
Pop_SBR (Popularity Baseline for Sequential Recommendation)
################################################

A simple popularity-based baseline for sequential recommendation.
This model recommends the most popular items seen during training,
ignoring the user's interaction history.

Reference:
    Adapted from RecBole's Pop model for sequential recommendation tasks.
"""

import torch
import torch.nn as nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType, ModelType


class Pop_SBR(SequentialRecommender):
    """
    Popularity baseline for sequential recommendation.

    This model simply recommends items based on their global popularity
    (frequency of occurrence in the training data), independent of the
    user's session history.
    """

    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL  # No trainable parameters

    def __init__(self, config, dataset):
        super(Pop_SBR, self).__init__(config, dataset)

        # Item popularity counter
        self.item_cnt = torch.zeros(
            self.n_items, 1, dtype=torch.long, device=self.device, requires_grad=False
        )
        self.max_cnt = None

        # Fake parameter for optimizer
        self.fake_loss = nn.Parameter(torch.zeros(1))

        # Register non-parameter state for saving/loading
        self.other_parameter_name = ["item_cnt", "max_cnt"]

    def forward(self, item_seq, item_seq_len):
        """
        Not used for this model, but required by interface.
        """
        batch_size = item_seq.size(0)
        return torch.zeros(batch_size, self.n_items, device=self.device)

    def calculate_loss(self, interaction):
        """
        During training, count item occurrences to build popularity statistics.

        Note: For sequential data, we count the target items.
        Sequences are not available during training (only during evaluation).
        """
        # Count target items (the items being predicted)
        pos_items = interaction[self.POS_ITEM_ID]
        self.item_cnt[pos_items] += 1

        # Update max count
        self.max_cnt = torch.max(self.item_cnt, dim=0)[0]

        # Return fake loss
        return torch.nn.Parameter(torch.zeros(1)).to(self.device)

    def predict(self, interaction):
        """
        Pointwise prediction: return popularity score for specific items.
        """
        item = interaction[self.POS_ITEM_ID]

        # Handle case when model hasn't been trained yet
        if self.max_cnt is None or self.max_cnt == 0:
            return torch.zeros(len(item), device=self.device, dtype=torch.float)

        result = torch.true_divide(self.item_cnt[item], self.max_cnt)
        return result.squeeze(-1)

    def full_sort_predict(self, interaction):
        """
        Full sort prediction: return popularity scores for all items.

        Returns a [batch_size, n_items] tensor where each row contains
        the same popularity scores for all items.
        """
        batch_user_num = interaction[self.USER_ID].shape[0]

        # Compute popularity scores for all items
        result = self.item_cnt.to(torch.float64) / self.max_cnt.to(torch.float64)

        # Repeat for each user in the batch
        result = torch.repeat_interleave(result.unsqueeze(0), batch_user_num, dim=0)

        return result.view(-1)
