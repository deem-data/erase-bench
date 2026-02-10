# -*- coding: utf-8 -*-
"""
SKNN_SBR (Session-based K-Nearest Neighbors for Sequential Recommendation)
################################################

A session-based KNN baseline model adapted for sequential recommendation tasks.
This model finds similar sessions based on item overlap and recommends items
from the most similar sessions.

Reference:
    Based on SKNN approach adapted for RecBole's SequentialRecommender framework.
"""

import torch
import torch.nn as nn
import numpy as np

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType


class SKNN_SBR(SequentialRecommender):
    """
    Session-based K-Nearest Neighbors model for sequential recommendation.

    This model treats each sequence as a session and uses KNN-based similarity
    to find similar sessions, then scores items based on their appearance in
    similar sessions.
    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(SKNN_SBR, self).__init__(config, dataset)

        # Read config parameters
        self.k = config.get("knn_k", 100)
        self.sample_size = config.get("knn_sample_size", 1000)
        self.sampling = config.get("knn_sampling", "recent")
        self.similarity = config.get("knn_similarity", "jaccard")
        self.remind = config.get("knn_remind", False)
        self.pop_boost = config.get("knn_pop_boost", 0)
        self.normalize = config.get("knn_normalize", True)

        # Internal data structures for session-based KNN
        self.session_item_map = {}  # session_id -> set of items
        self.item_session_map = {}  # item_id -> set of sessions
        self.session_time = {}      # session_id -> timestamp

        # Dummy parameter for optimizer (memory-based model has no trainable params)
        self.dummy_param = nn.Parameter(torch.zeros(1, requires_grad=True))

        # Build the KNN data structures from the dataset
        self._build_data_structures(dataset)

    def _build_data_structures(self, dataset):
        """
        Build session→items, item→sessions, and session→time mappings
        from all interactions in the dataset.
        """
        inter_feat = dataset.inter_feat

        # Get the interaction data
        sessions_np = inter_feat[self.USER_ID].cpu().numpy()
        item_seqs = inter_feat[self.ITEM_SEQ].cpu().numpy()
        item_seq_lens = inter_feat[self.ITEM_SEQ_LEN].cpu().numpy()

        # Get timestamps if available
        if dataset.time_field in inter_feat:
            times_np = inter_feat[dataset.time_field].cpu().numpy()
        else:
            times_np = np.arange(len(sessions_np))

        # Build the mappings
        for idx, (session_id, item_seq, seq_len, time) in enumerate(
            zip(sessions_np, item_seqs, item_seq_lens, times_np)
        ):
            # Get the actual items in this sequence (excluding padding)
            items = set(item_seq[:seq_len].tolist()) - {0}  # Remove padding (0)

            if len(items) == 0:
                continue

            # Store session -> items mapping
            if session_id not in self.session_item_map:
                self.session_item_map[session_id] = items
                self.session_time[session_id] = time
            else:
                # Merge items if session appears multiple times
                self.session_item_map[session_id] |= items
                self.session_time[session_id] = max(self.session_time[session_id], time)

            # Build item -> sessions mapping
            for item in items:
                if item not in self.item_session_map:
                    self.item_session_map[item] = set()
                self.item_session_map[item].add(session_id)

    def calculate_loss(self, interaction):
        """
        Memory-based model has no training phase.
        Return a small loss to satisfy RecBole's training loop.
        """
        return torch.zeros(1, device=self.device, requires_grad=True)

    def forward(self, item_seq, item_seq_len):
        """
        Not used for memory-based models, but required by the interface.
        """
        batch_size = item_seq.size(0)
        return torch.zeros(batch_size, self.n_items, device=self.device)

    def predict(self, interaction):
        """
        Pointwise prediction: predict score for specific user-item pairs.
        """
        batch_sessions = interaction[self.USER_ID]
        batch_items = interaction[self.POS_ITEM_ID]

        # Check if sequences are available (not available during training/FLOPS calculation)
        if self.ITEM_SEQ not in interaction:
            # Return zeros during FLOPS calculation
            return torch.zeros(len(batch_sessions), device=batch_sessions.device, dtype=torch.float)

        batch_seq = interaction[self.ITEM_SEQ]
        batch_seq_len = interaction[self.ITEM_SEQ_LEN]

        out_scores = []
        for session_id, target_item, seq, seq_len in zip(
            batch_sessions.tolist(),
            batch_items.tolist(),
            batch_seq.tolist(),
            batch_seq_len.tolist()
        ):
            # Get current session items
            session_items = set(seq[:seq_len]) - {0}  # Remove padding

            # Compute KNN scores
            knn_scores = self._compute_knn_scores(session_id, session_items)

            # Get score for target item
            score = knn_scores.get(target_item, 0.0)
            out_scores.append(score)

        return torch.tensor(out_scores, device=batch_sessions.device, dtype=torch.float)

    def full_sort_predict(self, interaction):
        """
        Full sort prediction: score all items for each session.
        Returns a [batch_size, n_items] tensor of scores.
        """
        batch_sessions = interaction[self.USER_ID]
        batch_seq = interaction[self.ITEM_SEQ]
        batch_seq_len = interaction[self.ITEM_SEQ_LEN]

        all_scores = []
        for session_id, seq, seq_len in zip(
            batch_sessions.tolist(),
            batch_seq.tolist(),
            batch_seq_len.tolist()
        ):
            # Get current session items
            session_items = set(seq[:seq_len]) - {0}  # Remove padding

            # Compute KNN scores for all items
            knn_scores = self._compute_knn_scores(session_id, session_items)

            # Convert to array of shape (n_items,)
            row_score = np.zeros(self.n_items, dtype=np.float32)
            for item_id, score in knn_scores.items():
                if 0 <= item_id < self.n_items:
                    row_score[item_id] = score

            all_scores.append(row_score)

        all_scores = np.array(all_scores, dtype=np.float32)
        return torch.from_numpy(all_scores).to(batch_sessions.device)

    def _compute_knn_scores(self, session_id, session_items):
        """
        Core KNN logic: find similar sessions and score items.

        Args:
            session_id: Current session ID
            session_items: Set of items in the current session

        Returns:
            Dictionary mapping item_id -> score
        """
        if len(session_items) == 0:
            return {}

        # 1) Find neighbor sessions
        neighbors = self._find_neighbors(session_items, session_id)

        # 2) Score items based on neighbors
        scores = self._score_items(neighbors)

        # 3) Apply remind boost (boost recently seen items in the session)
        if self.remind and len(session_items) > 0:
            remind_score = 5.0
            take_last_n = min(3, len(session_items))
            session_list = list(session_items)

            for idx, item in enumerate(session_list[-take_last_n:], start=1):
                old_score = scores.get(item, 0.0)
                new_score = old_score + remind_score
                new_score = new_score * remind_score + (idx / 100.0)
                scores[item] = new_score

        # 4) Apply popularity boost
        if self.pop_boost > 0:
            pop_map = self._item_pop(neighbors)
            for item in scores:
                pop_val = pop_map.get(item, 0.0)
                scores[item] = scores[item] + self.pop_boost * pop_val

        # 5) Normalize scores
        if self.normalize and len(scores) > 0:
            max_val = max(scores.values())
            if max_val > 0:
                for item in scores:
                    scores[item] /= max_val

        return scores

    def _find_neighbors(self, session_items, session_id):
        """
        Find k most similar sessions based on item overlap.

        Returns:
            List of (session_id, similarity_score) tuples
        """
        # Get possible neighbor sessions
        possible_neighbors = self._possible_neighbor_sessions(session_items, session_id)

        # Calculate similarity with each neighbor
        neighbors = self._calc_similarity(session_items, possible_neighbors)

        # Sort by similarity and keep top k
        neighbors.sort(key=lambda x: x[1], reverse=True)
        if len(neighbors) > self.k:
            neighbors = neighbors[:self.k]

        return neighbors

    def _possible_neighbor_sessions(self, session_items, session_id):
        """
        Find candidate neighbor sessions that share items with current session.
        """
        relevant = set()

        # Union all sessions that contain any item from current session
        for item in session_items:
            if item in self.item_session_map:
                relevant |= self.item_session_map[item]

        # Remove self
        relevant.discard(session_id)

        # Sample if too many candidates
        if self.sample_size == 0 or len(relevant) <= self.sample_size:
            return relevant
        else:
            if self.sampling == "recent":
                return self._most_recent_sessions(relevant, self.sample_size)
            else:
                import random
                return set(random.sample(list(relevant), self.sample_size))

    def _most_recent_sessions(self, sessions, number):
        """
        Return the 'number' most recent sessions by timestamp.
        """
        tmp = [(s, self.session_time.get(s, 0)) for s in sessions]
        tmp.sort(key=lambda x: x[1], reverse=True)
        subset = tmp[:number]
        return set(x[0] for x in subset)

    def _calc_similarity(self, session_items, session_ids):
        """
        Calculate similarity between current session and candidate sessions.

        Returns:
            List of (session_id, similarity) tuples
        """
        out = []
        for s in session_ids:
            other_items = self.session_item_map.get(s, set())
            sim = self._compute_sim(session_items, other_items)
            if sim > 0:
                out.append((s, sim))
        return out

    def _compute_sim(self, first, second):
        """
        Compute similarity between two item sets.
        """
        if self.similarity == "jaccard":
            return self._jaccard(first, second)
        elif self.similarity == "cosine":
            return self._cosine(first, second)
        elif self.similarity == "tanimoto":
            return self._tanimoto(first, second)
        elif self.similarity == "binary":
            return self._binary(first, second)
        else:
            return self._jaccard(first, second)

    def _jaccard(self, first, second):
        """Jaccard similarity: |intersection| / |union|"""
        inter = len(first & second)
        union = len(first | second)
        return inter / union if union > 0 else 0.0

    def _cosine(self, first, second):
        """Cosine similarity for sets"""
        inter = len(first & second)
        denom = (len(first) * len(second)) ** 0.5
        return inter / denom if denom > 0 else 0.0

    def _tanimoto(self, first, second):
        """Tanimoto coefficient"""
        inter = len(first & second)
        denom = len(first) + len(second) - inter
        return inter / denom if denom > 0 else 0.0

    def _binary(self, first, second):
        """Binary similarity (Dice coefficient)"""
        inter = len(first & second)
        denom = len(first) + len(second)
        return (2 * inter) / denom if denom > 0 else 0.0

    def _score_items(self, neighbors):
        """
        Aggregate scores from neighbor sessions.
        Items are scored by the sum of similarities of sessions containing them.

        Returns:
            Dictionary mapping item_id -> score
        """
        scores = {}
        for session_id, similarity in neighbors:
            items = self.session_item_map.get(session_id, set())
            for item in items:
                scores[item] = scores.get(item, 0.0) + similarity
        return scores

    def _item_pop(self, neighbors):
        """
        Calculate normalized item popularity within neighbor sessions.

        Returns:
            Dictionary mapping item_id -> normalized_popularity
        """
        result = {}
        max_pop = 0

        for session_id, _ in neighbors:
            items = self.session_item_map.get(session_id, set())
            for item in items:
                cnt = result.get(item, 0) + 1
                result[item] = cnt
                if cnt > max_pop:
                    max_pop = cnt

        # Normalize
        if max_pop > 0:
            for item in result:
                result[item] = result[item] / max_pop

        return result
