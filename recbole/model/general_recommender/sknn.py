import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType


class SKNN(GeneralRecommender):
    """
    A RecBole-compatible session-based KNN model
    adapted from your ContextKNN class.

    We treat the 'session_id' as if it were the user,
    and the 'item_id' as the current item the user just clicked.

    Then at prediction time (pointwise or full-sort),
    we find neighbors of the session by:
      1) retrieving the session's itemset
      2) using your chosen similarity metric
      3) scoring items from the neighbors

    Because RecBole calls (user, item) but does not
    pass the entire session sequence, we have a simpler
    approach that uses only the single item as the
    'current item'. If you truly want the entire session's
    item sequence, you'd need to track it externally or
    do a custom pipeline.
    """

    input_type = InputType.POINTWISE  # or PAIRWISE if you prefer BPR style

    def __init__(self, config, dataset):
        super(SKNN, self).__init__(config, dataset)

        # ---------- read config ----------
        self.k = config["knn_k"] if "knn_k" in config else 100
        self.sample_size = config["knn_sample_size"] if "knn_sample_size" in config else 1000
        self.sampling = config["knn_sampling"] if "knn_sampling" in config else "recent"
        self.similarity = config["knn_similarity"] if "knn_similarity" in config else "jaccard"
        self.remind = config["knn_remind"] if "knn_remind" in config else False
        self.pop_boost = config["knn_pop_boost"] if "knn_pop_boost" in config else 0
        self.extend = config["knn_extend"] if "knn_extend" in config else False
        self.normalize = config["knn_normalize"] if "knn_normalize" in config else True

        # We'll treat the dataset's user ID field as the session ID
        # and the item ID field as the item ID
        self.SESSION_ID = dataset.uid_field
        self.ITEM_ID = dataset.iid_field
        self.TIME_KEY = dataset.time_field  # might be 'timestamp' or so

        # ---------- internal structures ----------
        self.session_item_map = dict()
        self.item_session_map = dict()
        self.session_time = dict()

        # Because we are memory-based, we do not have trainable params, but
        # we add a dummy param so RecBole won't crash building an optimizer.
        self.dummy_param = nn.Parameter(torch.zeros(1, requires_grad=True))

        # ---------- build the session data structures ----------
        self._build_data_structures(dataset)

    def _build_data_structures(self, dataset):
        """
        Build session→items, item→sessions, and session→time from all interactions in dataset.inter_feat.
        """
        inter_feat = dataset.inter_feat
        # We'll convert them to numpy for iteration
        sessions_np = inter_feat[self.SESSION_ID].numpy()
        items_np = inter_feat[self.ITEM_ID].numpy()
        # for time:
        if self.TIME_KEY in inter_feat:
            times_np = inter_feat[self.TIME_KEY].numpy()
        else:
            # fallback
            times_np = np.arange(len(sessions_np))

        current_session = None
        session_items_set = set()
        current_time = -1

        last_session_idx = None

        # We'll sort by session/time so we can group
        # but often dataset is already reindexed. We can do it in a single pass if guaranteed order.
        # For safety, let's do a stable sort by session & time:
        # (If your dataset is guaranteed to be sorted, skip this.)
        # We'll do:
        # can't do it easily if it's large, but let's do a small demonstration
        # Create a combined array
        arr = np.rec.fromarrays((sessions_np, items_np, times_np))
        # sort by session, then time
        # if the dataset is guaranteed sorted by time, skip this step
        arr.sort(order=('f0','f2'))  # f0->session, f2->time
        sessions_np = arr.f0
        items_np    = arr.f1
        times_np    = arr.f2

        for s, i, t in zip(sessions_np, items_np, times_np):
            if s != current_session:
                # store the old session
                if current_session is not None and len(session_items_set) > 0:
                    self.session_item_map[current_session] = session_items_set
                    self.session_time[current_session] = current_time

                current_session = s
                session_items_set = set()
            session_items_set.add(i)
            current_time = t

            # item→session
            if i not in self.item_session_map:
                self.item_session_map[i] = set()
            self.item_session_map[i].add(s)

        # store the last session
        if current_session is not None and len(session_items_set) > 0:
            self.session_item_map[current_session] = session_items_set
            self.session_time[current_session] = current_time

    def calculate_loss(self, interaction):
        """
        Memory-based => no real training. Return zero so RecBole won't error out.
        """
        return torch.zeros(1, device=self.device, requires_grad=True)

    def forward(self, user, item):
        """
        For RecBole usage, we'll interpret `user` as session_id, `item` as current item.
        We'll just call `_predict_internal` which returns a single score. For pointwise usage,
        we might do that. But typically we need a vector of scores.
        We'll produce a single score as if we are checking how likely is 'item' the next item
        after the session? This is not entirely standard, but we do it to fit RecBole's interface.
        """
        # We'll do a dummy single score. Because in session-based next-item, we normally
        # rank all items. We'll see if we can produce that. For pointwise we just produce one.
        scores = self._predict_internal(user, item)
        return scores

    def predict(self, interaction):
        """
        For pointwise test. Called with (session, item). We'll produce a single numeric score
        indicating how likely 'item' is next. We'll do the KNN approach to get item scores
        for all candidate items, then pick out the score of 'item'.
        """
        batch_sessions = interaction[self.SESSION_ID]
        batch_items = interaction[self.ITEM_ID]

        # We'll gather result in a tensor of shape (batch,)
        out_scores = []
        for s_id, i_id in zip(batch_sessions.tolist(), batch_items.tolist()):
            # Score for i_id
            full_scores = self._compute_knn_scores(s_id, i_id)
            score_iid = full_scores.get(i_id, 0.0)
            out_scores.append(score_iid)

        return torch.tensor(out_scores, device=batch_sessions.device, dtype=torch.float)

    def full_sort_predict(self, interaction):
        """
        For a user (session) or a batch of them, produce a score for *every item*.
        RecBole uses this for ranking-based evaluation.

        We'll do the KNN approach for each session in the batch. We'll produce
        a [batch_size, n_items] matrix of scores.
        """
        batch_sessions = interaction[self.SESSION_ID]
        # The item field is not necessarily used in full sort. We can glean the "current item" from that if needed,
        # or we can pick the last item from that session's item set if the dataset includes partial info. We'll
        # do a simpler approach: for each session, pick the last item from that session's set as "current item."
        # If you want the actual "current item," you'd have to store a partial sequence externally.
        # We'll produce a matrix [batch_size, n_items].
        n_items = self.n_items
        all_item_ids = torch.arange(n_items, device=batch_sessions.device)

        all_scores = []
        for s_id in batch_sessions.tolist():
            # pick "current item" as an arbitrary item from the session's item set?
            # We'll use the max item ID or something. Or we can skip and do a multi-neighbor approach?
            session_items = self.session_item_map.get(s_id, set())
            if len(session_items) > 0:
                current_item = next(iter(session_items))  # just pick one
            else:
                # session not found or empty
                # fallback
                current_item = 0

            knn_scores = self._compute_knn_scores(s_id, current_item)
            # convert to array of shape (n_items,)
            row_score = np.zeros(n_items, dtype=np.float32)
            for it, sc in knn_scores.items():
                if it < n_items:  # in case IDs are reindexed 0..n_items-1
                    row_score[it] = sc
            if self.normalize:
                mx = row_score.max()
                if mx > 0:
                    row_score /= mx
            all_scores.append(row_score)

        all_scores = np.array(all_scores, dtype=np.float32)
        # convert to torch
        all_scores = torch.from_numpy(all_scores).to(batch_sessions.device)
        return all_scores

    # ----------------------------------------------------
    # Below are the core KNN logic, adapted from your code
    # ----------------------------------------------------
    def _compute_knn_scores(self, session_id, current_item_id):
        """
        Our internal function that replicates `predict_next` logic: find neighbors, score items,
        optionally remind/popup. We'll do a simplified approach that uses just the single item
        as 'input_item_id' in your original code.
        """
        session_items = self.session_item_map.get(session_id, set())

        # 1) find neighbors
        neighbors = self._find_neighbors(session_items, current_item_id, session_id)
        # 2) score items
        scores = self._score_items(neighbors)

        # 3) remind
        if self.remind and len(session_items) > 0:
            # do a small topN approach
            reminderScore = 5.0
            takeLastN = min(3, len(session_items))
            # if session_items is a set, let's turn it into a list
            session_list = list(session_items)
            # we'll do the last "takeLastN" items from that list
            for idx, elem in enumerate(session_list[-takeLastN:], start=1):
                oldScore = scores.get(elem, 0.0)
                newScore = oldScore + reminderScore
                # small boost for recency
                newScore = newScore * reminderScore + (idx / 100.0)
                scores[elem] = newScore

        # 4) pop boost
        if self.pop_boost > 0:
            pop_map = self._item_pop(neighbors)
            for it in scores:
                # add pop
                pval = pop_map.get(it, 0.0)
                scores[it] = scores[it] + self.pop_boost * pval

        if self.normalize and len(scores) > 0:
            max_val = max(scores.values())
            if max_val > 0:
                for it in scores:
                    scores[it] /= max_val

        return scores

    def _item_pop(self, neighbors):
        """
        Equivalent to item_pop in your code
        """
        result = {}
        max_pop = 0
        for (s, sim) in neighbors:
            its = self.session_item_map.get(s, set())
            for item in its:
                cnt = result.get(item, 0)
                new_cnt = cnt + 1
                result[item] = new_cnt
                if new_cnt > max_pop:
                    max_pop = new_cnt
        if max_pop > 0:
            for it in result:
                result[it] = result[it] / max_pop
        return result

    def _find_neighbors(self, session_items, current_item_id, session_id):
        """
        Based on your find_neighbors, we do possible neighbor sessions, then compute similarity,
        then sort top-k.
        """
        possible_neigh = self._possible_neighbor_sessions(session_items, current_item_id, session_id)
        # compute similarity
        neighbors = self._calc_similarity(session_items, possible_neigh)
        # sort
        neighbors.sort(key=lambda x: x[1], reverse=True)
        if len(neighbors) > self.k:
            neighbors = neighbors[: self.k]
        return neighbors

    def _possible_neighbor_sessions(self, session_items, current_item_id, session_id):
        """
        Gathers the sessions that have the current_item_id or overlap with session_items,
        then samples up to sample_size. 
        This is a simplified approach. 
        """
        # union of sessions for the input item
        relevant = set()
        if current_item_id in self.item_session_map:
            relevant |= self.item_session_map[current_item_id]

        # For a more robust approach, you might union all sessions for each item in session_items
        # if you want to handle multiple items as context. For now, we only use current_item.
        # If you want to use the entire session, do something like:
        # for it in session_items:
        #     relevant |= self.item_session_map.get(it, set())

        if self.sample_size == 0 or len(relevant) <= self.sample_size:
            return relevant
        else:
            # sampling
            if self.sampling == "recent":
                return self._most_recent_sessions(relevant, self.sample_size)
            else:
                # random
                import random
                return set(random.sample(relevant, self.sample_size))

    def _most_recent_sessions(self, sessions, number):
        """
        Return the 'number' most recent sessions by session_time
        """
        # gather (session, time)
        tmp = []
        for s in sessions:
            t = self.session_time.get(s, 0)
            tmp.append((s, t))
        # sort descending by time
        tmp.sort(key=lambda x: x[1], reverse=True)
        subset = tmp[:number]
        return set(x[0] for x in subset)

    def _calc_similarity(self, session_items, session_ids):
        """
        For each session in session_ids, compute similarity with 'session_items'
        using self.similarity.
        Return a list of (session_id, sim).
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
        Depending on self.similarity, compute jaccard, cosine, etc.
        """
        if self.similarity == "jaccard":
            return self._jaccard(first, second)
        elif self.similarity == "cosine":
            return self._cosine(first, second)
        elif self.similarity == "tanimoto":
            return self._tanimoto(first, second)
        elif self.similarity == "binary":
            return self._binary(first, second)
        elif self.similarity == "random":
            import random
            return random.random()
        else:
            # fallback jaccard
            return self._jaccard(first, second)

    def _jaccard(self, first, second):
        inter = len(first & second)
        union = len(first | second)
        if union == 0:
            return 0.0
        return inter / union

    def _cosine(self, first, second):
        inter = len(first & second)
        denom = (len(first) * len(second)) ** 0.5
        if denom == 0:
            return 0.0
        return inter / denom

    def _tanimoto(self, first, second):
        inter = len(first & second)
        la = len(first)
        lb = len(second)
        denom = (la + lb - inter)
        if denom == 0:
            return 0.0
        return inter / denom

    def _binary(self, first, second):
        inter = len(first & second)
        la = len(first)
        lb = len(second)
        numerator = 2 * inter
        denom = (2 * inter) + la + lb
        if denom == 0:
            return 0.0
        return numerator / denom

    def _score_items(self, neighbors):
        """
        Sums the similarity over items in neighbor sessions.
        Returns a dict: item -> score
        """
        scores = {}
        for (sess, sim) in neighbors:
            items_of_sess = self.session_item_map.get(sess, set())
            for it in items_of_sess:
                old = scores.get(it, 0.0)
                scores[it] = old + sim
        return scores
