import torch
import torch.nn as nn
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType


class ItemBasedCF(GeneralRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # Configurable parameters
        self.topk       = config['top_k']       if 'top_k' in config else 50
        self.shrink     = config['shrink']      if 'shrink' in config else 0.0
        self.block_size = config['block_size']  if 'block_size' in config else 100

        # Field names
        self.USER_ID = dataset.uid_field
        self.ITEM_ID = dataset.iid_field
        self.LABEL   = dataset.label_field

        # 1) Build sparse COO, then convert ONCE to dense
        coo = dataset.inter_matrix(form='coo')
        vals   = torch.tensor(coo.data, dtype=torch.float, device=self.device)
        row_idx = torch.tensor(coo.row, dtype=torch.long, device=self.device)
        col_idx = torch.tensor(coo.col, dtype=torch.long, device=self.device)
        idx     = torch.stack([row_idx, col_idx], dim=0)
        # idx    = torch.tensor([coo.row, coo.col], dtype=torch.long, device=self.device)
        sparse_R = torch.sparse_coo_tensor(
            idx, vals,
            size=(dataset.num(self.USER_ID), dataset.num(self.ITEM_ID)),
            device=self.device
        ).coalesce()
        # now a dense user×item matrix
        self.R_dense = sparse_R.to_dense()
        self.num_users, self.num_items = self.R_dense.shape

        # 2) Precompute item L2 norms
        sq_vals = vals * vals
        sq = torch.sparse_coo_tensor(idx, sq_vals,
                                     size=sparse_R.shape,
                                     device=self.device).coalesce()
        col_sum = torch.sparse.sum(sq, dim=0).to_dense()      # [num_items]
        self.item_norms = torch.sqrt(col_sum + 1e-8)           # [num_items]

        # 3) Blockwise adjusted-cosine similarity → sparse W
        rows, cols, sims = [], [], []
        for start in range(0, self.num_items, self.block_size):
            end    = min(start + self.block_size, self.num_items)
            block  = self.R_dense[:, start:end]                # [U, block_size]
            # raw dot: [I, U] @ [U, block_size] → [I, block_size]
            raw    = self.R_dense.transpose(0,1).matmul(block)
            norm_b = self.item_norms[start:end]                # [block_size]
            denom  = (self.item_norms.unsqueeze(1) * norm_b.unsqueeze(0)) \
                     + self.shrink + 1e-8
            sim_b  = raw / denom                               # [I, block_size]

            # extract topk for each column in this block
            for b in range(end - start):
                i = start + b                                  # target item
                col_sim = sim_b[:, b]                          # similarities to all items
                _, idx_k = torch.topk(col_sim.abs(), self.topk, sorted=False)
                sim_k    = col_sim[idx_k]
                rows.append(torch.full((self.topk,), i,
                                       dtype=torch.long,
                                       device=self.device))
                cols.append(idx_k)
                sims.append(sim_k)

        rows = torch.cat(rows)
        cols = torch.cat(cols)
        sims = torch.cat(sims)
        w_idx = torch.stack([rows, cols], dim=0)
        self.W = torch.sparse_coo_tensor(
            w_idx, sims,
            size=(self.num_items, self.num_items),
            device=self.device
        ).coalesce()

        # 4) Precompute full prediction matrix: [U×I] = R_dense @ W_dense
        self.pred_mat = self.R_dense.matmul(self.W.to_dense())

    def calculate_loss(self, interaction):
        # No training for memory-based CF
        return torch.zeros((), device=self.device, requires_grad=True)

    def forward(self, interaction):
        return self.calculate_loss(interaction)

    def predict(self, interaction):
        users = interaction[self.USER_ID].tolist()
        items = interaction[self.ITEM_ID].tolist()
        scores = [self.pred_mat[u, i] for u, i in zip(users, items)]
        return torch.tensor(scores, device=self.device)

    def full_sort_predict(self, interaction):
        users = interaction[self.USER_ID].tolist()
        all_scores = [self.pred_mat[u, :] for u in users]
        return torch.stack(all_scores, dim=0)
