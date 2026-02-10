import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType
from torch_scatter import scatter



class DCCF(GeneralRecommender):
    """Disentangled Contrastive Collaborative Filtering in RecBole."""
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.n_intents = config['n_intents']
        self.temp = config['temp']
        self.emb_reg = config['emb_reg']
        self.cen_reg = config['cen_reg']
        self.ssl_reg = config['ssl_reg']

        # --- build normalized adjacency (torch only) ---
        self.adj_index, self.adj_weight = self._build_sparse_adj(dataset)

        # --- model parameters ---
        self.user_embedding = nn.Embedding(self.n_users, self.emb_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.emb_dim)
        # prototype embeddings: [K, d]
        self.user_intent = nn.Parameter(torch.empty(self.n_intents, self.emb_dim))
        self.item_intent = nn.Parameter(torch.empty(self.n_intents, self.emb_dim))

        # weight init
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)
        nn.init.xavier_normal_(self.user_intent)
        nn.init.xavier_normal_(self.item_intent)

        # cache for inference
        self._cached_epoch = -1
        self.ua_embedding = None
        self.ia_embedding = None

    def _build_sparse_adj(self, dataset):
        """Build raw and then normalized adjacency in pure torch."""
        uid = torch.LongTensor(dataset.inter_feat[dataset.uid_field].numpy()).to(self.device)
        iid = torch.LongTensor(dataset.inter_feat[dataset.iid_field].numpy()).to(self.device) + self.n_users
        # bi‐directional edges
        row = torch.cat([uid, iid], dim=0)
        col = torch.cat([iid, uid], dim=0)
        # raw weights = 1
        weight = torch.ones_like(row, dtype=torch.float, device=self.device)
        N = self.n_users + self.n_items

        # 1) compute raw degrees: deg[u] = sum_{(u,*)} weigh
        deg_row = scatter(src=weight, index=row, dim=0, dim_size=N, reduce='sum')

        # 2) compute inv sqrt
        deg_inv_sqrt = deg_row.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.

        # 3) normalize weights: w' = w * d^{-1/2}[row] * d^{-1/2}[col]
        norm_weight = weight * deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # stack index and weight for spmm
        index = torch.stack([row, col], dim=0)
        return index, norm_weight

    def _adaptive_mask(self, head, tail, raw_index, raw_weight):
        """
        Mask the *raw* adjacency (not yet normalized), re-normalize per Eq. 11–12 :contentReference[oaicite:7]{index=7}.
        head, tail = node embeddings at indices raw_index[:,0], raw_index[:,1]
        """
        sim = (F.normalize(head) * F.normalize(tail)).sum(dim=-1)        # cosine similarity
        mask = (sim + 1.) * 0.5                                         # [0,1]
        # apply to raw weight
        w_masked = raw_weight * mask

        # new degree = sum over masked edges
        N = self.n_users + self.n_items
        deg_new = torch_sparse.sum(w_masked, raw_index[0], dim=N)
        inv_deg = deg_new.pow(-1)
        inv_deg[inv_deg == float('inf')] = 0.

        # re-renormalize: w'' = w_masked * inv_deg[u]
        w_final = w_masked * inv_deg[raw_index[0]]
        return raw_index, w_final

    def inference(self, cur_epoch=None):
        """
        Multi-layer propagation + intent aggregation + adaptive contrastive masks.
        Cached once per epoch for efficiency :contentReference[oaicite:8]{index=8}.
        """
        # only recompute if epoch advanced
        if cur_epoch == self._cached_epoch and self.ua_embedding is not None:
            return

        index, weight = self.adj_index, self.adj_weight
        N = self.n_users + self.n_items
        all_emb = torch.cat([self.user_embedding.weight,
                             self.item_embedding.weight], dim=0)
        embs = [all_emb]
        g_list, r_list, b_list, g2_list = [], [], [], []

        for _ in range(self.n_layers):
            # 1) GNN prop over normalized A
            z = torch_sparse.spmm(index, weight, N, N, embs[-1])

            # 2) Disentangled global intent
            u_e, i_e = torch.split(embs[-1], [self.n_users, self.n_items], 0)
            u_att = F.softmax(u_e @ self.user_intent.T, dim=1)
            i_att = F.softmax(i_e @ self.item_intent.T, dim=1)
            r = torch.cat([u_att @ self.user_intent,
                           i_att @ self.item_intent], dim=0)

            # 3a) local adaptive augment (GAA)
            head_z, tail_z = z[index[0]], z[index[1]]
            idx1, w1 = self._adaptive_mask(head_z, tail_z, index, weight)

            h_beta = torch_sparse.spmm(idx1, w1, N, N, embs[-1])

            # 3b) global adaptive augment (IAA)
            head_r, tail_r = r[index[0]], r[index[1]]
            idx2, w2 = self._adaptive_mask(head_r, tail_r, index, weight)

            h_gamma = torch_sparse.spmm(idx2, w2, N, N, embs[-1])

            # collect
            g_list.append(z); r_list.append(r)
            b_list.append(h_beta); g2_list.append(h_gamma)

            # aggregate
            out = embs[-1] + z + r + h_beta + h_gamma
            embs.append(out)

        # final embeddings = sum over layers
        res = torch.stack(embs, dim=1).sum(dim=1)
        self.ua_embedding, self.ia_embedding = torch.split(res, [self.n_users, self.n_items], 0)
        self._cached_epoch = cur_epoch

        # store all views for SSL
        self._g_list, self._r_list, self._b_list, self._g2_list = g_list, r_list, b_list, g2_list

    def cal_ssl_loss(self, users, pos_items):
        """
        Compute layer-wise InfoNCE across (z, r, b, g2) views per Eq. 14–15.
        """
        u_idx = torch.unique(users)
        i_idx = torch.unique(pos_items) + self.n_users
        total = 0.
        for z, r, b, g2 in zip(self._g_list, self._r_list, self._b_list, self._g2_list):
            # normalize
            z_u, r_u = F.normalize(z[u_idx]), F.normalize(r[u_idx])
            b_u, g2_u = F.normalize(b[u_idx]), F.normalize(g2[u_idx])
            z_i, r_i = F.normalize(z[i_idx]), F.normalize(r[i_idx])
            b_i, g2_i = F.normalize(b[i_idx]), F.normalize(g2[i_idx])

            def nce(x,y):
                pos = torch.exp((x*y).sum(-1)/self.temp)
                neg = torch.sum(torch.exp(x@y.T/self.temp), dim=1)
                return torch.mean(-torch.log(pos/(neg+1e-8)))

            # users: 3 components; items: 3 components
            total += nce(z_u, r_u) + nce(z_u, b_u) + nce(z_u, g2_u)
            total += nce(z_i, r_i) + nce(z_i, b_i) + nce(z_i, g2_i)
        return total

    def forward(self, user, pos_item, neg_item):
        # ensure embeddings are up-to-date
        self.inference(cur_epoch=self.cur_epoch)
        u_e = self.ua_embedding[user]
        p_e = self.ia_embedding[pos_item]
        n_e = self.ia_embedding[neg_item]
        pos_score = (u_e * p_e).sum(-1)
        neg_score = (u_e * n_e).sum(-1)
        return pos_score, neg_score

    def calculate_loss(self, interaction):
        u = interaction[self.USER_ID]
        i = interaction[self.ITEM_ID]
        j = interaction[self.NEG_ITEM_ID]

        pos, neg = self.forward(u, i, j)
        # BPR loss
        bpr = torch.mean(F.softplus(neg - pos))

        # embedding ℓ2 reg on batch
        ue, pe, ne = self.user_embedding(u), self.item_embedding(i), self.item_embedding(j)
        emb_loss = self.emb_reg * (ue.norm()**2 + pe.norm()**2 + ne.norm()**2)

        # prototype ℓ2 reg
        cen_loss = self.cen_reg * (self.user_intent.norm()**2 + self.item_intent.norm()**2)

        # SSL
        ssl = self.ssl_reg * self.cal_ssl_loss(u, i)

        return bpr + emb_loss + cen_loss + ssl

    def predict(self, interaction):
        u = interaction[self.USER_ID]
        i = interaction[self.ITEM_ID]
        self.inference(cur_epoch=self.cur_epoch)
        return (self.ua_embedding[u] * self.ia_embedding[i]).sum(-1)

    def full_sort_predict(self, interaction):
        u = interaction[self.USER_ID]
        self.inference(cur_epoch=self.cur_epoch)
        return self.ua_embedding[u] @ self.ia_embedding.T
