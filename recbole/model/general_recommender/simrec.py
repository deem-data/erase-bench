import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType
from recbole.model.loss import BPRLoss


class LightGCNTeacher(nn.Module):
    r"""
    Minimal LightGCN-like teacher network that:
      - Has user/item embeddings
      - Applies L layers of light graph convolution
    """
    def __init__(self, n_users, n_items, emb_size, n_layers, reg_weight=1e-5):
        super(LightGCNTeacher, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.reg_weight = reg_weight

        # User/item embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.emb_size)
        self.item_embedding = nn.Embedding(self.n_items, self.emb_size)
        # xavier init or uniform init
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Because we need adjacency for LightGCN, we rely on being passed an adjacency (or the data).
        # We'll store that externally (see forward).
        # The adjacency is typically in "normalized" form for LightGCN.

    def forward(self, norm_adj):
        r"""
        norm_adj: shape (n_users + n_items, n_users + n_items), often a sparse FloatTensor
        Returns:
            all_user_emb: shape (n_users, emb_size)
            all_item_emb: shape (n_items, emb_size)
        """
        # 1) Start with zero-layer embeddings
        user_emb_0 = self.user_embedding.weight
        item_emb_0 = self.item_embedding.weight
        all_emb = torch.cat([user_emb_0, item_emb_0], dim=0)

        # 2) LightGCN Propagation
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(norm_adj, all_emb)
            embs.append(all_emb)
        # 3) Sum (or average) the embeddings from each layer
        # Paper uses sum in LightGCN
        final_emb = torch.stack(embs, dim=0).mean(dim=0)  # or .sum(dim=0)

        all_user_emb = final_emb[: self.n_users]
        all_item_emb = final_emb[self.n_users :]
        return all_user_emb, all_item_emb

    def calculate_bpr_loss(self, norm_adj, users, pos_items, neg_items):
        r"""
        For pairwise training on the teacher itself.
        """
        all_user_emb, all_item_emb = self.forward(norm_adj)
        u_e = all_user_emb[users]    # (batch, emb_size)
        pi_e = all_item_emb[pos_items]
        ni_e = all_item_emb[neg_items]

        pos_scores = torch.mul(u_e, pi_e).sum(dim=1)
        neg_scores = torch.mul(u_e, ni_e).sum(dim=1)
        bpr_loss = F.softplus(neg_scores - pos_scores).sum()

        # L2 Regularization on embeddings
        reg_loss = (u_e.norm(p=2).pow(2) +
                    pi_e.norm(p=2).pow(2) +
                    ni_e.norm(p=2).pow(2)) * self.reg_weight
        return bpr_loss + reg_loss

    def get_user_item_embeddings(self, norm_adj):
        return self.forward(norm_adj)

    def predict_score(self, norm_adj, user_idx, item_idx):
        # For predict usage
        all_user_emb, all_item_emb = self.forward(norm_adj)
        user_vec = all_user_emb[user_idx]
        item_vec = all_item_emb[item_idx]
        return torch.mul(user_vec, item_vec).sum(dim=1)


class MLPStudent(nn.Module):
    r"""
    A simple MLP-based student.  
    We'll store user/item embeddings and pass them through 1 or 2 FC layers (residual style).
    """
    def __init__(self, n_users, n_items, emb_size, n_layers=1, reg_weight=1e-5):
        super(MLPStudent, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.reg_weight = reg_weight

        self.user_embedding = nn.Embedding(self.n_users, self.emb_size)
        self.item_embedding = nn.Embedding(self.n_items, self.emb_size)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # A small MLP stack. For example:
        #  user_vec -> FC -> LeakyReLU -> residual -> FC ...
        layers = []
        hidden_size = emb_size
        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
            # We can add a small residual skip or do a direct pass
        self.seq_layers = nn.Sequential(*layers)

    def forward(self):
        return self.user_embedding.weight, self.item_embedding.weight

    def encode_user(self, u):
        # run user emb through MLP
        ue = self.user_embedding(u)
        # pass through seq
        ue2 = ue
        ue2 = self.seq_layers(ue2)
        return ue2

    def encode_item(self, i):
        ie = self.item_embedding(i)
        ie2 = self.seq_layers(ie)
        return ie2

    def pairwise_bpr_loss(self, users, pos_items, neg_items):
        # optional local BPR for the student, if we want
        u_e = self.encode_user(users)
        pi_e = self.encode_item(pos_items)
        ni_e = self.encode_item(neg_items)
        pos_scores = torch.mul(u_e, pi_e).sum(dim=1)
        neg_scores = torch.mul(u_e, ni_e).sum(dim=1)
        bpr_loss = F.softplus(neg_scores - pos_scores).sum()

        reg_loss = (u_e.norm(p=2).pow(2) +
                    pi_e.norm(p=2).pow(2) +
                    ni_e.norm(p=2).pow(2)) * self.reg_weight
        return bpr_loss + reg_loss

    def predict_score(self, user_idx, item_idx):
        u_e = self.encode_user(user_idx)
        i_e = self.encode_item(item_idx)
        return torch.mul(u_e, i_e).sum(dim=1)


class SimRec(GeneralRecommender):
    r"""
    SimRec: Graph-less Collaborative Filtering
      - Has a LightGCN-based teacher (optionally frozen)
      - Has an MLP-based student
      - Incorporates dual-level knowledge distillation
      - Uses an adaptive contrastive regularization

    This is a RecBole-style model with PAIRWISE input.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SimRec, self).__init__(config, dataset)

        # 1) Basic settings
        self.embedding_size = config["embedding_size"]
        self.n_users = dataset.num(dataset.uid_field)  # or dataset.num_users (if your version supports)
        self.n_items = dataset.num(dataset.iid_field)
        self.reg_weight_teacher = config["reg_weight_teacher"] if "reg_weight_teacher" in config else 1e-5
        self.reg_weight_student = config["reg_weight_student"] if "reg_weight_student" in config else 1e-5

        # Number of LightGCN layers
        self.teacher_n_layers = config["teacher_n_layers"] if "teacher_n_layers" in config else 2
        # Number of MLP layers
        self.student_n_layers = config["student_n_layers"] if "student_n_layers" in config else 1

        # 2) Build submodules
        # Teacher
        self.teacher = LightGCNTeacher(
            n_users=self.n_users,
            n_items=self.n_items,
            emb_size=self.embedding_size,
            n_layers=self.teacher_n_layers,
            reg_weight=self.reg_weight_teacher
        )
        # Student
        self.student = MLPStudent(
            n_users=self.n_users,
            n_items=self.n_items,
            emb_size=self.embedding_size,
            n_layers=self.student_n_layers,
            reg_weight=self.reg_weight_student
        )

        self.lambda_pred_distill = config["lambda_pred_distill"] if "lambda_pred_distill" in config else 1.0
        self.lambda_emb_distill  = config["lambda_emb_distill"]  if "lambda_emb_distill"  in config else 1.0
        self.lambda_contrast_reg = config["lambda_contrast_reg"] if "lambda_contrast_reg" in config else 0.1

        self.temp_pred_distill   = config["temp_pred_distill"]   if "temp_pred_distill"   in config else 1.0
        self.temp_emb_distill    = config["temp_emb_distill"]    if "temp_emb_distill"    in config else 1.0
        self.temp_contrast_reg   = config["temp_contrast_reg"]   if "temp_contrast_reg"   in config else 1.0

        self.freeze_teacher      = config["freeze_teacher"]      if "freeze_teacher"      in config else True

        # We also need adjacency for the teacher LightGCN
        # Because RecBole doesn't store a single adjacency matrix by default, we must build it ourselves.
        self.norm_adj = self._build_sparse_graph(dataset)  # see helper below

        # Define a BPRLoss instance if we want to do pairwise
        self.bpr_loss = BPRLoss()

    def _build_sparse_graph(self, dataset):
        r"""
        Creates a lightGCN-style normalized adjacency from the dataset’s user–item edges.
        For large data, you’d do a more efficient approach. This is a simple example.
        """
        import torch_sparse  # or your own builder
        inter_feat = dataset.inter_feat
        users = inter_feat[dataset.uid_field].numpy()
        items = inter_feat[dataset.iid_field].numpy()
        n_users = self.n_users
        n_items = self.n_items

        # adjacency dimension: (n_users + n_items) x (n_users + n_items)
        row = []
        col = []
        # user -> item edges
        for u, i in zip(users, items):
            row.append(u)
            col.append(n_users + i)
            # symmetrical
            row.append(n_users + i)
            col.append(u)

        row = torch.tensor(row, dtype=torch.long)
        col = torch.tensor(col, dtype=torch.long)
        edge = torch.ones(len(row), dtype=torch.float)
        # build coo
        ui_graph = torch.sparse_coo_tensor(
            indices=torch.stack([row, col], dim=0),
            values=edge,
            size=(n_users + n_items, n_users + n_items)
        )
        ui_graph = ui_graph.coalesce()

        # deg
        deg = torch.sparse.sum(ui_graph, dim=1).to_dense()  # (n_users+n_items,)
        deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

        # normalize
        # A_norm = D^-1/2 * A * D^-1/2
        idx = ui_graph.indices()
        vals = ui_graph.values()
        row_idx, col_idx = idx[0, :], idx[1, :]
        val_new = deg_inv_sqrt[row_idx] * vals * deg_inv_sqrt[col_idx]
        norm_adj = torch.sparse_coo_tensor(
            indices=idx,
            values=val_new,
            size=ui_graph.size()
        ).coalesce()
        return norm_adj.to(self.device)

    def forward(self, user, pos_item, neg_item):
        r"""
        For pairwise training, we'll return (pos_score, neg_score) predicted by the *student*.
        The teacher can be used for distillation inside `calculate_loss`.
        """
        # We'll compute the student’s forward for pairwise:
        user_e = self.student.encode_user(user)
        pos_e = self.student.encode_item(pos_item)
        neg_e = self.student.encode_item(neg_item)
        pos_score = (user_e * pos_e).sum(dim=-1)
        neg_score = (user_e * neg_e).sum(dim=-1)
        return pos_score, neg_score

    def calculate_loss(self, interaction):
        r"""
        This is called each training step in pairwise mode. We'll:
          1) Possibly do BPR training on the teacher (unless frozen).
          2) Do knowledge distillation for the student:
             - Prediction-level distillation (L1)
             - Embedding-level distillation (L2)
             - Contrastive regularization (L3)
          3) Possibly do standard BPR for the student as well (optional).
        """
        # Unpack
        user = interaction[self.USER_ID]     # shape: (batch,)
        pos_item = interaction[self.ITEM_ID] # shape: (batch,)
        neg_item = interaction[self.NEG_ITEM_ID]

        # 1) Teacher BPR (optional)
        if not self.freeze_teacher:
            teacher_bpr = self.teacher.calculate_bpr_loss(
                self.norm_adj, user, pos_item, neg_item
            )
        else:
            teacher_bpr = torch.zeros(1, device=self.device, requires_grad=False)

        # 2) Student Distillation
        #    2.1 get teacher embeddings (freeze them if needed)
        if self.freeze_teacher:
            with torch.no_grad():
                teacher_user_all, teacher_item_all = self.teacher.get_user_item_embeddings(self.norm_adj)
        else:
            teacher_user_all, teacher_item_all = self.teacher.get_user_item_embeddings(self.norm_adj)
        #    2.2 get student embeddings
        student_user_all, student_item_all = self.student.forward()  # (n_users, d), (n_items, d)

        # (A) Prediction-level Distillation (L1)
        # We'll sample a random pair (pos, neg) for each user to get "dark knowledge"
        # or re-use this batch? The paper samples random items for knowledge distillation.
        # For simplicity, let's re-use the same pos, neg from interaction.
        # The "teacher" preference difference:
        with torch.no_grad():
            t_u = teacher_user_all[user]
            t_pi = teacher_item_all[pos_item]
            t_ni = teacher_item_all[neg_item]
            t_pos_scores = (t_u * t_pi).sum(dim=1)
            t_neg_scores = (t_u * t_ni).sum(dim=1)
            t_diff = (t_pos_scores - t_neg_scores) / self.temp_pred_distill
            t_prob = torch.sigmoid(t_diff)  # shape: (batch,)

        s_u = student_user_all[user]
        s_pi = student_item_all[pos_item]
        s_ni = student_item_all[neg_item]
        s_pos_scores = (s_u * s_pi).sum(dim=1) / self.temp_pred_distill
        s_neg_scores = (s_u * s_ni).sum(dim=1) / self.temp_pred_distill
        s_diff = s_pos_scores - s_neg_scores
        s_prob = torch.sigmoid(s_diff)

        # Binary cross-entropy to align s_prob with t_prob
        # L1 = - [ t_prob * log(s_prob) + (1-t_prob)*log(1 - s_prob ) ]
        pred_distill_loss = F.binary_cross_entropy(s_prob, t_prob)

        # (B) Embedding-level Distillation (L2)
        # The paper does user-wise InfoNCE or just direct MSE alignment. 
        # We'll do a simple MSE for demonstration on the user and item embeddings:
        # but only for *certain sampled* users, items or the entire set (small data).
        # E.g. we do "student_user = s_u, teacher_user = t_u" for each user in batch.
        # For a large dataset, you'd sample.
        # We'll do user in batch, item in pos_item as an example:
        with torch.no_grad():
            t_u_batch = teacher_user_all[user]
            t_i_batch = teacher_item_all[pos_item]  # or union of pos_item, neg_item
        s_u_batch = student_user_all[user]
        s_i_batch = student_item_all[pos_item]

        # e.g. MSE as embedding-level distillation:
        emb_distill_loss = F.mse_loss(s_u_batch, t_u_batch) + F.mse_loss(s_i_batch, t_i_batch)

        # (C) Contrastive Regularization (L3)  (adaptive)
        # For brevity, we’ll do a simpler approach: push away user-user or user-item embeddings if we suspect over-smoothing.
        # The original code checks gradient conflicts, but that’s complicated. We'll show a simpler approach:
        # A naive approach: "push away negative pairs with a margin" or "push away random pairs of users."
        # We'll do a quick example: user-user separation for the batch:
        # If user has no overlapping items, we push them away slightly. (This is a toy example!)
        # In practice you'd do something akin to the code snippet in the question with gradient checks.
        batch_size = user.size(0)
        # pick random other user in the batch to push away
        rand_idx = torch.randperm(batch_size, device=self.device)
        user_emb = s_u_batch  # shape (batch, d)
        user_emb_shuffled = s_u_batch[rand_idx]
        # a contrastive "divergence" (like InfoNCE):
        sim_uu = (user_emb * user_emb_shuffled).sum(dim=1) / self.temp_contrast_reg
        contrast_reg_loss = F.logsigmoid(-sim_uu).mean()

        # Weighted sum of the 3 losses
        kd_loss = (self.lambda_pred_distill * pred_distill_loss +
                   self.lambda_emb_distill  * emb_distill_loss +
                   self.lambda_contrast_reg * (-contrast_reg_loss))  # minus because logsigmoid(-sim)

        # 3) (Optional) Student BPR
        # Some versions skip direct BPR on student, focusing purely on knowledge distillation.
        # If you want the student to see real data, do:
        student_bpr = self.student.pairwise_bpr_loss(user, pos_item, neg_item)

        # Combine everything
        loss = teacher_bpr + student_bpr + kd_loss
        return loss

    def predict(self, interaction):
        r"""
        For pointwise evaluation or ranking, we typically only use the student’s predicted score.
        Because the teacher is only used for distillation.
        """
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        scores = self.student.predict_score(user, item)
        return scores

    def full_sort_predict(self, interaction):
        r"""
        For a user or batch of users, predict on ALL items (ranking).
        We'll use student’s embeddings for inference.
        """
        user = interaction[self.USER_ID]  # shape (batch,)

        # student forward
        all_user_emb, all_item_emb = self.student.forward()
        # gather user embedding
        u_e = all_user_emb[user]  # shape (batch, emb_size)
        # matrix multiply with all items
        scores = torch.matmul(u_e, all_item_emb.transpose(0, 1))  # shape (batch, n_items)
        return scores

