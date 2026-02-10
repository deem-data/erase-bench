"""
DNNTSP: Dynamic Neural Networks for Temporal Set Prediction
Paper: "Sets2Sets" approach extended with graph neural networks and temporal modeling
Reference implementation: A-Next-Basket-Recommendation-Reality-Check
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
from torch_geometric.data import Data, Batch
from collections import defaultdict
import itertools
from sklearn.preprocessing import normalize

from recbole.model.abstract_recommender import SequentialRecommender


# ============================================================================
# Graph Convolution Layers
# ============================================================================

class WeightedGraphConv(MessagePassing):
    """Apply graph convolution over an input signal with edge weights."""
    
    def __init__(self, in_features: int, out_features: int):
        super(WeightedGraphConv, self).__init__(aggr="sum")
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, edge_index, node_features, edge_weights):
        """
        Args:
            edge_index: Tensor of shape [2, num_edges]
            node_features: Tensor of shape (N, in_features) or (N, T, in_features)
            edge_weights: Tensor of shape (num_edges,) or (T, num_edges)
        Returns:
            Transformed node features of shape (N, T, out_features)
        """
        # If edge_weights is 2D (T, num_edges), transpose to (num_edges, T) for proper indexing
        if edge_weights.dim() == 2:
            edge_weights = edge_weights.t()  # Now (num_edges, T)

        # Store original shape if 3D
        is_3d = node_features.dim() == 3
        if is_3d:
            N, T, F = node_features.shape
            # Reshape to 2D for PyG processing: merge T into features
            # This allows PyG to properly index nodes
            node_features_2d = node_features.reshape(N, T * F)
            result = self.propagate(edge_index, x=node_features_2d, edge_weight=edge_weights, temporal_dim=T)
            # Note: we pass temporal_dim as a kwarg so message/aggregate can access it
        else:
            result = self.propagate(edge_index, x=node_features, edge_weight=edge_weights, temporal_dim=None)

        return result

    def message(self, x_j, edge_weight, temporal_dim):
        """
        Equivalent to fn.u_mul_e('n', 'e', 'msg') in DGL.
        Args:
            x_j: Feature of source node (neighbor) - shape (num_edges, features) or (num_edges, T*F) if temporal
            edge_weight: Feature of edge - shape (num_edges,) or (num_edges, T) after transpose in forward
            temporal_dim: T dimension if node features were 3D, None otherwise
        Returns:
            Message tensor of shape (num_edges, T, features) when temporal, else (num_edges, features)
        """
        if edge_weight.dim() == 2:  # If edge weights have a time dimension (num_edges, T)
            # Unsqueeze to (num_edges, T, 1)
            edge_weight_t = edge_weight.unsqueeze(-1)  # (num_edges, T, 1)

            if temporal_dim is not None:
                # x_j has shape (num_edges, T*F), need to reshape to (num_edges, T, F)
                num_edges = x_j.size(0)
                F = x_j.size(1) // temporal_dim
                x_j = x_j.reshape(num_edges, temporal_dim, F)  # (num_edges, T, F)
            else:
                # First layer: x_j is (num_edges, features), expand to (num_edges, 1, features)
                x_j = x_j.unsqueeze(1)  # (num_edges, 1, features)

            # Broadcast multiply: (num_edges, T, 1) * (num_edges, 1 or T, features) -> (num_edges, T, features)
            result = edge_weight_t * x_j
            return result
        else:
            return edge_weight.view(-1, 1) * x_j  # (num_edges, features)

    def aggregate(self, inputs, index):
        """
        Equivalent to fn.sum('msg', 'h') in DGL.
        Args:
            inputs: Aggregated messages from neighbors - shape (num_edges, features) or (num_edges, T, features)
            index: Target node indices - shape (num_edges,)
        Returns:
            Aggregated node features - shape (num_nodes, features) or (num_nodes, T, features)
        """
        # Always scatter along dimension 0 (the edge/neighbor dimension)
        # This aggregates messages from all neighbors for each node
        return scatter(inputs, index, dim=0, reduce='sum')  # Sum over neighbors

    def update(self, aggr_out):
        """Apply linear transformation."""
        return self.linear(aggr_out)


class WeightedGCN(nn.Module):
    """Weighted Graph Convolutional Network with multiple layers."""
    
    def __init__(self, in_features: int, hidden_sizes: List[int], out_features: int):
        super(WeightedGCN, self).__init__()
        gcns, relus, bns = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        
        # Layers for hidden_size
        input_size = in_features
        for hidden_size in hidden_sizes:
            gcns.append(WeightedGraphConv(input_size, hidden_size))
            relus.append(nn.ReLU())
            bns.append(nn.BatchNorm1d(hidden_size))
            input_size = hidden_size
        
        # Output layer
        gcns.append(WeightedGraphConv(hidden_sizes[-1], out_features))
        relus.append(nn.ReLU())
        bns.append(nn.BatchNorm1d(out_features))
        
        self.gcns, self.relus, self.bns = gcns, relus, bns

    def forward(self, edge_index, node_features: torch.Tensor, edges_weight: torch.Tensor):
        """
        Args:
            edge_index: [2, num_edges]
            node_features: shape (n_1+n_2+..., n_features)
            edges_weight: shape (T, n_1^2+n_2^2+...)
        Returns:
            Node features of shape (n_1+n_2+..., T, features)
        """
        h = node_features
        for gcn, relu, bn in zip(self.gcns, self.relus, self.bns):
            # (n_1+n_2+..., T, features)
            h = gcn(edge_index, h, edges_weight)
            
            h = h.transpose(1, -1)

            # Need this when only 1 item is in the combined baskets
            if h.size(-1) > 1:
                h = bn(h)  # Normal BN update
            else:
                h = torch.nn.functional.batch_norm(
                    h, bn.running_mean, bn.running_var,
                    bn.weight, bn.bias,
                    training=False,
                    eps=bn.eps,
                )
            h = h.transpose(1, -1)
            h = relu(h)
        return h


class StackedWeightedGCNBlocks(nn.ModuleList):
    """Stack multiple WeightedGCN blocks."""
    
    def __init__(self, *args, **kwargs):
        super(StackedWeightedGCNBlocks, self).__init__(*args, **kwargs)

    def forward(self, edge_index, nodes_feature, edge_weights):
        h = nodes_feature
        for module in self:
            h = module(edge_index, h, edge_weights)
        return h


# ============================================================================
# Masked Self-Attention
# ============================================================================

class MaskedSelfAttention(nn.Module):
    """Multi-head self-attention with causal masking for temporal modeling."""
    
    def __init__(self, input_dim, output_dim, n_heads=4, attention_aggregate="concat"):
        super(MaskedSelfAttention, self).__init__()
        # Aggregate multi-heads by concatenation or mean
        self.attention_aggregate = attention_aggregate

        # The dimension of each head is dq // n_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_heads = n_heads

        if attention_aggregate == "concat":
            self.per_head_dim = self.dq = self.dk = self.dv = output_dim // n_heads
        elif attention_aggregate == "mean":
            self.per_head_dim = self.dq = self.dk = self.dv = output_dim
        else:
            raise ValueError(f"wrong value for aggregate {attention_aggregate}")

        self.Wq = nn.Linear(input_dim, n_heads * self.dq, bias=False)
        self.Wk = nn.Linear(input_dim, n_heads * self.dk, bias=False)
        self.Wv = nn.Linear(input_dim, n_heads * self.dv, bias=False)

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: tensor, shape (nodes_num, T_max, features_num)
        Returns:
            output: tensor, shape (nodes_num, T_max, output_dim = features_num)
        """
        seq_length = input_tensor.shape[1]
        # Tensor, shape (nodes_num, T_max, n_heads * dim_per_head)
        Q = self.Wq(input_tensor)
        K = self.Wk(input_tensor)
        V = self.Wv(input_tensor)
        
        # Multi-head attention
        # Q, tensor, shape (nodes_num, n_heads, T_max, dim_per_head)
        Q = Q.reshape(input_tensor.shape[0], input_tensor.shape[1], self.n_heads, self.dq).transpose(1, 2)
        # K after transpose, tensor, shape (nodes_num, n_heads, dim_per_head, T_max)
        K = K.reshape(input_tensor.shape[0], input_tensor.shape[1], self.n_heads, self.dk).permute(0, 2, 3, 1)
        # V, tensor, shape (nodes_num, n_heads, T_max, dim_per_head)
        V = V.reshape(input_tensor.shape[0], input_tensor.shape[1], self.n_heads, self.dv).transpose(1, 2)

        # Scaled attention_score, tensor, shape (nodes_num, n_heads, T_max, T_max)
        attention_score = Q.matmul(K) / np.sqrt(self.per_head_dim)

        # Attention_mask, tensor, shape -> (T_max, T_max)  -inf in the top and right
        attention_mask = torch.zeros(seq_length, seq_length).masked_fill(
            torch.tril(torch.ones(seq_length, seq_length)) == 0, -np.inf).to(input_tensor.device)
        # Attention_mask will be broadcast to (nodes_num, n_heads, T_max, T_max)
        attention_score = attention_score + attention_mask
        # (nodes_num, n_heads, T_max, T_max)
        attention_score = torch.softmax(attention_score, dim=-1)

        # Multi_result, tensor, shape (nodes_num, n_heads, T_max, dim_per_head)
        multi_head_result = attention_score.matmul(V)
        
        if self.attention_aggregate == "concat":
            # Multi_result, tensor, shape (nodes_num, T_max, n_heads * dim_per_head = output_dim)
            # Concat multi-head attention results
            output = multi_head_result.transpose(1, 2).reshape(
                input_tensor.shape[0], seq_length, self.n_heads * self.per_head_dim
            )
        elif self.attention_aggregate == "mean":
            # Multi_result, tensor, shape (nodes_num, T_max, dim_per_head = output_dim)
            # Mean multi-head attention results
            output = multi_head_result.transpose(1, 2).mean(dim=2)
        else:
            raise ValueError(f"wrong value for aggregate {self.attention_aggregate}")

        return output


# ============================================================================
# Aggregate Nodes Temporal Feature
# ============================================================================

class AggregateNodesTemporalFeature(nn.Module):
    """Aggregate node features across temporal dimension using learned weights."""
    
    def __init__(self, item_embed_dim):
        """
        Args:
            item_embed_dim: the dimension of input features
        """
        super(AggregateNodesTemporalFeature, self).__init__()
        self.Wq = nn.Linear(item_embed_dim, 1, bias=False)

    def forward(self, graph, lengths, nodes_output):
        """
        Args:
            graph: batched graphs (PyG Batch object)
            lengths: tensor, (batch_size, )
            nodes_output: the output of self-attention model in time dimension, 
                         (n_1+n_2+..., T_max, F)
        Returns:
            aggregated_features: (n_1+n_2+..., F)
        """
        num_nodes_per_graph = graph.ptr[1:] - graph.ptr[:-1]  # Equivalent to batch_num_nodes()

        aggregated_features = []
        start_idx = 0

        for num_nodes, length in zip(num_nodes_per_graph, lengths):
            num_nodes = num_nodes.item()  # Convert to Python int
            length = length.item() if torch.is_tensor(length) else length  # Ensure length is int

            # Get node feature slice
            output_node_features = nodes_output[start_idx:start_idx + num_nodes, :length, :]

            # Compute weights (user_nodes, 1, user_length)
            weights = self.Wq(output_node_features).transpose(1, 2)

            # Weighted aggregation (user_nodes, 1, item_embed_dim) -> (user_nodes, item_embed_dim)
            aggregated_feature = weights.matmul(output_node_features).squeeze(dim=1)

            aggregated_features.append(aggregated_feature)
            start_idx += num_nodes

        # Concatenate results from all graphs (num_total_users, item_embed_dim)
        aggregated_features = torch.cat(aggregated_features, dim=0)

        return aggregated_features


# ============================================================================
# Global Gated Update
# ============================================================================

class GlobalGatedUpdate(nn.Module):
    """Global gated update mechanism for item embeddings."""
    
    def __init__(self, items_total, item_embedding):
        super(GlobalGatedUpdate, self).__init__()
        self.items_total = items_total
        self.item_embedding = item_embedding

        # alpha -> the weight for updating
        self.alpha = nn.Parameter(torch.rand(items_total, 1), requires_grad=True)

    def forward(self, graph, nodes, nodes_output):
        """
        Args:
            graph: batched graphs (PyG Batch object)
            nodes: tensor (n_1+n_2+..., ) - item IDs for each node
            nodes_output: the output of self-attention model in time dimension, (n_1+n_2+..., F)
        Returns:
            batch_embedding: (batch_size, items_total, item_embed_dim)
        """
        num_nodes_per_graph = graph.ptr[1:] - graph.ptr[:-1]  # Equivalent to batch_num_nodes()
        batch_embedding = []
        start_idx = 0  # Start index for slicing nodes in a batch

        # Get the global item embeddings
        items_embedding = self.item_embedding(torch.arange(self.items_total, device=nodes.device))

        for num_nodes in num_nodes_per_graph:
            num_nodes = num_nodes.item()  # Convert to Python int
            # Slice node features for this graph
            output_node_features = nodes_output[start_idx:start_idx + num_nodes, :]  # Shape: (user_nodes, item_embed_dim)
            output_nodes = nodes[start_idx:start_idx + num_nodes]  # Item IDs for this subgraph

            # output_nodes should already be unique within each graph
            unique_nodes, inverse_indices = torch.unique(output_nodes, return_inverse=True)

            # Aggregate features for unique nodes (average if there are duplicates)
            unique_features = torch.zeros(len(unique_nodes), output_node_features.size(1),
                                         device=nodes.device, dtype=output_node_features.dtype)
            unique_features.index_add_(0, inverse_indices, output_node_features)
            counts = torch.bincount(inverse_indices, minlength=len(unique_nodes)).float().unsqueeze(1)
            unique_features = unique_features / counts

            # Initialize beta (items_total, 1) and set indicators
            beta = torch.zeros(self.items_total, 1, device=nodes.device)
            beta[unique_nodes] = 1

            # Compute updated embedding using gated mechanism
            embed = (1 - beta * self.alpha) * items_embedding.clone()

            # Apply gated update for appearing items (using unique nodes)
            embed[unique_nodes, :] = embed[unique_nodes, :] + self.alpha[unique_nodes] * unique_features

            # Append processed embedding for this batch
            batch_embedding.append(embed)

            start_idx += num_nodes  # Move to the next graph in the batch

        # Stack embeddings into shape (batch_size, items_total, item_embed_dim)
        batch_embedding = torch.stack(batch_embedding)
        
        return batch_embedding


# ============================================================================
# Main DNNTSP Model
# ============================================================================

class DNNTSP(SequentialRecommender):
    """DNNTSP: Dynamic Neural Networks for Temporal Set Prediction.
    
    This model uses graph neural networks with temporal modeling for next basket 
    recommendation. It constructs fully-connected graphs from user basket histories
    and applies temporal graph convolutions, self-attention, and gated updates.
    
    Requires task_type='NBR' and NextBasketDataset.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # Configuration
        self.embedding_size = config["embedding_size"]
        self.items_total = self.n_items
        
        # Dataset specific field names (from NextBasketDataset)
        self.history_items_field = (
            config["DNNTSP_HISTORY_ITEMS_FIELD"]
            if "DNNTSP_HISTORY_ITEMS_FIELD" in config
            else "history_item_matrix"
        )
        self.history_length_field = (
            config["DNNTSP_HISTORY_LENGTH_FIELD"]
            if "DNNTSP_HISTORY_LENGTH_FIELD" in config
            else "history_basket_length"
        )
        self.history_item_len_field = (
            config["DNNTSP_HISTORY_ITEM_LENGTH_FIELD"]
            if "DNNTSP_HISTORY_ITEM_LENGTH_FIELD" in config
            else "history_item_length_per_basket"
        )
        self.target_items_field = (
            config["DNNTSP_TARGET_ITEMS_FIELD"]
            if "DNNTSP_TARGET_ITEMS_FIELD" in config
            else "target_item_list"
        )
        self.target_length_field = (
            config["DNNTSP_TARGET_LENGTH_FIELD"]
            if "DNNTSP_TARGET_LENGTH_FIELD" in config
            else "target_item_length"
        )

        # Model components
        self.item_embedding = nn.Embedding(self.items_total, self.embedding_size)
        
        self.stacked_gcn = StackedWeightedGCNBlocks([
            WeightedGCN(self.embedding_size, [self.embedding_size], self.embedding_size)
        ])

        self.masked_self_attention = MaskedSelfAttention(
            input_dim=self.embedding_size,
            output_dim=self.embedding_size
        )

        self.aggregate_nodes_temporal_feature = AggregateNodesTemporalFeature(
            item_embed_dim=self.embedding_size
        )

        self.global_gated_update = GlobalGatedUpdate(
            items_total=self.items_total,
            item_embedding=self.item_embedding
        )

        self.fc_output = nn.Linear(self.embedding_size, 1)
        
        # Loss function
        self.loss_func = nn.MultiLabelSoftMarginLoss(reduction="mean")
        
        # Apply weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _get_nodes(self, baskets):
        """
        Get unique items from baskets.
        Args:
            baskets: list of tensors (baskets_num,) each containing item IDs
        Returns:
            tensor of unique item IDs
        """
        # Flatten and get unique items
        all_items = torch.cat([basket for basket in baskets if len(basket) > 0])
        unique_items = torch.unique(all_items)
        return unique_items

    def _get_edges_weight(self, baskets):
        """
        Count the appearing counts of item pairs in baskets.
        Args:
            baskets: list of tensors containing item IDs
        Returns:
            dict, key -> (item_1, item_2), value -> weight
        """
        edges_weight_dict = defaultdict(float)
        for basket in baskets:
            basket_list = basket.tolist()
            for i in range(len(basket_list)):
                for j in range(i + 1, len(basket_list)):
                    edges_weight_dict[(basket_list[i], basket_list[j])] += 1.0
                    edges_weight_dict[(basket_list[j], basket_list[i])] += 1.0
        return edges_weight_dict

    def _build_graph_for_user(self, history_items, history_item_len):
        """
        Build fully-connected graph for a single user's basket history.
        
        Args:
            history_items: [num_baskets, max_items_per_basket] - item IDs in each basket
            history_item_len: [num_baskets] - actual number of items per basket
            
        Returns:
            pyg_data: PyG Data object with edge_index
            nodes_feature: [num_nodes, embedding_size]
            edges_weight: [num_baskets, num_edges]
            nodes: [num_nodes] - item IDs
            user_data: list of tensors, each containing items in a basket
        """
        device = history_items.device
        
        # Extract baskets (list of tensors)
        user_data = []
        for basket_idx in range(history_items.size(0)):
            basket_len = history_item_len[basket_idx].item()
            if basket_len > 0:
                basket_items = history_items[basket_idx, :basket_len]
                # Filter out padding (0) and get unique items
                basket_items = basket_items[basket_items > 0]
                if len(basket_items) > 0:
                    basket_items = torch.unique(basket_items)
                    user_data.append(basket_items)
        
        if len(user_data) == 0:
            # Handle edge case: no valid baskets
            nodes = torch.tensor([1], device=device, dtype=torch.long)  # Use dummy item
            user_data = [torch.tensor([1], device=device, dtype=torch.long)]
        else:
            # Get unique nodes (items) across all baskets
            nodes = self._get_nodes(user_data)
        
        # Get node features (item embeddings)
        nodes_feature = self.item_embedding(nodes)
        
        # Construct fully connected graph
        num_nodes = nodes.size(0)
        project_nodes = torch.arange(num_nodes, device=device)

        # Create edge index: fully connected
        src = project_nodes.repeat_interleave(num_nodes)
        dst = project_nodes.repeat(num_nodes)
        edge_index = torch.stack([src, dst], dim=0).long()

        # Compute edge weights based on co-occurrence
        edges_weight_dict = self._get_edges_weight(user_data)

        # Add self-loops
        for node in nodes.tolist():
            if edges_weight_dict[(node, node)] == 0.0:
                edges_weight_dict[(node, node)] = 1.0

        # Normalize weights
        max_weight = max(edges_weight_dict.values()) if edges_weight_dict else 1.0
        for key in edges_weight_dict:
            edges_weight_dict[key] = edges_weight_dict[key] / max_weight

        # Create PyG Data object
        pyg_data = Data(x=nodes_feature, edge_index=edge_index, num_nodes=num_nodes)
        
        # Get edge weights for each timestamp
        edges_weight = []
        for basket in user_data:
            basket_list = basket.tolist()
            edge_weight = []
            for node_1 in nodes.tolist():
                for node_2 in nodes.tolist():
                    if (node_1 in basket_list and node_2 in basket_list) or (node_1 == node_2):
                        edge_weight.append(edges_weight_dict[(node_1, node_2)])
                    else:
                        edge_weight.append(0.0)
            edges_weight.append(torch.tensor(edge_weight, device=device))
        
        # Stack to [T, num_edges]
        if len(edges_weight) > 0:
            edges_weight = torch.stack(edges_weight)
        else:
            # Handle edge case
            edges_weight = torch.ones(1, num_nodes * num_nodes, device=device)
        
        return pyg_data, nodes_feature, edges_weight, nodes, user_data

    def _transform_interaction_to_graphs(self, interaction):
        """
        Transform NextBasketDataset interaction to graph format.
        
        Args:
            interaction: Interaction object from NextBasketDataset
            
        Returns:
            batched_graph: PyG Batch object
            all_nodes_feature: [total_nodes, embedding_size]
            all_edges_weight: [T_max, total_edges]
            lengths: [batch_size]
            all_nodes: [total_nodes]
            users_frequency: [batch_size, n_items]
        """
        history_items = interaction[self.history_items_field]  # [batch, max_baskets, max_items]
        history_len = interaction[self.history_length_field]  # [batch]
        per_basket_len = interaction[self.history_item_len_field]  # [batch, max_baskets]
        
        batch_size = history_items.size(0)
        device = history_items.device
        
        graphs = []
        nodes_features = []
        edges_weights = []
        all_nodes = []
        lengths = []
        all_user_data = []
        
        for batch_idx in range(batch_size):
            # Get data for this user
            user_history = history_items[batch_idx]  # [max_baskets, max_items]
            user_basket_len = per_basket_len[batch_idx]  # [max_baskets]
            user_num_baskets = history_len[batch_idx].item()
            
            # Only use actual baskets (up to user_num_baskets)
            user_history = user_history[:user_num_baskets]
            user_basket_len = user_basket_len[:user_num_baskets]
            
            # Build graph for this user
            pyg_data, nodes_feature, edges_weight, nodes, user_data = self._build_graph_for_user(
                user_history, user_basket_len
            )


            graphs.append(pyg_data)
            nodes_features.append(nodes_feature)
            edges_weights.append(edges_weight)
            all_nodes.append(nodes)
            lengths.append(edges_weight.size(0))  # Number of baskets
            all_user_data.append(user_data)
        
        # Batch graphs
        batched_graph = Batch.from_data_list(graphs)

        # Concatenate node features and nodes
        all_nodes_feature = torch.cat(nodes_features, dim=0)
        all_nodes_tensor = torch.cat(all_nodes, dim=0)
        
        # Pad and concatenate edge weights
        max_length = max(lengths)
        padded_edges_weights = []
        for edges_weight in edges_weights:
            num_edges = edges_weight.size(1)
            if edges_weight.size(0) < max_length:
                # Pad with identity matrix (self-loops)
                num_nodes = int(np.sqrt(num_edges))
                padding = torch.eye(num_nodes, device=device).flatten().unsqueeze(0).repeat(
                    max_length - edges_weight.size(0), 1
                )
                edges_weight = torch.cat([edges_weight, padding], dim=0)
            padded_edges_weights.append(edges_weight)
        
        # Concatenate along edge dimension: [T_max, total_edges]
        all_edges_weight = torch.cat(padded_edges_weights, dim=1)
        lengths_tensor = torch.tensor(lengths, device=device)
        
        # Compute user frequency
        users_frequency = np.zeros([batch_size, self.items_total])
        for idx, user_data in enumerate(all_user_data):
            for basket in user_data:
                for item in basket:
                    item_idx = item.item()
                    if item_idx > 0:  # Exclude padding
                        users_frequency[idx, item_idx] = users_frequency[idx, item_idx] + 1
        users_frequency = normalize(users_frequency, axis=1, norm='max')
        users_frequency = torch.tensor(users_frequency, dtype=torch.float32, device=device)
        
        return (batched_graph, all_nodes_feature, all_edges_weight, 
                lengths_tensor, all_nodes_tensor, users_frequency)

    def forward(self, interaction):
        """
        Forward pass of DNNTSP.
        
        Args:
            interaction: Interaction object from NextBasketDataset
            
        Returns:
            output: [batch_size, n_items] - scores for all items
        """
        # Transform interaction to graph format
        (graph, nodes_feature, edges_weight, lengths, nodes, users_frequency
         ) = self._transform_interaction_to_graphs(interaction)
        
        # Perform weighted GCN on dynamic graphs (n_1+n_2+..., T_max, item_embed_dim)
        nodes_output = self.stacked_gcn(graph.edge_index, nodes_feature, edges_weight)

        # Self-attention in time dimension, (n_1+n_2+..., T_max, item_embed_dim)
        nodes_output = self.masked_self_attention(nodes_output)

        # Aggregate node features in temporal dimension, (n_1+n_2+..., item_embed_dim)
        nodes_output = self.aggregate_nodes_temporal_feature(graph, lengths, nodes_output)

        # (batch_size, items_total, item_embed_dim)
        nodes_output = self.global_gated_update(graph, nodes, nodes_output)

        # (batch_size, items_total)
        output = self.fc_output(nodes_output).squeeze(dim=-1)

        return output

    def calculate_loss(self, interaction):
        """
        Calculate training loss.
        
        Args:
            interaction: Interaction object from NextBasketDataset
            
        Returns:
            loss: scalar tensor
        """
        # Get predictions
        output = self.forward(interaction)  # [batch_size, n_items]
        
        # Prepare target (multi-label)
        target_items = interaction[self.target_items_field]  # [batch_size, max_target_items]
        target_len = interaction[self.target_length_field].long()  # [batch_size]
        
        batch_size = target_items.size(0)
        device = target_items.device
        
        # Create multi-label target
        target = torch.zeros(batch_size, self.items_total, device=device)
        for batch_idx in range(batch_size):
            items = target_items[batch_idx, :target_len[batch_idx]]
            # Filter valid items (> 0)
            items = items[items > 0]
            if len(items) > 0:
                # Clamp to valid range
                items = torch.clamp(items, 0, self.items_total - 1)
                target[batch_idx].scatter_(0, items.long(), 1.0)
        
        # Exclude padding token (0) from target
        target[:, 0] = 0.0
        
        # Calculate loss
        loss = self.loss_func(output, target)
        
        return loss

    def predict(self, interaction):
        """
        Predict scores for specific items.
        
        Args:
            interaction: Interaction object
            
        Returns:
            scores: [batch_size] - scores for specified items
        """
        output = self.forward(interaction)  # [batch_size, n_items]
        item_ids = interaction[self.ITEM_ID]  # [batch_size]
        scores = output.gather(1, item_ids.view(-1, 1)).squeeze(1)
        return scores

    def full_sort_predict(self, interaction):
        """
        Predict scores for all items.
        
        Args:
            interaction: Interaction object
            
        Returns:
            scores: [batch_size, n_items] - scores for all items
        """
        output = self.forward(interaction)  # [batch_size, n_items]
        return output
    
    def get_basket_embedding(self, interaction):
        """
        Extract basket/user embedding representation for contrastive learning.
        
        This extracts the aggregated item embeddings before the final FC layer,
        representing the user's basket history.
        
        Args:
            interaction: Interaction object from NextBasketDataset
            
        Returns:
            basket_emb: [batch_size, item_embed_dim] - basket representation
        """
        # Transform interaction to graph format
        (graph, nodes_feature, edges_weight, lengths, nodes, users_frequency
         ) = self._transform_interaction_to_graphs(interaction)
        
        # Perform weighted GCN on dynamic graphs
        nodes_output = self.stacked_gcn(graph.edge_index, nodes_feature, edges_weight)
        
        # Self-attention in time dimension
        nodes_output = self.masked_self_attention(nodes_output)
        
        # Aggregate node features in temporal dimension
        nodes_output = self.aggregate_nodes_temporal_feature(graph, lengths, nodes_output)
        
        # Get item embeddings after global gated update: [batch_size, items_total, item_embed_dim]
        item_embeddings = self.global_gated_update(graph, nodes, nodes_output)
        
        # Aggregate item embeddings to get basket representation
        # Use mean pooling over items that appear in user's history (non-zero frequency)
        # users_frequency: [batch_size, items_total] - indicates which items appear
        item_mask = (users_frequency > 0).float()  # [batch_size, items_total]
        
        # Weighted mean: sum embeddings weighted by frequency, then normalize
        # item_embeddings: [batch_size, items_total, item_embed_dim]
        # users_frequency: [batch_size, items_total, 1] after unsqueeze
        weighted_emb = item_embeddings * users_frequency.unsqueeze(-1)  # [batch_size, items_total, item_embed_dim]
        basket_emb = weighted_emb.sum(dim=1)  # [batch_size, item_embed_dim]
        
        # Normalize by sum of frequencies (avoid division by zero)
        freq_sum = users_frequency.sum(dim=1, keepdim=True)  # [batch_size, 1]
        freq_sum = torch.clamp(freq_sum, min=1e-8)  # Avoid division by zero
        basket_emb = basket_emb / freq_sum
        
        return basket_emb

