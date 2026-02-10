# @Time   : 2021/6/23
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn

# UPDATE
# @Time   : 2021/7/18
# @Author : Zhichao Feng
# @email  : fzcbupt@gmail.com

"""
recbole.evaluator.collector
################################################
"""

from recbole.evaluator.register import Register
import torch
import copy


class DataStruct(object):
    def __init__(self):
        self._data_dict = {}

    def __getitem__(self, name: str):
        return self._data_dict[name]

    def __setitem__(self, name: str, value):
        self._data_dict[name] = value

    def __delitem__(self, name: str):
        self._data_dict.pop(name)

    def __contains__(self, key: str):
        return key in self._data_dict

    def get(self, name: str):
        if name not in self._data_dict:
            raise IndexError("Can not load the data without registration !")
        return self[name]

    def set(self, name: str, value):
        self._data_dict[name] = value

    def update_tensor(self, name: str, value: torch.Tensor):
        if name not in self._data_dict:
            self._data_dict[name] = value.clone().detach()
        else:
            if not isinstance(self._data_dict[name], torch.Tensor):
                raise ValueError("{} is not a tensor.".format(name))
            self._data_dict[name] = torch.cat(
                (self._data_dict[name], value.clone().detach()), dim=0
            )

    def __str__(self):
        data_info = "\nContaining:\n"
        for data_key in self._data_dict.keys():
            data_info += data_key + "\n"
        return data_info


class Collector(object):
    """The collector is used to collect the resource for evaluator.
    As the evaluation metrics are various, the needed resource not only contain the recommended result
    but also other resource from data and model. They all can be collected by the collector during the training
    and evaluation process.

    This class is only used in Trainer.

    """

    def __init__(self, config):
        self.config = config
        self.data_struct = DataStruct()
        self.register = Register(config)
        self.full = "full" in config["eval_args"]["mode"]
        self.topk = self.config["topk"]
        self.device = self.config["device"]

    def data_collect(self, train_data):
        """Collect the evaluation resource from training data.
        Args:
            train_data (AbstractDataLoader): the training dataloader which contains the training data.

        """
        if self.register.need("data.num_items"):
            item_id = self.config["ITEM_ID_FIELD"]
            self.data_struct.set("data.num_items", train_data.dataset.num(item_id))
        if self.register.need("data.num_users"):
            user_id = self.config["USER_ID_FIELD"]
            self.data_struct.set("data.num_users", train_data.dataset.num(user_id))
        if self.register.need("data.count_items"):
            self.data_struct.set("data.count_items", train_data.dataset.item_counter)
        if self.register.need("data.count_users"):
            self.data_struct.set("data.count_users", train_data.dataset.user_counter)

    def _average_rank(self, scores):
        """Get the ranking of an ordered tensor, and take the average of the ranking for positions with equal values.

        Args:
            scores(tensor): an ordered tensor, with size of `(N, )`

        Returns:
            torch.Tensor: average_rank

        Example:
            >>> average_rank(tensor([[1,2,2,2,3,3,6],[2,2,2,2,4,5,5]]))
            tensor([[1.0000, 3.0000, 3.0000, 3.0000, 5.5000, 5.5000, 7.0000],
            [2.5000, 2.5000, 2.5000, 2.5000, 5.0000, 6.5000, 6.5000]])

        Reference:
            https://github.com/scipy/scipy/blob/v0.17.1/scipy/stats/stats.py#L5262-L5352

        """
        length, width = scores.shape
        true_tensor = torch.full(
            (length, 1), True, dtype=torch.bool, device=self.device
        )

        obs = torch.cat([true_tensor, scores[:, 1:] != scores[:, :-1]], dim=1)
        # bias added to dense
        bias = (
            torch.arange(0, length, device=self.device)
            .repeat(width)
            .reshape(width, -1)
            .transpose(1, 0)
            .reshape(-1)
        )
        dense = obs.view(-1).cumsum(0) + bias

        # cumulative counts of each unique value
        count = torch.where(torch.cat([obs, true_tensor], dim=1))[1]
        # get average rank
        avg_rank = 0.5 * (count[dense] + count[dense - 1] + 1).view(length, -1)

        return avg_rank

    def eval_batch_collect(
        self,
        scores_tensor: torch.Tensor,
        interaction,
        positive_u: torch.Tensor,
        positive_i: torch.Tensor,
    ):
        """Collect the evaluation resource from batched eval data and batched model output.
        Args:
            scores_tensor (Torch.Tensor): the output tensor of model with the shape of `(N, )`
            interaction(Interaction): batched eval data.
            positive_u(Torch.Tensor): the row index of positive items for each user.
            positive_i(Torch.Tensor): the positive item id for each user.
        """
        # Check if this is NBR task and we need to use first k items from target basket
        is_nbr = (self.config["task_type"] if "task_type" in self.config else None) == "NBR"
        target_item_list_field = None
        if is_nbr:
            # Check if target_item_list field exists (from NextBasketDataset)
            # Default field name used by NextBasketDataset
            target_item_list_field = "target_item_list"
            # Verify the field exists in interaction
            try:
                _ = interaction[target_item_list_field]
            except (KeyError, AttributeError):
                target_item_list_field = None
        
        if self.register.need("rec.items"):

            # get topk
            _, topk_idx = torch.topk(
                scores_tensor, max(self.topk), dim=-1
            )  # n_users x k
            self.data_struct.update_tensor("rec.items", topk_idx)

        if self.register.need("rec.topk"):

            _, topk_idx = torch.topk(
                scores_tensor, max(self.topk), dim=-1
            )  # n_users x k
            
            # For NBR: use first k items from target basket for each k
            if is_nbr and target_item_list_field is not None:
                # Get target_item_list for each user in the batch
                try:
                    target_lists = interaction[target_item_list_field]  # [batch_size, max_basket_size]
                    # Debug: verify field is accessible and has correct shape
                    if not isinstance(target_lists, torch.Tensor):
                        # Field exists but is not a tensor - try to convert or fallback
                        target_item_list_field = None
                        is_nbr = False
                except (KeyError, AttributeError) as e:
                    # Fallback to standard behavior if target_item_list not available
                    if not hasattr(self, '_nbr_field_not_found_logged'):
                        import logging
                        logger = logging.getLogger()
                        logger.warning(f"[NBR Debug] target_item_list field not found in interaction. "
                                      f"Available fields: {list(interaction.interaction.keys()) if hasattr(interaction, 'interaction') else 'N/A'}. "
                                      f"Error: {e}")
                        self._nbr_field_not_found_logged = True
                    target_item_list_field = None
                    is_nbr = False
                
                if target_item_list_field is not None:
                    # Ensure target_lists is on the correct device
                    target_lists = target_lists.to(self.device)
                    batch_size = target_lists.shape[0]
                    max_topk = max(self.topk)
                    
                    # Verify target_lists has expected shape [batch_size, max_basket_size]
                    if len(target_lists.shape) != 2:
                        # Fallback if shape is unexpected
                        target_item_list_field = None
                        is_nbr = False
                    else:
                        # Debug: Print first batch to verify data format (remove after debugging)
                        if batch_size > 0:
                            sample_target = target_lists[0]
                            sample_actual = sample_target[sample_target >= 0]  # Filter padding (-1)
                            # Only print once per evaluation to avoid spam
                            if not hasattr(self, '_nbr_debug_printed'):
                                import logging
                                logger = logging.getLogger()
                                logger.info(f"[NBR Debug] target_item_list shape: {target_lists.shape}, "
                                           f"sample first user target items (first 10 non-padding): {sample_actual[:10].tolist()}, "
                                           f"sample first user raw values (first 20): {sample_target[:20].tolist()}, "
                                           f"num_items in scores: {scores_tensor.shape[1]}, "
                                           f"min/max in target_lists: {target_lists.min().item()}/{target_lists.max().item()}")
                                self._nbr_debug_printed = True
                        # Build pos_matrix marking ALL items in target basket as positive (original RecBole behavior)
                        # Compare top k predictions against ALL items in target basket (ignoring padding)
                        pos_matrix = torch.zeros_like(scores_tensor, dtype=torch.int)
                        num_items = scores_tensor.shape[1]
                        
                        # For each user, mark ALL items in their target basket as positive
                        # Note: target_item_list contains remapped item IDs (after _remap_ID_all)
                        # Format: [batch_size, max_basket_size] tensor with item IDs (-1 is padding)
                        actual_lengths = (target_lists >= 0).sum(dim=1)  # [batch_size] - count non-padding items (>= 0)
                        
                        for u_idx in range(batch_size):
                            target_items = target_lists[u_idx]  # [max_basket_size] with structure: [item1, item2, ..., -1, -1]
                            # Filter out padding (-1) to get actual items
                            actual_items = target_items[target_items >= 0]
                            
                            # Mark ALL items in target basket as positive (not just first k)
                            for item_id_tensor in actual_items:
                                item_id = int(item_id_tensor.item() if isinstance(item_id_tensor, torch.Tensor) else item_id_tensor)
                                # Ensure valid index (item IDs should be within [0, num_items) range)
                                if 0 <= item_id < num_items:
                                    pos_matrix[u_idx, item_id] = 1
                        
                        # For NBR, we need both:
                        # 1. actual_target_length (for Recall: compare against all items in target basket)
                        # 2. min(k, actual_target_length) (for NDCG: IDCG uses min(k, actual_length))
                        # We'll store: [actual_length, min(1, actual_length), min(2, actual_length), ..., min(max_topk, actual_length)]
                        pos_len_matrix = torch.zeros(batch_size, max_topk + 1, dtype=torch.int, device=self.device)
                        pos_len_matrix[:, 0] = actual_lengths  # Store actual_length in first column
                        for k_idx, k in enumerate(range(1, max_topk + 1)):
                            k_tensor = torch.tensor(k, dtype=torch.int, device=self.device)
                            pos_len_matrix[:, k_idx + 1] = torch.minimum(actual_lengths, k_tensor)
                        
                        pos_idx = torch.gather(pos_matrix, dim=1, index=topk_idx)
                        # Store pos_len_matrix flattened: [pos_idx (max_topk), actual_length (1), min(1,actual_length) (1), ..., min(max_topk,actual_length) (1)]
                        result = torch.cat((pos_idx, pos_len_matrix), dim=1)
                else:
                    # Fallback: if target_item_list not available, use standard behavior
                    is_nbr = False
            
            if not is_nbr:
                # Standard behavior: use all positive items
                pos_matrix = torch.zeros_like(scores_tensor, dtype=torch.int)
                pos_matrix[positive_u, positive_i] = 1
                pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
                pos_idx = torch.gather(pos_matrix, dim=1, index=topk_idx)
                result = torch.cat((pos_idx, pos_len_list), dim=1)
            
            self.data_struct.update_tensor("rec.topk", result)

        if self.register.need("rec.meanrank"):

            desc_scores, desc_index = torch.sort(scores_tensor, dim=-1, descending=True)

            # get the index of positive items in the ranking list
            pos_matrix = torch.zeros_like(scores_tensor)
            pos_matrix[positive_u, positive_i] = 1
            pos_index = torch.gather(pos_matrix, dim=1, index=desc_index)

            avg_rank = self._average_rank(desc_scores)
            pos_rank_sum = torch.where(
                pos_index == 1, avg_rank, torch.zeros_like(avg_rank)
            ).sum(dim=-1, keepdim=True)

            pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
            user_len_list = desc_scores.argmin(dim=1, keepdim=True)
            result = torch.cat((pos_rank_sum, user_len_list, pos_len_list), dim=1)
            self.data_struct.update_tensor("rec.meanrank", result)

        if self.register.need("rec.score"):

            self.data_struct.update_tensor("rec.score", scores_tensor)

        if self.register.need("data.label"):
            self.label_field = self.config["LABEL_FIELD"]
            self.data_struct.update_tensor(
                "data.label", interaction[self.label_field].to(self.device)
            )

    def model_collect(self, model: torch.nn.Module):
        """Collect the evaluation resource from model.
        Args:
            model (nn.Module): the trained recommendation model.
        """
        pass
        # TODO:

    def eval_collect(self, eval_pred: torch.Tensor, data_label: torch.Tensor):
        """Collect the evaluation resource from total output and label.
        It was designed for those models that can not predict with batch.
        Args:
            eval_pred (torch.Tensor): the output score tensor of model.
            data_label (torch.Tensor): the label tensor.
        """
        if self.register.need("rec.score"):
            self.data_struct.update_tensor("rec.score", eval_pred)

        if self.register.need("data.label"):
            self.label_field = self.config["LABEL_FIELD"]
            self.data_struct.update_tensor("data.label", data_label.to(self.device))

    def get_data_struct(self):
        """Get all the evaluation resource that been collected.
        And reset some of outdated resource.
        """
        for key in self.data_struct._data_dict:
            self.data_struct._data_dict[key] = self.data_struct._data_dict[key].cpu()
        returned_struct = copy.deepcopy(self.data_struct)
        for key in ["rec.topk", "rec.meanrank", "rec.score", "rec.items", "data.label"]:
            if key in self.data_struct:
                del self.data_struct[key]
        return returned_struct
