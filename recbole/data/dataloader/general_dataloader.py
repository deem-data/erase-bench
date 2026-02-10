# @Time   : 2020/7/7
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2022/7/8, 2020/9/9, 2020/9/29, 2021/7/15, 2022/7/6
# @Author : Zhen Tian, Yupeng Hou, Yushuo Chen, Xingyu Pan, Gaowei Zhang
# @email  : chenyuwuxinn@gmail.com, houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, xy_pan@foxmail.com, zgw15630559577@163.com

"""
recbole.data.dataloader.general_dataloader
################################################
"""

import numpy as np
import torch
from logging import getLogger
from recbole.data.dataloader.abstract_dataloader import (
    AbstractDataLoader,
    NegSampleDataLoader,
)
from recbole.data.interaction import Interaction, cat_interactions
from recbole.utils import InputType, ModelType


class TrainDataLoader(NegSampleDataLoader):
    """:class:`TrainDataLoader` is a dataloader for training.
    It can generate negative interaction when :attr:`training_neg_sample_num` is not zero.
    For the result of every batch, we permit that every positive interaction and its negative interaction
    must be in the same batch.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.logger = getLogger()
        self._set_neg_sample_args(
            config, dataset, config["MODEL_INPUT_TYPE"], config["train_neg_sample_args"]
        )
        self.sample_size = len(dataset)
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config["train_batch_size"]
        if self.neg_sample_args["distribution"] != "none":
            batch_num = max(batch_size // self.times, 1)
            new_batch_size = batch_num * self.times
            self.step = batch_num
            self.set_batch_size(new_batch_size)
        else:
            self.step = batch_size
            self.set_batch_size(batch_size)

    def update_config(self, config):
        self._set_neg_sample_args(
            config,
            self._dataset,
            config["MODEL_INPUT_TYPE"],
            config["train_neg_sample_args"],
        )
        super().update_config(config)

    def collate_fn(self, index):
        index = np.array(index)
        data = self._dataset[index]
        transformed_data = self.transform(self._dataset, data)
        return self._neg_sampling(transformed_data)



class UnlearnTrainDataLoader(NegSampleDataLoader):
    """:class:`UnlearnTrainDataLoader` yields, on each __next__(), **all** positive interactions of one user,
    then applies RecBole’s sequential transform (sliding‐window), without any negative sampling.

    There are two possible “dataset” inputs:

      1. A RecBole `Dataset` whose `.inter_feat` has already been filtered to contain only 
         the “unlearning” rows. In that case, we slice `dataset.inter_feat` directly.

      2. A RecBole `Interaction` (which already holds exactly the rows to unlearn) — in which case 
         we slice it by row‐index (via `interaction[indices]`) and call `to_dict()`.

    In both cases, after building a “raw” Interaction of shape [(num_rows_for_user,) for each field],
    we call `self.transform(...)` to produce the sliding‐window inputs and labels. Finally,
    we do `_neg_sampling(...)`, which is a no-op if `train_neg_sample_args: None`.

    Args:
        config (Config):    RecBole config (must have train_neg_sample_args: None).
        dataset (Dataset or Interaction):
            - If it has `.inter_feat`, we treat it as a full Dataset.
            - Otherwise we treat it as an Interaction.
        sampler (Sampler):  Ignored (we never use it in unlearning mode).
        shuffle (bool):     If True, randomize the user order each epoch.
    """

    def __init__(self, config, dataset, sampler=None, shuffle=False):
        # 1) Neg‐sampling setup (copied from RecBole). We assume train_neg_sample_args=None.
        self.logger = getLogger()
        self._set_neg_sample_args(
            config,
            dataset,
            config["MODEL_INPUT_TYPE"],
            config["train_neg_sample_args"],
        )
        # Although AbstractDataLoader expects `sample_size`, we will bypass its ``batch_size/step`` logic.
        self.sample_size = len(dataset)

        # 2) Call parent constructor → calls _init_batch_size_and_step() internally
        super().__init__(config, dataset, sampler=None, shuffle=shuffle)

        # 3) Remember which column is the user ID
        self.uid_field = dataset.uid_field


        all_uids = dataset.inter_feat[self.uid_field].numpy()

        user2indices = {}
        for row_idx, u in enumerate(all_uids):
            uid = int(u)
            user2indices.setdefault(uid, []).append(row_idx)

        self.user2indices = user2indices
        self.users = list(self.user2indices.keys())

        # 5) A pointer through self.users
        self._pointer = 0
        self.shuffle = shuffle

    def _init_batch_size_and_step(self):
        # AbstractDataLoader requires this stub, but we never use batch_size/step.
        self.step = 1
        self._batch_size = 1

    def __len__(self):
        return len(self.users)

    def __iter__(self):
        # At epoch start, optionally shuffle user order, then reset pointer
        if self.shuffle:
            np.random.shuffle(self.users)
        self._pointer = 0
        return self

    def __next__(self):
        if self._pointer >= len(self.users):
            raise StopIteration

        curr_uid = self.users[self._pointer]
        indices = self.user2indices[curr_uid]  # list of row‐indices for this user
        
        # ─── Step A: Build a “raw” Interaction of all rows for curr_uid ───

        raw_array = self._dataset.inter_feat
        raw_data = {
            field_name: torch.tensor(raw_array[field_name][indices])
            for field_name in self._dataset.field2type.keys()
        }
        raw_inter = Interaction(raw_data)

        # ─── Step B: Apply RecBole’s sequential transform (sliding‐window augmentation) ───
        # This generates new fields like `item_id_list` (shape [num_windows × max_seq_len]),
        # `label` (shape [num_windows]), `timestamp_list`, etc., according to your config["max_seq_len"].
        seq_inter = self.transform(self._dataset, raw_inter)

        # ─── Step C: Negative sampling (no-op if train_neg_sample_args=None) ───
        batch_inter = self._neg_sampling(seq_inter)

        self._pointer += 1
        return batch_inter


class NegSampleEvalDataLoader(NegSampleDataLoader):
    """:class:`NegSampleEvalDataLoader` is a dataloader for neg-sampling evaluation.
    It is similar to :class:`TrainDataLoader` which can generate negative items,
    and this dataloader also permits that all the interactions corresponding to each user are in the same batch
    and positive interactions are before negative interactions.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.logger = getLogger()
        phase = sampler.phase if sampler is not None else "test"
        self._set_neg_sample_args(
            config, dataset, InputType.POINTWISE, config[f"{phase}_neg_sample_args"]
        )
        if (
            self.neg_sample_args["distribution"] != "none"
            and self.neg_sample_args["sample_num"] != "none"
        ):
            user_num = dataset.user_num
            dataset.sort(by=dataset.uid_field, ascending=True)
            self.uid_list = []
            start, end = dict(), dict()
            for i, uid in enumerate(dataset.inter_feat[dataset.uid_field].numpy()):
                if uid not in start:
                    self.uid_list.append(uid)
                    start[uid] = i
                end[uid] = i
            self.uid2index = np.array([None] * user_num)
            self.uid2items_num = np.zeros(user_num, dtype=np.int64)
            for uid in self.uid_list:
                self.uid2index[uid] = slice(start[uid], end[uid] + 1)
                self.uid2items_num[uid] = end[uid] - start[uid] + 1
            self.uid_list = np.array(self.uid_list)
            self.sample_size = len(self.uid_list)
        else:
            self.sample_size = len(dataset)
        if shuffle:
            self.logger.warning("NegSampleEvalDataLoader can't shuffle")
            shuffle = False
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config["eval_batch_size"]
        if (
            self.neg_sample_args["distribution"] != "none"
            and self.neg_sample_args["sample_num"] != "none"
        ):
            inters_num = sorted(self.uid2items_num * self.times, reverse=True)
            batch_num = 1
            new_batch_size = inters_num[0]
            for i in range(1, len(inters_num)):
                if new_batch_size + inters_num[i] > batch_size:
                    break
                batch_num = i + 1
                new_batch_size += inters_num[i]
            self.step = batch_num
            self.set_batch_size(new_batch_size)
        else:
            self.step = batch_size
            self.set_batch_size(batch_size)

    def update_config(self, config):
        phase = self._sampler.phase if self._sampler.phase is not None else "test"
        self._set_neg_sample_args(
            config,
            self._dataset,
            InputType.POINTWISE,
            config[f"{phase}_neg_sample_args"],
        )
        super().update_config(config)

    def collate_fn(self, index):
        index = np.array(index)
        if (
            self.neg_sample_args["distribution"] != "none"
            and self.neg_sample_args["sample_num"] != "none"
        ):
            uid_list = self.uid_list[index]
            data_list = []
            idx_list = []
            positive_u = []
            positive_i = torch.tensor([], dtype=torch.int64)

            for idx, uid in enumerate(uid_list):
                index = self.uid2index[uid]
                transformed_data = self.transform(self._dataset, self._dataset[index])
                data_list.append(self._neg_sampling(transformed_data))
                idx_list += [idx for i in range(self.uid2items_num[uid] * self.times)]
                positive_u += [idx for i in range(self.uid2items_num[uid])]
                positive_i = torch.cat(
                    (positive_i, self._dataset[index][self.iid_field]), 0
                )

            cur_data = cat_interactions(data_list)
            idx_list = torch.from_numpy(np.array(idx_list)).long()
            positive_u = torch.from_numpy(np.array(positive_u)).long()

            return cur_data, idx_list, positive_u, positive_i
        else:
            data = self._dataset[index]
            transformed_data = self.transform(self._dataset, data)
            cur_data = self._neg_sampling(transformed_data)
            return cur_data, None, None, None


class FullSortEvalDataLoader(AbstractDataLoader):
    """:class:`FullSortEvalDataLoader` is a dataloader for full-sort evaluation. In order to speed up calculation,
    this dataloader would only return then user part of interactions, positive items and used items.
    It would not return negative items.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.logger = getLogger()
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field
        self.is_sequential = config["MODEL_TYPE"] == ModelType.SEQUENTIAL
        if not self.is_sequential:
            user_num = dataset.user_num
            self.uid_list = []
            self.uid2items_num = np.zeros(user_num, dtype=np.int64)
            self.uid2positive_item = np.array([None] * user_num)
            self.uid2history_item = np.array([None] * user_num)

            dataset.sort(by=self.uid_field, ascending=True)
            last_uid = None
            positive_item = set()
            uid2used_item = sampler.used_ids
            for uid, iid in zip(
                dataset.inter_feat[self.uid_field].numpy(),
                dataset.inter_feat[self.iid_field].numpy(),
            ):
                if uid != last_uid:
                    self._set_user_property(
                        last_uid, uid2used_item[last_uid], positive_item
                    )
                    last_uid = uid
                    self.uid_list.append(uid)
                    positive_item = set()
                positive_item.add(iid)
            self._set_user_property(last_uid, uid2used_item[last_uid], positive_item)
            self.uid_list = torch.tensor(self.uid_list, dtype=torch.int64)
            self.user_df = dataset.join(Interaction({self.uid_field: self.uid_list}))

        self.sample_size = len(self.user_df) if not self.is_sequential else len(dataset)
        if shuffle:
            self.logger.warning("FullSortEvalDataLoader can't shuffle")
            shuffle = False
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _set_user_property(self, uid, used_item, positive_item):
        if uid is None:
            return
        history_item = used_item - positive_item
        self.uid2positive_item[uid] = torch.tensor(
            list(positive_item), dtype=torch.int64
        )
        self.uid2items_num[uid] = len(positive_item)
        self.uid2history_item[uid] = torch.tensor(list(history_item), dtype=torch.int64)

    def _init_batch_size_and_step(self):
        batch_size = self.config["eval_batch_size"]
        if not self.is_sequential:
            batch_num = max(batch_size // self._dataset.item_num, 1)
            new_batch_size = batch_num * self._dataset.item_num
            self.step = batch_num
            self.set_batch_size(new_batch_size)
        else:
            self.step = batch_size
            self.set_batch_size(batch_size)

    def update_config(self, config):
        super().update_config(config)

    def collate_fn(self, index):
        index = np.array(index)
        if not self.is_sequential:
            user_df = self.user_df[index]
            uid_list = list(user_df[self.uid_field])

            history_item = self.uid2history_item[uid_list]
            positive_item = self.uid2positive_item[uid_list]

            history_u = torch.cat(
                [
                    torch.full_like(hist_iid, i)
                    for i, hist_iid in enumerate(history_item)
                ]
            )
            history_i = torch.cat(list(history_item))

            positive_u = torch.cat(
                [torch.full_like(pos_iid, i) for i, pos_iid in enumerate(positive_item)]
            )
            positive_i = torch.cat(list(positive_item))

            return user_df, (history_u, history_i), positive_u, positive_i
        else:
            interaction = self._dataset[index]
            transformed_interaction = self.transform(self._dataset, interaction)
            inter_num = len(transformed_interaction)
            positive_u = torch.arange(inter_num)
            positive_i = transformed_interaction[self.iid_field]

            return transformed_interaction, None, positive_u, positive_i
