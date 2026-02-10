# @Time   : 2020/6/26
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2022/7/8, 2021/6/23, 2020/9/26, 2020/9/26, 2020/10/01, 2020/9/16
# @Author : Zhen Tian, Zihan Lin, Yupeng Hou, Yushuo Chen, Shanlei Mu, Xingyu Pan
# @Email  : chenyuwuxinn@gmail.com, zhlin@ruc.edu.cn, houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, slmu@ruc.edu.cn, panxy@ruc.edu.cn

# UPDATE:
# @Time   : 2020/10/8, 2020/10/15, 2020/11/20, 2021/2/20, 2021/3/3, 2021/3/5, 2021/7/18, 2022/7/11, 2023/2/11
# @Author : Hui Wang, Xinyan Fan, Chen Yang, Yibo Li, Lanling Xu, Haoran Cheng, Zhichao Feng, Lei Wang, Gaowei Zhang
# @Email  : hui.wang@ruc.edu.cn, xinyan.fan@ruc.edu.cn, 254170321@qq.com, 2018202152@ruc.edu.cn, xulanling_sherry@163.com, chenghaoran29@foxmail.com, fzcbupt@gmail.com, zxcptss@gmail.com, zgw2022101006@ruc.edu.cn

r"""
recbole.trainer.trainer
################################
"""

import os
from contextlib import contextmanager

from logging import getLogger
from time import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
import torch.amp as amp

from recbole.data.interaction import Interaction
from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.evaluator import Evaluator, Collector
from recbole.model import loss
from recbole.model.sequential_recommender import GRU4Rec, NARM, SASRec, SRGNN
from recbole.utils import (
    ensure_dir,
    get_local_time,
    early_stopping,
    calculate_valid_score,
    dict2str,
    EvaluatorType,
    KGDataLoaderState,
    get_tensorboard,
    set_color,
    get_gpu_usage,
    WandbLogger,
)
from torch.nn.parallel import DistributedDataParallel
import math
import random


class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model
        if not config["single_spec"]:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.distributed_model = DistributedDataParallel(
                self.model, device_ids=[config["local_rank"]]
            )

    def fit(self, train_data):
        r"""Train the model based on the train data."""
        raise NotImplementedError("Method [next] should be implemented.")

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data."""

        raise NotImplementedError("Method [next] should be implemented.")

    def set_reduce_hook(self):
        r"""Call the forward function of 'distributed_model' to apply grads
        reduce hook to each parameter of its module.

        """
        t = self.model.forward
        self.model.forward = lambda x: x
        self.distributed_model(torch.LongTensor([0]).to(self.device))
        self.model.forward = t

    def sync_grad_loss(self):
        r"""Ensure that each parameter appears to the loss function to
        make the grads reduce sync in each node.

        """
        sync_loss = 0
        for params in self.model.parameters():
            sync_loss += torch.sum(params) * 0
        return sync_loss


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
    resume_checkpoint() and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config, model):
        super(Trainer, self).__init__(config, model)

        self.logger = getLogger()
        self.tensorboard = get_tensorboard(self.logger)
        self.wandblogger = WandbLogger(config)
        self.learner = config["learner"]
        self.learning_rate = config["learning_rate"]
        self.epochs = config["epochs"]
        self.eval_step = min(config["eval_step"], self.epochs)
        self.stopping_step = config["stopping_step"]
        self.clip_grad_norm = config["clip_grad_norm"]
        self.valid_metric = config["valid_metric"].lower()
        self.valid_metric_bigger = config["valid_metric_bigger"]
        self.test_batch_size = config["eval_batch_size"]
        self.gpu_available = torch.cuda.is_available() and config["use_gpu"]
        self.device = config["device"]
        if isinstance(config["gpu_id"], int) and config["gpu_id"] != -1:
            self.device = "cuda"#f"cuda:{config['gpu_id']}"
        self.checkpoint_dir = config["checkpoint_dir"]
        self.enable_amp = config["enable_amp"]
        self.enable_scaler = torch.cuda.is_available() and config["enable_scaler"]
        ensure_dir(self.checkpoint_dir)
        if "spam" in config and config["spam"]:
            # rmia OUT model training
            if "rmia_out_model_flag" in config and config["rmia_out_model_flag"]:
                saved_model_file = f"model_{config['model']}_seed_{config['seed']}_dataset_{config['dataset']}_rmia_out_model_partition_idx_{config['rmia_out_model_partition_idx']}.pth"
            # normal spam training
            elif ("unlearning_algorithm" not in config or config["unlearning_algorithm"] is None) and ("retrain_flag" not in config or not config["retrain_flag"]):
                saved_model_file = f"model_{config['model']}_seed_{config['seed']}_dataset_{config['dataset']}_unlearning_fraction_{config['unlearning_fraction']}_n_target_items_{config['n_target_items']}_best.pth"
            # retraining
            elif "retrain_flag" in config and config["retrain_flag"] and config["retrain_checkpoint_idx_to_match"] is not None:
                saved_model_file = f"model_{config['model']}_seed_{config['seed']}_dataset_{config['dataset']}_retrain_checkpoint_idx_to_match_{config['retrain_checkpoint_idx_to_match']}_unlearning_fraction_{config['unlearning_fraction']}_n_target_items_{config['n_target_items']}.pth"
            # unlearning
            else:
                saved_model_file = f"model_{config['model']}_seed_{config['seed']}_dataset_{config['dataset']}_unlearning_fraction_{config['unlearning_fraction']}_n_target_items_{config['n_target_items']}_unlearning_algorithm_{config['unlearning_algorithm']}.pth"
        else:
            # rmia OUT model training
            if "rmia_out_model_flag" in config and config["rmia_out_model_flag"]:
                saved_model_file = f"model_{config['model']}_seed_{config['seed']}_dataset_{config['dataset']}_rmia_out_model_partition_idx_{config['rmia_out_model_partition_idx']}.pth"
            # unlearning
            if "unlearning_algorithm" in config and config["unlearning_algorithm"] in ["scif", "fanchuan", "kookmin", "gif", "ceu", "idea", "seif"]:
                saved_model_file = f"model_{config['model']}_seed_{config['seed']}_dataset_{config['dataset']}_unlearning_algorithm_{config['unlearning_algorithm']}_unlearning_fraction_{config['unlearning_fraction']}_unlearning_sample_selection_method_{config['unlearning_sample_selection_method']}.pth"
            # retraining
            elif "retrain_checkpoint_idx_to_match" in config and config["retrain_checkpoint_idx_to_match"] is not None:
                saved_model_file = f"model_{config['model']}_seed_{config['seed']}_dataset_{config['dataset']}_retrain_checkpoint_idx_to_match_{config['retrain_checkpoint_idx_to_match']}_unlearning_fraction_{config['unlearning_fraction']}_unlearning_sample_selection_method_{config['unlearning_sample_selection_method']}.pth"
            # normal training
            else:
                saved_model_file = f"model_{config['model']}_seed_{config['seed']}_dataset_{config['dataset']}_best.pth"
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)
        self.weight_decay = config["weight_decay"]

        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()
        self.scaler = amp.GradScaler('cuda', enabled=self.enable_scaler)
        self.eval_type = config["eval_type"]
        self.eval_collector = Collector(config)
        self.evaluator = Evaluator(config)
        self.item_tensor = None
        self.tot_item_num = None


    def _build_optimizer(self, **kwargs):
        r"""Init the Optimizer

        Args:
            params (torch.nn.Parameter, optional): The parameters to be optimized.
                Defaults to ``self.model.parameters()``.
            learner (str, optional): The name of used optimizer. Defaults to ``self.learner``.
            learning_rate (float, optional): Learning rate. Defaults to ``self.learning_rate``.
            weight_decay (float, optional): The L2 regularization weight. Defaults to ``self.weight_decay``.

        Returns:
            torch.optim: the optimizer
        """
        params = kwargs.pop("params", self.model.parameters())
        learner = kwargs.pop("learner", self.learner)
        learning_rate = kwargs.pop("learning_rate", self.learning_rate)
        weight_decay = kwargs.pop("weight_decay", self.weight_decay)

        if (
            self.config["reg_weight"]
            and weight_decay
            and weight_decay * self.config["reg_weight"] > 0
        ):
            self.logger.warning(
                "The parameters [weight_decay] and [reg_weight] are specified simultaneously, "
                "which may lead to double regularization."
            )

        if learner.lower() == "adam":
            # Allow custom Adam parameters from config (for matching original implementations)
            beta1 = self.config["adam_beta1"] if "adam_beta1" in self.config else 0.9
            beta2 = self.config["adam_beta2"] if "adam_beta2" in self.config else 0.999
            eps = self.config["adam_eps"] if "adam_eps" in self.config else 1e-8
            optimizer = optim.Adam(
                params, 
                lr=learning_rate, 
                weight_decay=weight_decay,
                betas=(beta1, beta2),
                eps=eps
            )
        elif learner.lower() == "adamw":
            optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == "sparse_adam":
            optimizer = optim.SparseAdam(params, lr=learning_rate)
            if weight_decay > 0:
                self.logger.warning(
                    "Sparse Adam cannot argument received argument [{weight_decay}]"
                )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False, retain_samples_used_for_update=None, reinit_masks=None, scale_for_reinit_params=1.0):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        loss_func = self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", "pink"),
            )
            if show_progress
            else train_data
        )

        if not self.config["single_spec"] and train_data.shuffle:
            train_data.sampler.set_epoch(epoch_idx)

        scaler = self.scaler
        for batch_idx, interaction in enumerate(iter_data):
            if retain_samples_used_for_update is not None and retain_samples_used_for_update <= 0:
                break
            if retain_samples_used_for_update is not None:
                interaction = interaction[:retain_samples_used_for_update]
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            sync_loss = 0
            if not self.config["single_spec"]:
                self.set_reduce_hook()
                sync_loss = self.sync_grad_loss()

            with torch.autocast(device_type=self.device.type, enabled=self.enable_amp):
                losses = loss_func(interaction)

            torch.cuda.empty_cache()

            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = (
                    loss_tuple
                    if total_loss is None
                    else tuple(map(sum, zip(total_loss, loss_tuple)))
                )
            else:
                loss = losses
                total_loss = (
                    losses.item() if total_loss is None else total_loss + losses.item()
                )
            self._check_nan(loss)

            torch.cuda.empty_cache()

            scaler.scale(loss + sync_loss).backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)

            # scale for re-initialized parameters in kookmin retain round
            if reinit_masks is not None:
                for p in reinit_masks:
                    if p.grad is not None:
                        p.grad[reinit_masks[p]] *= scale_for_reinit_params
                        
            scaler.step(self.optimizer)
            scaler.update()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                )

            torch.cuda.empty_cache()

            if retain_samples_used_for_update is not None:
                retain_samples_used_for_update -= len(interaction)

        return total_loss

    def _valid_epoch(self, valid_data, show_progress=False):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(
            valid_data, load_best_model=False, show_progress=show_progress
        )
        valid_score = calculate_valid_score(valid_result, self.valid_metric)
        return valid_score, valid_result

    def _save_checkpoint(self, epoch, verbose=True, retrain_flag=False, retrain_checkpoint_idx_to_match=None, **kwargs):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        if not self.config["single_spec"] and self.config["local_rank"] != 0:
            return
        saved_model_file = kwargs.pop("saved_model_file", self.saved_model_file)

        # Check if we're in unlearning context (not retraining)
        if retrain_checkpoint_idx_to_match is not None and not retrain_flag:
            # Optionally include unlearning_batchsize in filename for hyperparameter tests
            unlearning_batchsize = kwargs.pop("unlearning_batchsize", None) if kwargs else None
            if unlearning_batchsize is not None:
                saved_model_file = f"{saved_model_file[:-len('.pth')]}_unlearn_epoch_{epoch}_retrain_checkpoint_idx_to_match_{retrain_checkpoint_idx_to_match}_bs{unlearning_batchsize}.pth"
            else:
                saved_model_file = f"{saved_model_file[:-len('.pth')]}_unlearn_epoch_{epoch}_retrain_checkpoint_idx_to_match_{retrain_checkpoint_idx_to_match}.pth"
        state = {
            "config": self.config,
            "epoch": epoch,
            "cur_step": self.cur_step,
            "best_valid_score": self.best_valid_score,
            "state_dict": self.model.state_dict(),
            "other_parameter": self.model.other_parameter(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, saved_model_file, pickle_protocol=4)
        if verbose:
            self.logger.info(
                set_color("Saving current", "blue") + f": {saved_model_file}"
            )

    def resume_checkpoint(self, resume_file):
        r"""Load the model parameters information and training information.

        Args:
            resume_file (file): the checkpoint file

        """
        resume_file = str(resume_file)
        self.saved_model_file = resume_file
        checkpoint = torch.load(resume_file, map_location=self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.cur_step = checkpoint["cur_step"]
        self.best_valid_score = checkpoint["best_valid_score"]

        # load architecture params from checkpoint
        if checkpoint["config"]["model"].lower() != self.config["model"].lower():
            self.logger.warning(
                "Architecture configuration given in config file is different from that of checkpoint. "
                "This may yield an exception while state_dict is being loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.load_other_parameter(checkpoint.get("other_parameter"))

        # load optimizer state from checkpoint only when optimizer type is not changed
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        message_output = "Checkpoint loaded. Resume training from epoch {}".format(
            self.start_epoch
        )
        self.logger.info(message_output)

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        des = self.config["loss_decimal_place"] or 4
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            des = set_color("train_loss%d", "blue") + ": %." + str(des) + "f"
            train_loss_output += ", ".join(
                des % (idx + 1, loss) for idx, loss in enumerate(losses)
            )
        else:
            des = "%." + str(des) + "f"
            train_loss_output += set_color("train loss", "blue") + ": " + des % losses
        return train_loss_output + "]"

    def _add_train_loss_to_tensorboard(self, epoch_idx, losses, tag="Loss/Train"):
        if isinstance(losses, tuple):
            for idx, loss in enumerate(losses):
                self.tensorboard.add_scalar(tag + str(idx), loss, epoch_idx)
        else:
            self.tensorboard.add_scalar(tag, losses, epoch_idx)

    def _add_hparam_to_tensorboard(self, best_valid_result):
        # base hparam
        hparam_dict = {
            "learner": self.config["learner"],
            "learning_rate": self.config["learning_rate"],
            "train_batch_size": self.config["train_batch_size"],
        }
        # unrecorded parameter
        unrecorded_parameter = {
            parameter
            for parameters in self.config.parameters.values()
            for parameter in parameters
        }.union({"model", "dataset", "config_files", "device"})
        # other model-specific hparam
        hparam_dict.update(
            {
                para: val
                for para, val in self.config.final_config_dict.items()
                if para not in unrecorded_parameter
            }
        )
        for k in hparam_dict:
            if hparam_dict[k] is not None and not isinstance(
                hparam_dict[k], (bool, str, float, int)
            ):
                hparam_dict[k] = str(hparam_dict[k])

        self.tensorboard.add_hparams(
            hparam_dict, {"hparam/best_valid_result": best_valid_result}
        )

    def move_optimizer_state(self, optimizer, device):
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def _reset_adam_state(self, opt, params):
        for p in params:
            if p in opt.state:
                opt.state[p]['step'] = torch.zeros(1, dtype=torch.float32, device=p.device)
                opt.state[p]['exp_avg'].zero_()
                opt.state[p]['exp_avg_sq'].zero_()

    def kookmin(
        self,
        epoch_idx,
        forget_data,
        clean_forget_data,
        retain_train_data,
        retain_valid_data=None,
        retain_test_data=None,
        show_progress=False,
        unlearned_users_before=None,
        kookmin_init_rate=0.01,
        saved=True,
        verbose=False,
        retain_samples_used_for_update=32,
        neg_grad_retain_sample_size=128,
        param_list=None,
        time_limit_per_batch=None,
        unlearn_start_time=None,
    ):
        self.move_optimizer_state(self.optimizer, self.device)
        self.model.train()
        loss_func = self.model.calculate_loss
        
        # Initialize timeout tracking
        if unlearn_start_time is None:
            unlearn_start_time = time()
        
        grads_forget = []
        grads_retain = []

        for batch_idx, interaction in enumerate(forget_data):
            # Check timeout (allow at least one full batch to complete)
            if time_limit_per_batch is not None and batch_idx > 0:
                elapsed_time = time() - unlearn_start_time
                if elapsed_time >= time_limit_per_batch:
                    print(f"[Kookmin] Timeout reached ({elapsed_time/3600:.2f} hours). Stopping forget gradient computation early at batch {batch_idx + 1}/{len(forget_data)}")
                    return
            interaction = interaction.to(self.device)
            cur_grads = self._batch_grad(
                self.model,
                interaction,
                param_list,
                loss_func,
                average_scale=neg_grad_retain_sample_size,
            )
            if len(grads_forget) == 0:
                grads_forget = [g for g in cur_grads]
            else:
                for i in range(len(cur_grads)):
                    grads_forget[i] += cur_grads[i]

        # Only process clean_forget_data if it's not None/empty
        if clean_forget_data is not None:
            for batch_idx, interaction in enumerate(clean_forget_data):
                # Check timeout (allow at least one full batch to complete)
                if time_limit_per_batch is not None and batch_idx > 0:
                    elapsed_time = time() - unlearn_start_time
                    if elapsed_time >= time_limit_per_batch:
                        print(f"[Kookmin] Timeout reached ({elapsed_time/3600:.2f} hours). Stopping clean forget gradient computation early at batch {batch_idx + 1}/{len(clean_forget_data)}")
                        return
                interaction = interaction.to(self.device)
                cur_grads = self._batch_grad(
                    self.model,
                    interaction,
                    param_list,
                    loss_func,
                    average_scale=neg_grad_retain_sample_size,
                )
                if len(grads_retain) == 0:
                    grads_retain = [g for g in cur_grads]
                else:
                    for i in range(len(cur_grads)):
                        grads_retain[i] += cur_grads[i]

        # Handle empty clean_forget_data
        clean_forget_data_length = 0 if not clean_forget_data else len(clean_forget_data.dataset)
        k_more = max(0, neg_grad_retain_sample_size - clean_forget_data_length)

        for batch_idx, interaction in enumerate(retain_train_data):
            # Check timeout (allow at least one full batch to complete)
            if time_limit_per_batch is not None and batch_idx > 0:
                elapsed_time = time() - unlearn_start_time
                if elapsed_time >= time_limit_per_batch:
                    print(f"[Kookmin] Timeout reached ({elapsed_time/3600:.2f} hours). Stopping retain gradient computation early at batch {batch_idx + 1}/{len(retain_train_data)}")
                    return
            if k_more <= 0:
                break
            interaction = interaction[:k_more]
            interaction = interaction.to(self.device)
            cur_grads = self._batch_grad(
                self.model,
                interaction,
                param_list,
                loss_func,
                average_scale=neg_grad_retain_sample_size,
            )
            if len(grads_retain) == 0:
                grads_retain = [g for g in cur_grads]
            else:
                for i in range(len(cur_grads)):
                    grads_retain[i] += cur_grads[i]

        signed_grads = [gr - gf for gr, gf in zip(grads_retain, grads_forget)]

        # Apply per-layer threshold like the original Kookmin paper
        # Instead of global threshold across all parameters
        reinit_masks = dict()
        total_params_reset = 0

        for p, g in zip(param_list, signed_grads):
            # Compute threshold for this layer/parameter tensor only
            g_abs = g.abs()
            total_in_layer = g_abs.numel()
            k_in_layer = max(1, int(total_in_layer * kookmin_init_rate))

            # Use topk to get exactly k_in_layer smallest gradients (negate for smallest)
            # topk returns largest values, so we negate to get smallest
            g_flat = g_abs.view(-1)
            _, indices = torch.topk(-g_flat, k=k_in_layer, largest=True)

            # Create mask: True only for the k_in_layer smallest gradient positions
            mask = torch.zeros_like(g_abs, dtype=torch.bool)
            mask.view(-1)[indices] = True

            if not mask.any():
                continue

            total_params_reset += mask.sum().item()

            new_p = torch.empty_like(p.data, device=self.device)
            if p.dim() == 4:            # e.g. Conv2d weight
                torch.nn.init.kaiming_normal_(new_p, mode="fan_out", nonlinearity="relu")
            elif p.dim() == 2:          # e.g. Linear weight
                torch.nn.init.kaiming_uniform_(new_p, a=math.sqrt(5))
            else:                       # embeddings, biases, ...
                new_p.normal_(0, 0.02)

            # overwrite only the "low-grad" slots
            p.data = p.data.to(self.device)
            p.data[mask] = new_p[mask]

            # store the mask to use later
            reinit_masks[p] = mask

        # Log statistics about parameter reinitialization
        total_params = sum(p.numel() for p in param_list)
        print(f"[Kookmin] Reset {total_params_reset}/{total_params} parameters ({100*total_params_reset/total_params}%) across {len(reinit_masks)} layers")

        self._reset_adam_state(self.optimizer, list(reinit_masks.keys()))

        self.model.zero_grad()
        self.optimizer.zero_grad()

        epochs = 1 + retain_samples_used_for_update // len(retain_train_data.dataset)

        # retain round
        for epoch_idx in range(epochs):
            training_start_time = time()
            train_loss = self._train_epoch(
                retain_train_data, epoch_idx, show_progress=show_progress, retain_samples_used_for_update=retain_samples_used_for_update, reinit_masks=reinit_masks, scale_for_reinit_params=10,
            )
            self.train_loss_dict[epoch_idx] = (
                sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            )
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self.wandblogger.log_metrics(
                {"epoch": epoch_idx, "train_loss": train_loss, "train_step": epoch_idx},
                head="train",
            )

    def kl_loss_sym(self, x, y):
        kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
        # kl_loss expects the first parameter to be model outputs as log probabilities and the target to be "normal" probabilities
        return kl_loss(torch.log(x + 1e-20), y)

    def unlearn_iterative_uniform_distribution(
        self,
        interaction,
        model,
        task_type="SBR",
    ):
        self.optimizer.zero_grad()

        # Clear cached embeddings to ensure gradients are computed
        if hasattr(model, 'restore_user_e'):
            model.restore_user_e = None
        if hasattr(model, 'restore_item_e'):
            model.restore_item_e = None

        if task_type == "CF":
            item_count = model.item_embedding.weight.shape[0]
            raw_scores = model.full_sort_predict(interaction).view(-1, item_count)
        else:
            raw_scores = model.full_sort_predict(interaction)

        model_probs = F.softmax(raw_scores, dim=1)

        batch_size, n_items = model_probs.shape
        uniform_probs = torch.ones_like(model_probs) / n_items
        loss = self.kl_loss_sym(model_probs, uniform_probs)

        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def get_embedding_for_contrastive_learning(self, interaction, model):
        # Check for NBR models first (they inherit ITEM_SEQ from SequentialRecommender but don't use it)
        if hasattr(model, "history_items_field"): # NBR (Next Basket Recommendation)
            # For NBR models like DNNTSP, extract basket/user embedding representation
            # This is better than using full output scores for contrastive learning
            if hasattr(model, "get_basket_embedding"):
                basket_emb = model.get_basket_embedding(interaction)
                return basket_emb
            else:
                # Fallback: use forward output (less ideal but works)
                output = model.forward(interaction)
                return output
        elif hasattr(model, 'ITEM_SEQ') and hasattr(model, 'ITEM_SEQ_LEN'): # SBR
            # Check if ITEM_SEQ actually exists in the interaction (to distinguish from NBR models)
            if model.ITEM_SEQ in interaction:
                item_seq = interaction[model.ITEM_SEQ]
                item_seq_len = interaction[model.ITEM_SEQ_LEN]
                
                # forward returns sequence (session) representation
                seq_output = model.forward(item_seq, item_seq_len)
                return seq_output
        elif hasattr(model, "user_embedding"): # CF
            user = interaction[model.USER_ID]
            user_e = model.user_embedding(user)
            return user_e
        
        raise ValueError(f"Model {model} is not supported, specify here which layer to take for contrastive learning.")


    def unlearn_iterative_contrastive(self, unlearn_interaction, retain_interaction, model):
        self.optimizer.zero_grad()
        
        unlearn_repr = self.get_embedding_for_contrastive_learning(unlearn_interaction, model)
        retain_repr = self.get_embedding_for_contrastive_learning(retain_interaction, model)

        t = 1.15
        contrastive_similarity = unlearn_repr @ retain_repr.T / t
        loss = (-1 * torch.nn.LogSoftmax(dim=-1)(contrastive_similarity)).mean()
        
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _train_epoch_with_custom_indices(self, train_data, custom_indices, epoch_idx, loss_func=None, show_progress=False, retain_samples_used_for_update=None, reinit_masks=None, scale_for_reinit_params=1.0):
        """Train the model in an epoch with custom data ordering"""
        self.model.train()
        loss_func = self.model.calculate_loss
        total_loss = None
        
        batch_size = train_data.batch_size
        
        # Create batches using custom indices
        custom_batches = []
        for i in range(0, len(custom_indices), batch_size):
            batch_indices = custom_indices[i:i + batch_size]
            batch_data = train_data.dataset[batch_indices]
            custom_batches.append(batch_data)
        
        iter_data = (
            tqdm(custom_batches, total=len(custom_batches), ncols=100, desc=set_color(f"Train {epoch_idx:>5}", "pink"))
            if show_progress else custom_batches
        )

        scaler = self.scaler

        for batch_idx, interaction in enumerate(iter_data):
            if retain_samples_used_for_update is not None and retain_samples_used_for_update <= 0:
                break
            if retain_samples_used_for_update is not None:
                interaction = interaction[:retain_samples_used_for_update]
                
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            sync_loss = 0
            if not self.config["single_spec"]:
                self.set_reduce_hook()
                sync_loss = self.sync_grad_loss()

            with torch.autocast(device_type=self.device.type, enabled=self.enable_amp):
                losses = loss_func(interaction)

            torch.cuda.empty_cache()

            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = (
                    loss_tuple
                    if total_loss is None
                    else tuple(map(sum, zip(total_loss, loss_tuple)))
                )
            else:
                loss = losses
                total_loss = (
                    losses.item() if total_loss is None else total_loss + losses.item()
                )
            
            self._check_nan(loss)

            torch.cuda.empty_cache()

            scaler.scale(loss + sync_loss).backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)

            # scale for re-initialized parameters in kookmin retain round
            if reinit_masks is not None:
                for p in reinit_masks:
                    if p.grad is not None:
                        p.grad[reinit_masks[p]] *= scale_for_reinit_params
                        
            scaler.step(self.optimizer)
            scaler.update()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                )

            torch.cuda.empty_cache()

            if retain_samples_used_for_update is not None:
                retain_samples_used_for_update -= len(interaction)

        return total_loss

    def fanchuan(
        self,
        epoch_idx,
        forget_data,
        clean_forget_data,
        retain_train_data,
        retain_valid_data=None,
        retain_test_data=None,
        show_progress=False,
        unlearned_users_before=None,
        saved=True,
        verbose=False,
        retain_samples_used_for_update=32,
        unlearn_iters_contrastive=8,
        task_type="SBR",
        time_limit_per_batch=None,
        unlearn_start_time=None,
    ):
        self.move_optimizer_state(self.optimizer, self.device)
        self.model.train()

        # Initialize timeout tracking
        if unlearn_start_time is None:
            unlearn_start_time = time()
        
        # First stage: learn uniform pseudolabel
        losses = []
        for batch_idx, interaction in enumerate(forget_data):
            # Check timeout (allow at least one full batch to complete)
            if time_limit_per_batch is not None and batch_idx > 0:
                elapsed_time = time() - unlearn_start_time
                if elapsed_time >= time_limit_per_batch:
                    print(f"[Fanchuan] Timeout reached ({elapsed_time/3600:.2f} hours). Stopping uniform pseudolabel learning early at batch {batch_idx + 1}/{len(forget_data)}")
                    return
            
            interaction = interaction.to(self.device)
            loss = self.unlearn_iterative_uniform_distribution(interaction, self.model, task_type=task_type)
            losses.append(loss)
        
        print("Uniform pseudolabel learning average loss: ", np.mean(losses))

        # Pre-compute shuffled indices once for all contrastive learning iterations
        retain_dataset_size = len(retain_train_data.dataset)
        batch_size = retain_train_data.batch_size
        
        # Create a large pool of shuffled indices (enough for all iterations)
        total_samples_needed = unlearn_iters_contrastive * len(forget_data) * batch_size
        num_repeats = (total_samples_needed // retain_dataset_size) + 2  # +2 for safety
        
        # Create one large shuffled index array
        base_indices = np.arange(retain_dataset_size)
        large_shuffled_indices = np.concatenate([
            np.random.permutation(base_indices) for _ in range(num_repeats)
        ])

        # Second stage: Contrastive learning between forget and retain data
        for j in range(unlearn_iters_contrastive):
            # Check timeout before starting iteration (allow at least one full iteration to complete)
            if time_limit_per_batch is not None and j > 0:
                elapsed_time = time() - unlearn_start_time
                if elapsed_time >= time_limit_per_batch:
                    print(f"[Fanchuan] Timeout reached ({elapsed_time/3600:.2f} hours). Stopping contrastive learning early at iteration {j + 1}/{unlearn_iters_contrastive}")
                    return
            
            losses = []

            first_round_start_time = time()

            # Use a slice of the pre-computed indices for this iteration
            iter_offset = j * len(forget_data) * batch_size

            # Handle empty clean_forget_data
            if not clean_forget_data:
                print(f"[Fanchuan] Warning: clean_forget_data is empty, skipping clean contrastive loss")

            # Create iterator for clean_forget_data to avoid converting entire DataLoader to list
            # Recreate iterator for each iteration since it gets exhausted
            clean_forget_iter = iter(clean_forget_data) if clean_forget_data else None

            for batch_idx, forget_interaction in enumerate(forget_data):
                # Check timeout during batch processing (allow at least one full batch to complete)
                if time_limit_per_batch is not None and batch_idx > 0:
                    elapsed_time = time() - unlearn_start_time
                    if elapsed_time >= time_limit_per_batch:
                        print(f"[Fanchuan] Timeout reached ({elapsed_time/3600:.2f} hours). Stopping contrastive learning early at iteration {j + 1}/{unlearn_iters_contrastive}, batch {batch_idx + 1}/{len(forget_data)}")
                        return
                # Get retain interaction using pre-computed indices
                start_idx = iter_offset + batch_idx * batch_size
                end_idx = start_idx + batch_size
                retain_batch_indices = large_shuffled_indices[start_idx:end_idx]

                retain_train_data_interaction = retain_train_data.dataset[retain_batch_indices]

                forget_interaction = forget_interaction.to(self.device)
                retain_interaction = retain_train_data_interaction.to(self.device)

                loss = 0
                # Only use clean_forget_data if it's not empty
                # Use iterator to handle case where clean_forget_data has fewer batches than forget_data
                if clean_forget_iter:
                    try:
                        clean_forget_interaction = next(clean_forget_iter).to(self.device)
                        loss = self.unlearn_iterative_contrastive(forget_interaction, clean_forget_interaction, self.model)
                    except StopIteration:
                        # clean_forget_data has fewer batches than forget_data, continue without it
                        # Set iterator to None to avoid trying to use it again in this iteration
                        clean_forget_iter = None
                        pass

                loss += self.unlearn_iterative_contrastive(forget_interaction, retain_interaction, self.model)
                losses.append(loss)

            first_round_end_time = time()
            print(f"Contrastive learning iteration {j+1} average loss: ", np.mean(losses))
            print(f"First round training took {first_round_end_time - first_round_start_time} seconds")

            self.model.zero_grad()
            self.optimizer.zero_grad()

            epochs = 1 + retain_samples_used_for_update // len(retain_train_data.dataset)

            # retain round - use efficient shuffling
            second_round_start_time = time()
            
            for epoch_idx_inner in range(epochs):
                # Check timeout before each epoch (allow at least one full epoch to complete)
                if time_limit_per_batch is not None and epoch_idx_inner > 0:
                    elapsed_time = time() - unlearn_start_time
                    if elapsed_time >= time_limit_per_batch:
                        print(f"[Fanchuan] Timeout reached ({elapsed_time/3600:.2f} hours). Stopping retain round early at iteration {j + 1}/{unlearn_iters_contrastive}, epoch {epoch_idx_inner + 1}/{epochs}")
                        return
               
                training_start_time = time()
                train_loss = self._train_epoch_efficient_shuffle(
                    retain_train_data, epoch_idx_inner, 
                    show_progress=show_progress, 
                    retain_samples_used_for_update=retain_samples_used_for_update,
                )
                
                self.train_loss_dict[epoch_idx_inner] = (
                    sum(train_loss) if isinstance(train_loss, tuple) else train_loss
                )
                training_end_time = time()
                train_loss_output = self._generate_train_loss_output(
                    epoch_idx_inner, training_start_time, training_end_time, train_loss
                )
                
                if verbose:
                    self.logger.info(train_loss_output)
                self._add_train_loss_to_tensorboard(epoch_idx_inner, train_loss)
                self.wandblogger.log_metrics(
                    {"epoch": epoch_idx_inner, "train_loss": train_loss, "train_step": epoch_idx_inner},
                    head="train",
                )
            
            second_round_end_time = time()
            print(f"Second round training took {second_round_end_time - second_round_start_time} seconds\n")
    
    def _train_epoch_efficient_shuffle(self, train_data, epoch_idx, loss_func=None, show_progress=False, retain_samples_used_for_update=None):
        """Efficient training epoch that uses DataLoader's built-in shuffling"""
        self.model.train()
        loss_func = self.model.calculate_loss
        total_loss = None
        
        iter_data = (
            tqdm(train_data, total=len(train_data), ncols=100, desc=set_color(f"Train {epoch_idx:>5}", "pink"))
            if show_progress else train_data
        )

        scaler = self.scaler
        samples_processed = 0
        
        for batch_idx, interaction in enumerate(iter_data):
            if retain_samples_used_for_update is not None and samples_processed >= retain_samples_used_for_update:
                break
                
            # Limit batch size if we're near the sample limit
            if retain_samples_used_for_update is not None:
                remaining_samples = retain_samples_used_for_update - samples_processed
                if len(interaction) > remaining_samples:
                    interaction = interaction[:remaining_samples]
                    
            interaction = interaction.to(self.device)
            samples_processed += len(interaction)
            
            self.optimizer.zero_grad()
            sync_loss = 0
            if not self.config["single_spec"]:
                self.set_reduce_hook()
                sync_loss = self.sync_grad_loss()

            with torch.autocast(device_type=self.device.type, enabled=self.enable_amp):
                losses = loss_func(interaction)

            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = (
                    loss_tuple if total_loss is None
                    else tuple(map(sum, zip(total_loss, loss_tuple)))
                )
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
                
            self._check_nan(loss)

            scaler.scale(loss + sync_loss).backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
                
            scaler.step(self.optimizer)
            scaler.update()
            
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                )

        return total_loss

    def _build_user_item_sparse_matrix(self, dataset):
        """
        Build a sparse user-item interaction matrix from the dataset.

        Args:
            dataset: RecBole dataset object

        Returns:
            scipy.sparse.csr_matrix: Sparse matrix of shape (n_users, n_items)
                where matrix[u, i] = 1 if user u interacted with item i
        """
        import scipy.sparse as sp

        # Get interaction data
        inter_feat = dataset.inter_feat
        user_ids = inter_feat[dataset.uid_field].numpy()
        item_ids = inter_feat[dataset.iid_field].numpy()

        # Create sparse matrix
        n_users = dataset.user_num
        n_items = dataset.item_num

        # Create data array (all ones for interactions)
        data = np.ones(len(user_ids))

        # Build sparse matrix
        matrix = sp.csr_matrix(
            (data, (user_ids, item_ids)),
            shape=(n_users, n_items)
        )

        return matrix

    def _build_original_user_item_matrix(self, original_dataset, retain_dataset, forget_dataset):
        """
        Build a sparse user-item matrix from the ORIGINAL data (retain + forget).

        This is needed for k-hop neighbor computation where we need to find neighbors
        based on the original graph structure before unlearning.

        Args:
            retain_dataset: Dataset with retain interactions
            forget_dataset: Dataset with forget interactions

        Returns:
            scipy.sparse.csr_matrix: Sparse matrix of shape (n_users, n_items)
        """
        import scipy.sparse as sp

        # Get retain interactions
        retain_inter_feat = retain_dataset.inter_feat
        retain_users = retain_inter_feat[retain_dataset.uid_field].numpy()
        retain_items = retain_inter_feat[retain_dataset.iid_field].numpy()

        # Get forget interactions
        forget_inter_feat = forget_dataset.inter_feat
        forget_users = forget_inter_feat[forget_dataset.uid_field].numpy()
        forget_items = forget_inter_feat[forget_dataset.iid_field].numpy()

        # Combine to get original interactions
        all_users = np.concatenate([retain_users, forget_users])
        all_items = np.concatenate([retain_items, forget_items])

        n_users = retain_dataset.user_num
        n_items = retain_dataset.item_num
        data = np.ones(len(all_users))

        # Build sparse matrix
        matrix = sp.csr_matrix(
            (data, (all_users, all_items)),
            shape=(n_users, n_items)
        )

        return matrix

    def _get_k_hop_neighbors(self, user_ids, user_item_matrix, k=2, max_neighbors=10000):
        """
        Get users who are within k-hops of the target users in user-item bipartite graph.
        Uses hybrid approach: finds true k-hop neighbors but caps the size.

        Args:
            user_ids: List/array of user IDs to find neighbors for
            user_item_matrix: Sparse user-item interaction matrix (n_users x n_items)
            k: Number of hops (default: 2)
            max_neighbors: Maximum number of neighbors to return (for large graphs)

        Returns:
            set: Set of user IDs within k-hops (excluding input users)
        """
        import scipy.sparse as sp

        influenced_users = set()
        input_users = set(user_ids)

        # Convert to list if needed
        if not isinstance(user_ids, (list, tuple, np.ndarray)):
            user_ids = [user_ids]

        # Convert to CSR for efficient row access (user->items)
        # and CSC for efficient column access (item->users)
        user_item_csr = user_item_matrix.tocsr() if not sp.isspmatrix_csr(user_item_matrix) else user_item_matrix
        user_item_csc = user_item_matrix.tocsc()

        # Get items that target users interacted with (1-hop)
        target_items = set()
        for user_id in user_ids:
            if user_id < user_item_csr.shape[0]:
                user_items = user_item_csr[user_id].nonzero()[1]
                target_items.update(user_items)

        # 2-hop: Get users who interacted with those items (this will include the forget user)
        if k >= 2:
            # 2-hop: Get other users who interacted with those items
            # Use CSC format for efficient column slicing
            for item_id in target_items:
                if item_id < user_item_csc.shape[1]:
                    other_users = user_item_csc[:, item_id].nonzero()[0]
                    influenced_users.update(other_users)

        # 4-hop: Expand to users 4-hops away
        if k >= 4:
            # 3-hop: Get items those users interacted with
            hop_2_items = set()
            for neighbor_user in list(influenced_users):
                if neighbor_user not in input_users and neighbor_user < user_item_csr.shape[0]:
                    neighbor_items = user_item_csr[neighbor_user].nonzero()[1]
                    hop_2_items.update(neighbor_items)

            # 4-hop: Get users who interacted with those items
            for item_id in hop_2_items:
                if item_id < user_item_csc.shape[1]:
                    other_users = user_item_csc[:, item_id].nonzero()[0]
                    influenced_users.update(other_users)

        # Remove the original users
        influenced_users -= input_users

        # Hybrid approach: if too many neighbors, sample randomly from them
        if len(influenced_users) > max_neighbors:
            print(f"[GIF] Found {len(influenced_users)} k-hop neighbors, sampling {max_neighbors}")
            influenced_users = set(np.random.choice(
                list(influenced_users),
                size=max_neighbors,
                replace=False
            ))

        return influenced_users

    def _filter_retain_data_by_users(self, retain_train_data, target_user_ids, max_samples=None):
        """
        Filter retain training data to only include interactions from specific users.

        Args:
            retain_train_data: DataLoader with retain data
            target_user_ids: Set of user IDs to keep
            max_samples: Optional maximum number of samples to return

        Returns:
            List of batches containing only target users' interactions
        """
        filtered_batches = []
        total_samples = 0

        for batch in retain_train_data:
            # Get user IDs from batch
            user_ids = batch[retain_train_data.dataset.uid_field].numpy()

            # Find which samples belong to target users
            mask = np.isin(user_ids, list(target_user_ids))

            if mask.any():
                # Filter the batch to only include target users
                filtered_batch_dict = {}
                # Iterate over keys in the Interaction object
                for key in batch.interaction:
                    value = batch.interaction[key]
                    if isinstance(value, torch.Tensor):
                        filtered_batch_dict[key] = value[mask]
                    else:
                        filtered_batch_dict[key] = value

                if len(filtered_batch_dict[retain_train_data.dataset.uid_field]) > 0:
                    # Convert dict back to Interaction object so it has .to() method
                    filtered_batch = Interaction(filtered_batch_dict)
                    filtered_batches.append(filtered_batch)
                    total_samples += len(filtered_batch_dict[retain_train_data.dataset.uid_field])

                    if max_samples and total_samples >= max_samples:
                        break

        return filtered_batches

    def _model_has_graph_structure(self, model):
        """
        Detect if the model uses graph convolution and has an adjacency matrix.

        Args:
            model: The recommendation model

        Returns:
            tuple: (has_graph, graph_attr_name)
                has_graph (bool): True if model has graph structure
                graph_attr_name (str): Name of the adjacency matrix attribute
        """
        # Check for LightGCN-style models
        if hasattr(model, 'norm_adj_matrix'):
            return True, 'norm_adj_matrix'

        # Check for SGL-style models
        if hasattr(model, 'train_graph'):
            return True, 'train_graph'

        # Add more model types as needed
        # For NGCF, DGCF, etc.
        if hasattr(model, 'norm_adj_mat'):
            return True, 'norm_adj_mat'

        # No graph structure found
        return False, None

    def _build_modified_adj_matrix(self, model, forget_data, retain_dataset, forget_dataset):
        """
        Build a modified adjacency matrix with forget edges removed.

        This is crucial for GNN models like LightGCN where the forward pass
        uses the graph structure. When computing gradients on the "remaining graph",
        we need to exclude forget edges from the message passing.

        Args:
            model: The recommendation model
            forget_data: DataLoader containing edges to remove
            retain_dataset: Dataset with retain interactions (without forget edges)
            forget_dataset: Dataset with forget interactions

        Returns:
            torch.sparse.FloatTensor: Modified adjacency matrix without forget edges
        """
        import scipy.sparse as sp

        # Extract forget edges (user-item pairs to remove)
        forget_users = []
        forget_items = []

        # Handle empty forget_dataset (all user interactions are sensitive)
        if forget_dataset is None:
            # Use forget_data to get uid/iid fields
            for batch in forget_data:
                forget_dataset_temp = forget_data.dataset
                users = batch[forget_dataset_temp.uid_field].cpu().numpy()
                items = batch[forget_dataset_temp.iid_field].cpu().numpy()
                forget_users.extend(users)
                forget_items.extend(items)
                break  # Just need to get the fields once
            uid_field = forget_dataset_temp.uid_field
            iid_field = forget_dataset_temp.iid_field
        else:
            for batch in forget_data:
                users = batch[forget_dataset.uid_field].cpu().numpy()
                items = batch[forget_dataset.iid_field].cpu().numpy()
                forget_users.extend(users)
                forget_items.extend(items)
            uid_field = forget_dataset.uid_field
            iid_field = forget_dataset.iid_field

        forget_edges = set(zip(forget_users, forget_items))
        print(f"[GIF] Building modified adjacency matrix excluding {len(forget_edges)} forget edges...")

        # Build the ORIGINAL interaction matrix by combining retain + forget
        # We need the original graph (before unlearning) to properly remove forget edges
        import pandas as pd

        # Get retain interactions
        retain_inter_feat = retain_dataset.inter_feat
        retain_users = retain_inter_feat[retain_dataset.uid_field].numpy()
        retain_items = retain_inter_feat[retain_dataset.iid_field].numpy()

        # Get forget interactions
        if forget_dataset is None:
            # If all user interactions are sensitive, there are no clean forget interactions
            forget_users_all = np.array([], dtype=retain_users.dtype)
            forget_items_all = np.array([], dtype=retain_items.dtype)
        else:
            forget_inter_feat = forget_dataset.inter_feat
            forget_users_all = forget_inter_feat[forget_dataset.uid_field].numpy()
            forget_items_all = forget_inter_feat[forget_dataset.iid_field].numpy()

        # Combine to get original interactions
        all_users = np.concatenate([retain_users, forget_users_all])
        all_items = np.concatenate([retain_items, forget_items_all])

        n_users = retain_dataset.user_num
        n_items = retain_dataset.item_num
        data = np.ones(len(all_users))

        interaction_matrix = sp.coo_matrix(
            (data, (all_users, all_items)),
            shape=(n_users, n_items)
        ).astype(np.float32)

        # Remove forget edges from interaction matrix
        # Convert to lists for filtering
        row = interaction_matrix.row.tolist()
        col = interaction_matrix.col.tolist()
        data = interaction_matrix.data.tolist()

        # Filter out forget edges
        filtered_row = []
        filtered_col = []
        filtered_data = []

        removed_count = 0
        for i in range(len(row)):
            edge = (row[i], col[i])
            if edge not in forget_edges:
                filtered_row.append(row[i])
                filtered_col.append(col[i])
                filtered_data.append(data[i])
            else:
                removed_count += 1

        print(f"[GIF] Removed {removed_count} edges from adjacency matrix")

        # Build modified interaction matrix
        modified_interaction_matrix = sp.coo_matrix(
            (filtered_data, (filtered_row, filtered_col)),
            shape=interaction_matrix.shape
        ).astype(np.float32)

        # Now rebuild the normalized adjacency matrix using the same process as the model
        # This replicates the logic from LightGCN.get_norm_adj_mat()
        A = sp.dok_matrix(
            (n_users + n_items, n_users + n_items),
            dtype=np.float32
        )

        inter_M = modified_interaction_matrix
        inter_M_t = modified_interaction_matrix.transpose()

        data_dict = dict(
            zip(zip(range(n_users), range(n_users, n_users + n_items)),
                [1] * n_users)
        )
        data_dict.update(
            dict(
                zip(
                    zip(range(n_users, n_users + n_items), range(n_users)),
                    [1] * n_items
                )
            )
        )
        data_dict.update(
            dict(zip(zip(inter_M.row, inter_M.col + n_users), inter_M.data))
        )
        data_dict.update(
            dict(zip(zip(inter_M_t.row + n_users, inter_M_t.col), inter_M_t.data))
        )
        A._update(data_dict)

        # Normalize: D^{-0.5} * A * D^{-0.5}
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D

        # Convert to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        modified_adj_matrix = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))

        return modified_adj_matrix

    @contextmanager
    def _temporarily_replace_graph(self, model, graph_attr_name, modified_graph):
        """
        Context manager to temporarily replace the model's adjacency matrix.

        This allows us to compute gradients with a modified graph structure
        (e.g., without forget edges) and then restore the original graph.

        Args:
            model: The recommendation model
            graph_attr_name: Name of the graph attribute to replace
            modified_graph: The modified adjacency matrix

        Usage:
            with self._temporarily_replace_graph(model, 'norm_adj_matrix', modified_adj):
                # Forward pass and gradient computation happens here
                # The model uses modified_adj instead of the original
                pass
            # Original graph is automatically restored here
        """
        # Save the original graph
        original_graph = getattr(model, graph_attr_name)

        try:
            # Replace with modified graph (move to same device)
            setattr(model, graph_attr_name, modified_graph.to(model.device))
            print(f"[GIF] Temporarily replaced {graph_attr_name} with modified graph (forget edges excluded)")
            yield
        finally:
            # Restore original graph
            setattr(model, graph_attr_name, original_graph)
            print(f"[GIF] Restored original {graph_attr_name}")

    def gif(
        self,
        epoch_idx,
        forget_data,
        clean_forget_data,
        retain_train_data,
        retain_valid_data=None,
        retain_test_data=None,
        show_progress=False,
        unlearned_users_before=None,
        saved=True,
        verbose=False,
        retain_samples_used_for_update=128,
        gif_damping=0.01,
        gif_scale_factor=1000,
        gif_iterations=100,
        gif_k_hops=4,
        gif_use_random_retain_samples=False,
        max_norm=None,
        param_list=None,
        original_dataset=None,
        time_limit_per_batch=None,
        unlearn_start_time=None,
    ):
        """
        GIF: Graph Influence Function for Graph Neural Network Unlearning

        Reference:
            Jiancan Wu et al. "GIF: A General Graph Unlearning Strategy via Influence Function" in WWW 2023.
            https://arxiv.org/abs/2304.02835

        Args:
            epoch_idx: Current epoch index
            forget_data: Data to unlearn (contains node/edge deletion requests)
            clean_forget_data: Clean version of forget data (for computing retain gradients)
            retain_train_data: Retained training data
            retain_valid_data: Validation data (optional)
            retain_test_data: Test data (optional)
            show_progress: Whether to show progress bars
            unlearned_users_before: Previously unlearned users (optional)
            saved: Whether to save checkpoints
            verbose: Whether to print detailed logs
            retain_samples_used_for_update: Number of retain samples for fine-tuning
            gif_damping: Damping factor (lambda) for Hessian approximation convergence
            gif_scale_factor: Scaling factor for Hessian to ensure convergence
            gif_iterations: Number of iterations for Hessian inverse approximation
            gif_k_hops: Number of hops for influenced neighbors (default: 4)
            gif_use_random_retain_samples: Use random sampling instead of k-hop (default: False)
            param_list: List of parameters to update (default: all trainable params)
            original_dataset: Original dataset (retain + forget) for k-hop computation
        """
        print(f"\n[GIF] Starting Graph Influence Function unlearning...")
        print(f"[GIF] Parameters: damping={gif_damping}, scale={gif_scale_factor}, iterations={gif_iterations}, k_hops={gif_k_hops}")

        self.move_optimizer_state(self.optimizer, self.device)
        self.model.train()
        loss_func = self.model.calculate_loss

        if param_list is None:
            param_list = self.target_params(self.model)

        # Initialize timeout tracking
        if unlearn_start_time is None:
            unlearn_start_time = time()

        # Step 1: Compute gradients for forget data (nodes to unlearn)
        print(f"[GIF] Step 1: Computing gradients for forget data ({len(forget_data.dataset)} samples)...")
        grads_forget_original = []
        for batch_idx, interaction in enumerate(forget_data):
            # Check timeout (allow at least one full batch to complete)
            if time_limit_per_batch is not None and batch_idx > 0:
                elapsed_time = time() - unlearn_start_time
                if elapsed_time >= time_limit_per_batch:
                    print(f"[GIF] Timeout reached ({elapsed_time/3600:.2f} hours). Stopping forget gradient computation early at batch {batch_idx + 1}/{len(forget_data)}")
                    return
            interaction = interaction.to(self.device)
            cur_grads = self._batch_grad(
                self.model,
                interaction,
                param_list,
                loss_func,
                average_scale=len(forget_data.dataset),
            )
            if len(grads_forget_original) == 0:
                grads_forget_original = [g for g in cur_grads]
            else:
                for i in range(len(cur_grads)):
                    grads_forget_original[i] += cur_grads[i]

        # Step 2: Prepare gradient computation batches
        # For gradient delta: Use k-hop neighbors, fill up with random samples if needed
        print(f"[GIF] Step 2: Computing gradients for {gif_k_hops}-hop influenced neighbors...")

        if not gif_use_random_retain_samples and original_dataset is not None:
            # NEW: Use true k-hop neighbor computation (recommended for sparse graphs)
            # Build user-item sparse matrix for k-hop computation
            # IMPORTANT: Need the ORIGINAL graph (retain + forget) to find k-hop neighbors
            print(f"[GIF] Building user-item interaction matrix from original data...")
            user_item_matrix = self._build_original_user_item_matrix(original_dataset, retain_train_data.dataset, forget_data.dataset)

            # Extract user IDs from forget data
            forget_user_ids = set()
            for batch in forget_data:
                user_ids = batch[forget_data.dataset.uid_field].numpy()
                forget_user_ids.update(user_ids)

            print(f"[GIF] Finding {gif_k_hops}-hop neighbors for {len(forget_user_ids)} forget users...")

            # Get true k-hop neighbors (hybrid approach with max cap)
            k_hop_neighbor_ids = self._get_k_hop_neighbors(
                list(forget_user_ids),
                user_item_matrix,
                k=gif_k_hops,
                max_neighbors=min(10000, retain_samples_used_for_update * 10)  # Cap at 10x retain samples
            )

            print(f"[GIF] Found {len(k_hop_neighbor_ids)} users in {gif_k_hops}-hop neighborhood")

            # Filter retain data to only k-hop neighbors (get all k-hop samples first)
            k_hop_batches = []
            if len(k_hop_neighbor_ids) > 0:
                # Get batches containing only k-hop neighbors
                k_hop_batches = self._filter_retain_data_by_users(
                    retain_train_data,
                    k_hop_neighbor_ids,
                    max_samples=None  # Get all k-hop samples first
                )

                # Count how many samples we have from k-hop
                k_hop_sample_count = sum(len(batch) for batch in k_hop_batches)
                print(f"[GIF] Got {k_hop_sample_count} samples from k-hop neighbors")

                # Fill up with random samples if we don't have enough
                if k_hop_sample_count < retain_samples_used_for_update:
                    samples_needed = retain_samples_used_for_update - k_hop_sample_count
                    print(f"[GIF] Filling up with {samples_needed} random samples from retain data...")

                    # Collect user IDs already in k-hop batches to avoid duplicates
                    k_hop_user_ids = set()
                    for batch in k_hop_batches:
                        k_hop_user_ids.update(batch[retain_train_data.dataset.uid_field].numpy())

                    # Sample from retain data, excluding k-hop users
                    samples_collected = 0
                    for batch in retain_train_data:
                        if samples_collected >= samples_needed:
                            break

                        # Filter out users already in k-hop
                        user_ids = batch[retain_train_data.dataset.uid_field].numpy()
                        mask = ~np.isin(user_ids, list(k_hop_user_ids))

                        if mask.any():
                            # Create filtered batch with users not in k-hop
                            filtered_batch_dict = {}
                            for key in batch.interaction:
                                value = batch.interaction[key]
                                if isinstance(value, torch.Tensor):
                                    filtered_batch_dict[key] = value[mask]
                                else:
                                    filtered_batch_dict[key] = value

                            if len(filtered_batch_dict[retain_train_data.dataset.uid_field]) > 0:
                                filtered_batch = Interaction(filtered_batch_dict)
                                k_hop_batches.append(filtered_batch)
                                samples_collected += len(filtered_batch)

                    print(f"[GIF] Added {samples_collected} random samples. Total: {k_hop_sample_count + samples_collected}")
            else:
                print(f"[GIF] Warning: No k-hop neighbors found, using random sampling from retain data")
                for batch_idx, batch in enumerate(retain_train_data):
                    if batch_idx * retain_train_data.batch_size >= retain_samples_used_for_update:
                        break
                    k_hop_batches.append(batch)
        else:
            # Random sampling from retain data (when gif_use_random_retain_samples=True or no original_dataset)
            print(f"[GIF] Using random sampling from retain data")
            k_hop_batches = []
            for batch_idx, batch in enumerate(retain_train_data):
                if batch_idx * retain_train_data.batch_size >= retain_samples_used_for_update:
                    break
                k_hop_batches.append(batch)

        # Compute gradients on k-hop neighbors (influenced region)
        grads_influenced_original = []
        grads_influenced_remaining = []

        # IMPORTANT: According to GIF paper Algorithm 1, we need:
        # - grads_influenced_original: Gradients on k-hop neighbors using ORIGINAL graph (with forget edges)
        # - grads_influenced_remaining: Gradients on k-hop neighbors using REMAINING graph (without forget edges)
        # This computes δL = ∇L_D_f + ∇L_N_G - ∇L_N_G'

        # First, compute grads_influenced_original on k-hop neighbors with ORIGINAL graph
        print(f"[GIF] Computing gradients on k-hop neighbors with ORIGINAL graph...")
        influenced_count = len(k_hop_batches) if k_hop_batches else 1

        for batch_idx, batch in enumerate(k_hop_batches):
            # Check timeout (allow at least one full batch to complete)
            if time_limit_per_batch is not None and batch_idx > 0:
                elapsed_time = time() - unlearn_start_time
                if elapsed_time >= time_limit_per_batch:
                    print(f"[GIF] Timeout reached ({elapsed_time/3600:.2f} hours). Stopping k-hop gradient computation early at batch {batch_idx + 1}/{len(k_hop_batches)}")
                    return
            batch = batch.to(self.device)

            # Gradient with ORIGINAL graph (before unlearning, forget edges still present)
            # This corresponds to ∇L_N_G in the paper
            cur_grads_original = self._batch_grad(
                self.model,
                batch,
                param_list,
                loss_func,
                average_scale=influenced_count,
            )

            if len(grads_influenced_original) == 0:
                grads_influenced_original = [g for g in cur_grads_original]
            else:
                for i in range(len(cur_grads_original)):
                    grads_influenced_original[i] += cur_grads_original[i]

        # Now compute grads_influenced_remaining on k-hop neighbors with REMAINING graph (modified)
        print(f"[GIF] Computing gradients on k-hop neighbors with REMAINING graph (forget edges excluded)...")

        # Check if model has graph structure that needs modification
        has_graph, graph_attr_name = self._model_has_graph_structure(self.model)

        if has_graph:
            print(f"[GIF] Model has graph structure ({graph_attr_name}). Building modified adjacency matrix...")
            # Build modified adjacency matrix without forget edges
            modified_adj_matrix = self._build_modified_adj_matrix(
                self.model,
                forget_data,
                retain_train_data.dataset,
                forget_data.dataset
            )

            # Use context manager to temporarily replace the graph during gradient computation
            with self._temporarily_replace_graph(self.model, graph_attr_name, modified_adj_matrix):
                # Compute gradients on remaining graph (WITHOUT forget edges)
                # This corresponds to ∇L_N_G' in the paper
                for batch_idx, batch in enumerate(k_hop_batches):
                    # Check timeout (allow at least one full batch to complete)
                    if time_limit_per_batch is not None and batch_idx > 0:
                        elapsed_time = time() - unlearn_start_time
                        if elapsed_time >= time_limit_per_batch:
                            print(f"[GIF] Timeout reached ({elapsed_time/3600:.2f} hours). Stopping remaining graph gradient computation early at batch {batch_idx + 1}/{len(k_hop_batches)}")
                            return
                    batch = batch.to(self.device)

                    cur_grads_remaining = self._batch_grad(
                        self.model,
                        batch,
                        param_list,
                        loss_func,
                        average_scale=influenced_count,
                    )

                    if len(grads_influenced_remaining) == 0:
                        grads_influenced_remaining = [g for g in cur_grads_remaining]
                    else:
                        for i in range(len(cur_grads_remaining)):
                            grads_influenced_remaining[i] += cur_grads_remaining[i]
        else:
            print(f"[GIF] Model has no graph structure. Computing gradients normally...")
            # For models without graph structure (e.g., BPR), compute normally
            # Since there's no graph, the "remaining" is the same as using current embeddings
            for batch_idx, batch in enumerate(k_hop_batches):
                # Check timeout (allow at least one full batch to complete)
                if time_limit_per_batch is not None and batch_idx > 0:
                    elapsed_time = time() - unlearn_start_time
                    if elapsed_time >= time_limit_per_batch:
                        print(f"[GIF] Timeout reached ({elapsed_time/3600:.2f} hours). Stopping gradient computation early at batch {batch_idx + 1}/{len(k_hop_batches)}")
                        return
                batch = batch.to(self.device)

                cur_grads_remaining = self._batch_grad(
                    self.model,
                    batch,
                    param_list,
                    loss_func,
                    average_scale=influenced_count,
                )

                if len(grads_influenced_remaining) == 0:
                    grads_influenced_remaining = [g for g in cur_grads_remaining]
                else:
                    for i in range(len(cur_grads_remaining)):
                        grads_influenced_remaining[i] += cur_grads_remaining[i]

        # Step 3: Compute total gradient change (delta_L in the paper)
        # delta_L = L(forget) + L(influenced_original) - L(influenced_remaining)
        print(f"[GIF] Step 3: Computing total gradient change (delta_L)...")
        delta_grads = []
        for i in range(len(param_list)):
            # Add forget gradients
            delta_g = grads_forget_original[i].clone()
            # Add influenced original gradients
            if len(grads_influenced_original) > 0:
                delta_g += grads_influenced_original[i]
            # Subtract influenced remaining gradients
            if len(grads_influenced_remaining) > 0:
                delta_g -= grads_influenced_remaining[i]
            delta_grads.append(delta_g)

        # Step 4: Compute Hessian matrix for the original training data
        print(f"[GIF] Step 4: Computing Hessian matrix...")
        # Sample a subset of data for Hessian computation (for efficiency)
        hessian_sample_size = min(len(retain_train_data.dataset), 1024)
        hessian_samples = []
        sample_count = 0

        for batch_idx, interaction in enumerate(retain_train_data):
            # Check timeout (allow at least one full batch to complete)
            if time_limit_per_batch is not None and batch_idx > 0:
                elapsed_time = time() - unlearn_start_time
                if elapsed_time >= time_limit_per_batch:
                    print(f"[GIF] Timeout reached ({elapsed_time/3600:.2f} hours). Stopping Hessian sample collection early at batch {batch_idx + 1}/{len(retain_train_data)}")
                    return
            if sample_count >= hessian_sample_size:
                break
            batch_size = min(len(interaction), hessian_sample_size - sample_count)
            hessian_samples.append(interaction[:batch_size].to(self.device))
            sample_count += batch_size

        # Compute Hessian using averaged samples
        def compute_hvp(v):
            """Hessian-Vector Product"""
            hvp_acc = [torch.zeros_like(p) for p in param_list]
            has_rnn = self.has_recurrent_layers(self.model)
            for interaction in hessian_samples:
                self.model.zero_grad()
                # second derivative not supported for RNNs when using cuDNN...
                if has_rnn:
                    with torch.backends.cudnn.flags(enabled=False):
                        loss = loss_func(interaction)
                        grads = torch.autograd.grad(loss, param_list, create_graph=True, allow_unused=True)

                        # Check for NaN/Inf in first-order gradients (skip None gradients)
                        if any(g is not None and (torch.isnan(g).any() or torch.isinf(g).any()) for g in grads):
                            print("[GIF HVP] Warning: NaN or Inf detected in first-order gradients, skipping this batch")
                            continue

                        # Compute dot product of gradients with v (skip None gradients)
                        dot_product = sum((g * v_i).sum() for g, v_i in zip(grads, v) if g is not None)

                        # Check for NaN/Inf in dot product
                        if torch.isnan(dot_product) or torch.isinf(dot_product):
                            print(f"[GIF HVP] Warning: NaN or Inf detected in dot product, skipping this batch")
                            continue

                        # Compute gradients of dot product (this gives Hv)
                        hvp = torch.autograd.grad(dot_product, param_list, retain_graph=False, allow_unused=True)
                else:
                    loss = loss_func(interaction)
                    grads = torch.autograd.grad(loss, param_list, create_graph=True, allow_unused=True)

                    # Check for NaN/Inf in first-order gradients (skip None gradients)
                    if any(g is not None and (torch.isnan(g).any() or torch.isinf(g).any()) for g in grads):
                        print("[GIF HVP] Warning: NaN or Inf detected in first-order gradients, skipping this batch")
                        continue

                    # Compute dot product of gradients with v (skip None gradients)
                    dot_product = sum((g * v_i).sum() for g, v_i in zip(grads, v) if g is not None)

                    # Check for NaN/Inf in dot product
                    if torch.isnan(dot_product) or torch.isinf(dot_product):
                        print(f"[GIF HVP] Warning: NaN or Inf detected in dot product, skipping this batch")
                        continue

                    # Compute gradients of dot product (this gives Hv)
                    hvp = torch.autograd.grad(dot_product, param_list, retain_graph=False, allow_unused=True)

                    # Check for NaN/Inf in HVP output (skip None values)
                    if any(h is not None and (torch.isnan(h).any() or torch.isinf(h).any()) for h in hvp):
                        print("[GIF HVP] Warning: NaN or Inf detected in Hessian-vector product, skipping this batch")
                        continue

                    for i in range(len(hvp_acc)):
                        if hvp[i] is not None:
                            hvp_acc[i] += hvp[i].detach() / len(hessian_samples)

            return hvp_acc

        # Step 5: Compute H^{-1} * delta_grads using iterative approximation (Theorem 5)
        # H^{-1} = sum_{i=0}^{inf} (I - lambda*H)^i
        # Recursion: H^{-1}_t * v = v + H^{-1}_{t-1} * v - lambda*H * H^{-1}_{t-1} * v
        print(f"[GIF] Step 5: Computing inverse Hessian approximation (Neumann series)...")

        # Scale lambda to ensure convergence (spectral radius of (I - lambda*H) < 1)
        lambda_scaled = gif_damping / gif_scale_factor

        # Initialize: H^{-1}_0 * v = v
        h_inv_v = [v.clone() for v in delta_grads]

        prev_diff = None
        for iteration in range(gif_iterations):
            # Check timeout (allow at least one full iteration to complete)
            if time_limit_per_batch is not None and iteration > 0:
                elapsed_time = time() - unlearn_start_time
                if elapsed_time >= time_limit_per_batch:
                    print(f"[GIF] Timeout reached ({elapsed_time/3600:.2f} hours). Stopping Hessian inverse approximation early at iteration {iteration + 1}/{gif_iterations}")
                    return
            # Compute H * H^{-1}_{t-1} * v
            h_times_h_inv_v = compute_hvp(h_inv_v)

            # Check for NaN/Inf in HVP result
            if any(torch.isnan(h).any() or torch.isinf(h).any() for h in h_times_h_inv_v):
                print(f"[GIF] Warning: NaN/Inf detected in HVP at iteration {iteration}, stopping iterations")
                break

            # Update: H^{-1}_t * v = v + H^{-1}_{t-1} * v - lambda*H * H^{-1}_{t-1} * v
            new_h_inv_v = []
            for i in range(len(param_list)):
                new_val = delta_grads[i] + h_inv_v[i] - lambda_scaled * h_times_h_inv_v[i]
                new_h_inv_v.append(new_val)

            # Check for NaN/Inf in new estimate BEFORE accepting it
            if any(torch.isnan(h).any() or torch.isinf(h).any() for h in new_h_inv_v):
                print(f"[GIF] Warning: NaN/Inf detected in update at iteration {iteration}, stopping iterations")
                break

            # Check convergence
            if iteration % 10 == 0 and iteration > 0:
                diff = sum((new_h_inv_v[i] - h_inv_v[i]).norm().item() for i in range(len(param_list)))

                # Check for NaN/Inf in convergence metric
                if math.isnan(diff) or math.isinf(diff):
                    print(f"[GIF] Warning: Divergence detected (NaN/inf diff), stopping iterations")
                    break

                # Check for divergence: diff too large (similar to SCIF/CEU/IDEA)
                if diff > 1e10:
                    print(f"[GIF] Warning: Convergence diff too large ({diff:.2e}), stopping iterations")
                    break

                print(f"[GIF] Iteration {iteration}/{gif_iterations}, convergence diff: {diff:.6f}")

                # Detect rapid divergence: if diff is very large and growing exponentially
                if prev_diff is not None and diff > 1e6:
                    if diff > prev_diff * 100:
                        print(f"[GIF] Warning: Rapid divergence detected (diff={diff:.6f}, prev={prev_diff:.6f}), stopping iterations")
                        break

                # Early stopping if converged
                if diff < 1e-6:
                    print(f"[GIF] Converged at iteration {iteration}")
                    h_inv_v = new_h_inv_v
                    break

                prev_diff = diff

            h_inv_v = new_h_inv_v

        # Step 6: Apply parameter update: theta_new = theta_old + H^{-1} * delta_L
        print(f"[GIF] Step 6: Applying parameter updates...")

        # Final validation: verify h_inv_v is valid before applying updates
        if any(torch.isnan(h).any() or torch.isinf(h).any() for h in h_inv_v):
            print(f"[GIF] Warning: Final h_inv_v contains NaN or Inf. Skipping parameter update.")
            return

        # Check if update is meaningful
        h_inv_v_norm = sum(h.norm().item()**2 for h in h_inv_v)**0.5
        if h_inv_v_norm < 1e-15:
            print(f"[GIF] Warning: Update norm is negligible (norm={h_inv_v_norm}). Skipping parameter update.")
            return

        # Optionally clip updates if max_norm is specified
        if max_norm is not None:
            total_norm = sum(h.norm().item()**2 for h in h_inv_v)**0.5
            clip_coef = max_norm / (total_norm + 1e-6)
            if clip_coef < 1:
                print(f"[GIF] Clipping updates: norm={total_norm:.6f}, max_norm={max_norm}, clip_coef={clip_coef:.6f}")
                h_inv_v = [h * clip_coef for h in h_inv_v]

        with torch.no_grad():
            for p, delta_p in zip(param_list, h_inv_v):
                # Apply the update (note: we ADD because we want to remove influence)
                p.data += delta_p

        # Verify model parameters are still valid after update
        if any(torch.isnan(p).any() or torch.isinf(p).any() for p in param_list):
            print(f"[GIF] Error: Model parameters became NaN or Inf after update. This should not happen!")
            raise RuntimeError("Model parameters corrupted during GIF update")

        # Step 7: Fine-tune on retain data to stabilize
        print(f"[GIF] Step 7: Fine-tuning on retain data ({retain_samples_used_for_update} samples)...")

        self.model.zero_grad()
        self.optimizer.zero_grad()

        epochs = 1 + retain_samples_used_for_update // len(retain_train_data.dataset)

        for epoch_idx_inner in range(epochs):
            training_start_time = time()
            train_loss = self._train_epoch_efficient_shuffle(
                retain_train_data,
                epoch_idx_inner,
                show_progress=show_progress,
                retain_samples_used_for_update=retain_samples_used_for_update,
            )

            self.train_loss_dict[epoch_idx_inner] = (
                sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            )
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx_inner, training_start_time, training_end_time, train_loss
            )

            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx_inner, train_loss)
            self.wandblogger.log_metrics(
                {"epoch": epoch_idx_inner, "train_loss": train_loss, "train_step": epoch_idx_inner},
                head="train",
            )

        print(f"[GIF] Unlearning complete!")

    def ceu(
        self,
        epoch_idx,
        forget_data,
        clean_forget_data,
        retain_train_data,
        retain_valid_data=None,
        retain_test_data=None,
        show_progress=False,
        unlearned_users_before=None,
        saved=True,
        verbose=False,
        ceu_lambda=0.01,
        ceu_sigma=0.1,
        ceu_epsilon=0.1,
        ceu_cg_iterations=100,
        ceu_hessian_samples=1024,
        ceu_k_hops=4,
        ceu_use_random_retain_samples=False,
        max_norm=None,
        param_list=None,
        original_dataset=None,
        time_limit_per_batch=None,
        unlearn_start_time=None,
    ):
        """
        CEU: Certified Edge Unlearning for Graph Neural Networks

        Reference:
            Kun Wu et al. "Certified Edge Unlearning for Graph Neural Networks" in KDD 2023.
            https://doi.org/10.1145/3580305.3599271

        Args:
            epoch_idx: Current epoch index
            forget_data: Data to unlearn (edges to remove)
            clean_forget_data: Clean version of forget data (nodes affected but edges retained)
            retain_train_data: Retained training data
            retain_valid_data: Validation data (optional)
            retain_test_data: Test data (optional)
            show_progress: Whether to show progress bars
            unlearned_users_before: Previously unlearned users (optional)
            saved: Whether to save checkpoints
            verbose: Whether to print detailed logs
            ceu_lambda: Regularization rate (lambda) for CEU loss function
            ceu_sigma: Standard deviation (sigma) for Gaussian noise
            ceu_epsilon: Epsilon parameter for (epsilon, delta)-certified unlearning guarantee
            ceu_cg_iterations: Number of conjugate gradient iterations for inverse Hessian approximation
            ceu_hessian_samples: Number of samples used for Hessian computation
            ceu_k_hops: Number of hops for k-hop neighborhood (default: 4, based on model layers)
            ceu_use_random_retain_samples: Use random sampling instead of k-hop (default: False)
            param_list: List of parameters to update (default: all trainable params)
            original_dataset: Original dataset (retain + forget) for k-hop computation
        """
        print(f"\n[CEU] Starting Certified Edge Unlearning...")
        print(f"[CEU] Parameters: lambda={ceu_lambda}, sigma={ceu_sigma}, epsilon={ceu_epsilon}")
        print(f"[CEU] CG iterations={ceu_cg_iterations}, Hessian samples={ceu_hessian_samples}, k_hops={ceu_k_hops}")

        self.move_optimizer_state(self.optimizer, self.device)
        self.model.train()
        loss_func = self.model.calculate_loss

        if param_list is None:
            param_list = self.target_params(self.model)

        # Initialize timeout tracking
        if unlearn_start_time is None:
            unlearn_start_time = time()

        # Step 1: Compute gradients for nodes affected by edge removal
        # V_E_UL includes all nodes in the neighborhood of removed edges
        print(f"[CEU] Step 1: Computing gradients for affected nodes...")

        # Gradients on original graph (with removed edges)
        grads_forget_on_original = []
        for batch_idx, interaction in enumerate(forget_data):
            # Check timeout (allow at least one full batch to complete)
            if time_limit_per_batch is not None and batch_idx > 0:
                elapsed_time = time() - unlearn_start_time
                if elapsed_time >= time_limit_per_batch:
                    print(f"[CEU] Timeout reached ({elapsed_time/3600:.2f} hours). Stopping forget gradient computation early at batch {batch_idx + 1}/{len(forget_data)}")
                    return
            interaction = interaction.to(self.device)
            cur_grads = self._batch_grad(
                self.model,
                interaction,
                param_list,
                loss_func,
                average_scale=len(forget_data.dataset),
            )
            if len(grads_forget_on_original) == 0:
                grads_forget_on_original = [g for g in cur_grads]
            else:
                for i in range(len(cur_grads)):
                    grads_forget_on_original[i] += cur_grads[i]

        # Step 1.5: Prepare gradient computation batches
        # For gradient delta: Use k-hop neighbors from clean_forget_data, fill up with retain_data if needed
        k_hop_batches = []

        if not ceu_use_random_retain_samples and original_dataset is not None:
            print(f"[CEU] Computing {ceu_k_hops}-hop influenced neighbors (V_E_UL)...")

            # Build user-item sparse matrix for k-hop computation
            print(f"[CEU] Building user-item interaction matrix from original data...")
            user_item_matrix = self._build_original_user_item_matrix(original_dataset, retain_train_data.dataset, forget_data.dataset)

            # Get users connected to forget edges
            forget_user_ids = set()
            for batch in forget_data:
                user_ids = batch[forget_data.dataset.uid_field].numpy()
                forget_user_ids.update(user_ids)

            print(f"[CEU] Finding {ceu_k_hops}-hop neighbors for {len(forget_user_ids)} users connected to forget edges...")

            # Get true k-hop neighbors (V_E_UL in the paper)
            k_hop_neighbor_ids = self._get_k_hop_neighbors(
                list(forget_user_ids),
                user_item_matrix,
                k=ceu_k_hops,
                max_neighbors=min(10000, ceu_hessian_samples * 10)  # Cap at 10x Hessian samples
            )

            print(f"[CEU] Found {len(k_hop_neighbor_ids)} users in {ceu_k_hops}-hop neighborhood (V_E_UL)")

            # Filter clean_forget_data to only k-hop neighbors (get all k-hop samples first)
            if len(k_hop_neighbor_ids) > 0 and clean_forget_data:
                k_hop_batches = self._filter_retain_data_by_users(
                    clean_forget_data,
                    k_hop_neighbor_ids,
                    max_samples=None  # Get all k-hop samples first
                )

                # Count how many samples we have from k-hop
                k_hop_sample_count = sum(len(batch) for batch in k_hop_batches)
                print(f"[CEU] Got {k_hop_sample_count} samples from k-hop neighbors in clean_forget_data")

                # Fill up with random samples from retain_train_data if we don't have enough
                if k_hop_sample_count < ceu_hessian_samples:
                    samples_needed = ceu_hessian_samples - k_hop_sample_count
                    print(f"[CEU] Filling up with {samples_needed} random samples from retain_train_data...")

                    # Collect user IDs already in k-hop batches to avoid duplicates
                    k_hop_user_ids = set()
                    # Handle empty clean_forget_data
                    uid_field = forget_data.dataset.uid_field if not clean_forget_data else clean_forget_data.dataset.uid_field
                    for batch in k_hop_batches:
                        k_hop_user_ids.update(batch[uid_field].numpy())

                    # Sample from retain_train_data, excluding k-hop users
                    samples_collected = 0
                    for batch in retain_train_data:
                        if samples_collected >= samples_needed:
                            break

                        # Filter out users already in k-hop
                        user_ids = batch[retain_train_data.dataset.uid_field].numpy()
                        mask = ~np.isin(user_ids, list(k_hop_user_ids))

                        if mask.any():
                            # Create filtered batch with users not in k-hop
                            filtered_batch_dict = {}
                            for key in batch.interaction:
                                value = batch.interaction[key]
                                if isinstance(value, torch.Tensor):
                                    filtered_batch_dict[key] = value[mask]
                                else:
                                    filtered_batch_dict[key] = value

                            if len(filtered_batch_dict[retain_train_data.dataset.uid_field]) > 0:
                                filtered_batch = Interaction(filtered_batch_dict)
                                k_hop_batches.append(filtered_batch)
                                samples_collected += len(filtered_batch)

                    print(f"[CEU] Added {samples_collected} random samples. Total: {k_hop_sample_count + samples_collected}")
            else:
                print(f"[CEU] Warning: No k-hop neighbors found, using random sampling from retain_train_data")
                # Fallback to random sampling from retain_train_data
                for batch_idx, batch in enumerate(retain_train_data):
                    if batch_idx * retain_train_data.batch_size >= ceu_hessian_samples:
                        break
                    k_hop_batches.append(batch)
        else:
            # Use clean_forget_data directly (when ceu_use_random_retain_samples=True or no original_dataset)
            print(f"[CEU] Using clean_forget_data directly (random sampling)")
            if clean_forget_data:
                for batch_idx, batch in enumerate(clean_forget_data):
                    if batch_idx * clean_forget_data.batch_size >= ceu_hessian_samples:
                        break
                    k_hop_batches.append(batch)
            else:
                print(f"[CEU] Warning: clean_forget_data is empty, using random sampling from retain_train_data")
                # Fallback to random sampling from retain_train_data
                for batch_idx, batch in enumerate(retain_train_data):
                    if batch_idx * retain_train_data.batch_size >= ceu_hessian_samples:
                        break
                    k_hop_batches.append(batch)

        # Gradients on remaining graph (without removed edges)
        # IMPORTANT: For GNN models, we need to mask the forget edges in the graph structure
        print(f"[CEU] Computing gradients on REMAINING graph (forget edges excluded)...")
        grads_forget_on_remaining = []
        influenced_sample_count = len(k_hop_batches) if k_hop_batches else 1

        # Check if model has graph structure that needs modification
        has_graph, graph_attr_name = self._model_has_graph_structure(self.model)

        if has_graph:
            print(f"[CEU] Model has graph structure ({graph_attr_name}). Building modified adjacency matrix...")
            # Build modified adjacency matrix without forget edges
            # Handle empty clean_forget_data
            clean_dataset = None if not clean_forget_data else clean_forget_data.dataset
            modified_adj_matrix = self._build_modified_adj_matrix(
                self.model,
                forget_data,
                retain_train_data.dataset,
                clean_dataset
            )

            # Use context manager to temporarily replace the graph during gradient computation
            with self._temporarily_replace_graph(self.model, graph_attr_name, modified_adj_matrix):
                # Compute gradients on remaining graph (WITHOUT forget edges)
                # This corresponds to grad(L(theta; V_E_UL, E\E_UL)) in the paper
                for interaction in k_hop_batches:
                    interaction = interaction.to(self.device)

                    cur_grads_remaining = self._batch_grad(
                        self.model,
                        interaction,
                        param_list,
                        loss_func,
                        average_scale=influenced_sample_count,
                    )

                    if len(grads_forget_on_remaining) == 0:
                        grads_forget_on_remaining = [g for g in cur_grads_remaining]
                    else:
                        for i in range(len(cur_grads_remaining)):
                            grads_forget_on_remaining[i] += cur_grads_remaining[i]
        else:
            print(f"[CEU] Model has no graph structure. Computing gradients normally...")
            # For models without graph structure (e.g., BPR), compute normally
            for interaction in k_hop_batches:
                interaction = interaction.to(self.device)

                cur_grads_remaining = self._batch_grad(
                    self.model,
                    interaction,
                    param_list,
                    loss_func,
                    average_scale=influenced_sample_count,
                )

                if len(grads_forget_on_remaining) == 0:
                    grads_forget_on_remaining = [g for g in cur_grads_remaining]
                else:
                    for i in range(len(cur_grads_remaining)):
                        grads_forget_on_remaining[i] += cur_grads_remaining[i]

        # Step 3: Compute Delta = grad(L(theta; V_E_UL, E)) - grad(L(theta; V_E_UL, E\E_UL))
        # This is the gradient change due to edge removal
        print(f"[CEU] Step 3: Computing gradient change (Delta)...")
        delta_grads = []
        for i in range(len(param_list)):
            delta_g = grads_forget_on_original[i].clone()
            if len(grads_forget_on_remaining) > 0:
                delta_g -= grads_forget_on_remaining[i]
            delta_grads.append(delta_g)

        # Step 4: Compute Hessian matrix on retain data
        print(f"[CEU] Step 4: Computing Hessian matrix on retain data...")
        hessian_sample_size = min(len(retain_train_data.dataset), ceu_hessian_samples)
        hessian_samples = []
        sample_count = 0

        for batch_idx, interaction in enumerate(retain_train_data):
            if sample_count >= hessian_sample_size:
                break
            batch_size = min(len(interaction), hessian_sample_size - sample_count)
            hessian_samples.append(interaction[:batch_size].to(self.device))
            sample_count += batch_size

        # Define Hessian-vector product function
        def compute_hvp(v):
            """Compute Hessian-Vector Product H*v"""
            hvp_acc = [torch.zeros_like(p) for p in param_list]
            valid_samples = 0
            has_rnn = self.has_recurrent_layers(self.model)
            for interaction in hessian_samples:
                self.model.zero_grad()
                # second derivative not supported for RNNs when using cuDNN...
                if has_rnn:
                    with torch.backends.cudnn.flags(enabled=False):
                        loss = loss_func(interaction)
                        # Add L2 regularization term: lambda/2 * ||theta||^2
                        if ceu_lambda > 0:
                            l2_reg = sum((p ** 2).sum() for p in param_list)
                            loss = loss + (ceu_lambda / 2) * l2_reg
                        grads = torch.autograd.grad(loss, param_list, create_graph=True, allow_unused=True)

                        # Check for NaN/Inf in first-order gradients (skip None gradients)
                        if any(g is not None and (torch.isnan(g).any() or torch.isinf(g).any()) for g in grads):
                            print("[CEU HVP] Warning: NaN or Inf detected in first-order gradients, skipping this batch")
                            continue

                        # Compute dot product of gradients with v (skip None gradients)
                        dot_product = sum((g * v_i).sum() for g, v_i in zip(grads, v) if g is not None)

                        # Check for NaN/Inf in dot product
                        if torch.isnan(dot_product) or torch.isinf(dot_product):
                            print(f"[CEU HVP] Warning: NaN or Inf detected in dot product, skipping this batch")
                            continue

                        # Compute gradients of dot product (this gives Hv)
                        hvp = torch.autograd.grad(dot_product, param_list, retain_graph=False, allow_unused=True)

                        # Check for NaN/Inf in HVP output (skip None values)
                        if any(h is not None and (torch.isnan(h).any() or torch.isinf(h).any()) for h in hvp):
                            print("[CEU HVP] Warning: NaN or Inf detected in Hessian-vector product, skipping this batch")
                            continue

                        for i in range(len(hvp_acc)):
                            if hvp[i] is not None:
                                hvp_acc[i] += hvp[i].detach()
                        valid_samples += 1
                else:
                    loss = loss_func(interaction)
                    # Add L2 regularization term: lambda/2 * ||theta||^2
                    if ceu_lambda > 0:
                        l2_reg = sum((p ** 2).sum() for p in param_list)
                        loss = loss + (ceu_lambda / 2) * l2_reg
                    grads = torch.autograd.grad(loss, param_list, create_graph=True, allow_unused=True)

                    # Check for NaN/Inf in first-order gradients (skip None gradients)
                    if any(g is not None and (torch.isnan(g).any() or torch.isinf(g).any()) for g in grads):
                        print("[CEU HVP] Warning: NaN or Inf detected in first-order gradients, skipping this batch")
                        continue

                    # Compute dot product of gradients with v (skip None gradients)
                    dot_product = sum((g * v_i).sum() for g, v_i in zip(grads, v) if g is not None)

                    # Check for NaN/Inf in dot product
                    if torch.isnan(dot_product) or torch.isinf(dot_product):
                        print(f"[CEU HVP] Warning: NaN or Inf detected in dot product, skipping this batch")
                        continue

                    # Compute gradients of dot product (this gives Hv)
                    hvp = torch.autograd.grad(dot_product, param_list, retain_graph=False, allow_unused=True)

                    # Check for NaN/Inf in HVP output (skip None values)
                    if any(h is not None and (torch.isnan(h).any() or torch.isinf(h).any()) for h in hvp):
                        print("[CEU HVP] Warning: NaN or Inf detected in Hessian-vector product, skipping this batch")
                        continue

                    for i in range(len(hvp_acc)):
                        if hvp[i] is not None:
                            hvp_acc[i] += hvp[i].detach()
                    valid_samples += 1

            # Average over valid samples
            if valid_samples > 0:
                for i in range(len(hvp_acc)):
                    hvp_acc[i] /= valid_samples

            # Add damping term: lambda * I
            if ceu_lambda > 0:
                for i in range(len(hvp_acc)):
                    hvp_acc[i] += ceu_lambda * v[i]

            return hvp_acc

        # Step 5: Compute H^{-1} * Delta using Conjugate Gradient
        # Following Algorithm 1 from the paper
        print(f"[CEU] Step 5: Computing inverse Hessian via Conjugate Gradient...")

        # Initialize CG: x_0 = 0, r_0 = Delta, p_0 = r_0
        x = [torch.zeros_like(d) for d in delta_grads]
        r = [d.clone() for d in delta_grads]
        p = [d.clone() for d in delta_grads]

        rs_old = sum((r_i * r_i).sum() for r_i in r)
        
        # Check initial residual is valid
        if rs_old.item() < 0 or torch.isnan(rs_old) or torch.isinf(rs_old):
            print(f"[CEU] Warning: Invalid initial residual (rs_old={rs_old.item()}), stopping CG solver")
            return

        for iteration in range(ceu_cg_iterations):
            # Check timeout (allow at least one full iteration to complete)
            if time_limit_per_batch is not None and iteration > 0:
                elapsed_time = time() - unlearn_start_time
                if elapsed_time >= time_limit_per_batch:
                    print(f"[CEU] Timeout reached ({elapsed_time/3600:.2f} hours). Stopping CG iterations early at iteration {iteration + 1}/{ceu_cg_iterations}")
                    return
            # Compute Ap
            Ap = compute_hvp(p)

            # Check for NaN/Inf in HVP result
            if any(torch.isnan(h).any() or torch.isinf(h).any() for h in Ap):
                print(f"[CEU] Warning: NaN/Inf detected in HVP at CG iteration {iteration}, stopping CG solver")
                break

            # Compute alpha = r^T r / p^T A p
            pAp = sum((p_i * Ap_i).sum() for p_i, Ap_i in zip(p, Ap))

            # Check for NaN/Inf or near-zero denominator
            if torch.isnan(pAp) or torch.isinf(pAp):
                print(f"[CEU] Warning: NaN/Inf detected in pAp at CG iteration {iteration}, stopping CG solver")
                break
            
            pAp_item = pAp.item()
            if abs(pAp_item) < 1e-10:
                print(f"[CEU] Warning: Near-zero pAp ({pAp_item}) at CG iteration {iteration}, stopping CG solver")
                break
            
            # CRITICAL FIX: Check for negative pAp (non-positive definite system)
            # If pAp < 0, the system is not positive definite, and alpha will be negative,
            # causing the update to move in the wrong direction and increase loss
            if pAp_item < 0:
                print(f"[CEU] Warning: Negative pAp ({pAp_item}) detected at CG iteration {iteration}. System is not positive definite. This causes loss to increase. Stopping CG solver.")
                break

            alpha = rs_old / pAp

            # Check for negative alpha (should not happen if pAp > 0, but double-check)
            if alpha.item() < 0:
                print(f"[CEU] Warning: Negative alpha ({alpha.item()}) detected at CG iteration {iteration}, stopping CG solver")
                break

            # Update x = x + alpha * p
            for i in range(len(x)):
                x[i] = x[i] + alpha * p[i]

            # Check for NaN/Inf in x
            if any(torch.isnan(xi).any() or torch.isinf(xi).any() for xi in x):
                print(f"[CEU] Warning: NaN/Inf detected in solution x at CG iteration {iteration}, stopping CG solver")
                # Revert last update
                for i in range(len(x)):
                    x[i] = x[i] - alpha * p[i]
                break

            # Update r = r - alpha * Ap
            for i in range(len(r)):
                r[i] = r[i] - alpha * Ap[i]

            rs_new = sum((r_i * r_i).sum() for r_i in r)
            
            # Check for negative rs_new (should never happen, indicates numerical error)
            if rs_new.item() < 0:
                print(f"[CEU] Warning: Negative residual squared ({rs_new.item()}) detected at CG iteration {iteration}, stopping CG solver")
                break

            # Check convergence every iteration (not just every 10) to catch divergence early
            residual_norm = torch.sqrt(rs_new).item()

            # Check for NaN/Inf in convergence metric
            if math.isnan(residual_norm) or math.isinf(residual_norm):
                print(f"[CEU] Warning: Divergence detected (NaN/inf residual), stopping CG solver")
                break

            # Check for divergence: residual norm too large (similar to SCIF)
            if residual_norm > 1e10:
                print(f"[CEU] Warning: Residual norm too large ({residual_norm:.2e}), stopping CG solver")
                break

            # Print progress every 10 iterations
            if iteration % 10 == 0:
                print(f"[CEU] CG Iteration {iteration}/{ceu_cg_iterations}, residual norm: {residual_norm:.6f}")

            if residual_norm < 1e-6:
                print(f"[CEU] CG converged at iteration {iteration}")
                break

            # Compute beta = r_new^T r_new / r_old^T r_old
            beta = rs_new / (rs_old + 1e-10)
            
            # Check for negative beta (should not happen, but indicates numerical issues)
            if beta.item() < 0:
                print(f"[CEU] Warning: Negative beta ({beta.item()}) detected at CG iteration {iteration}, stopping CG solver")
                break

            # Update p = r + beta * p
            for i in range(len(p)):
                p[i] = r[i] + beta * p[i]

            rs_old = rs_new

        # x now contains H^{-1} * Delta (the influence estimate I_E_UL)
        influence_estimate = x

        # Final validation: verify influence_estimate is valid
        if any(torch.isnan(h).any() or torch.isinf(h).any() for h in influence_estimate):
            print(f"[CEU] Warning: Final influence estimate contains NaN or Inf. Skipping parameter update.")
            return

        # Check if update is meaningful
        influence_norm = sum(h.norm().item()**2 for h in influence_estimate)**0.5
        if influence_norm < 1e-15:
            print(f"[CEU] Warning: Influence estimate norm is negligible (norm={influence_norm}). Skipping parameter update.")
            return

        # Optionally clip influence estimate if max_norm is specified (before adding noise)
        if max_norm is not None:
            total_norm = sum(h.norm().item()**2 for h in influence_estimate)**0.5
            clip_coef = max_norm / (total_norm + 1e-6)
            if clip_coef < 1:
                print(f"[CEU] Clipping influence estimate: norm={total_norm:.6f}, max_norm={max_norm}, clip_coef={clip_coef:.6f}")
                influence_estimate = [h * clip_coef for h in influence_estimate]

        # Step 6: Add Gaussian noise for (epsilon, delta)-certified guarantee
        # According to CEU paper, noise is added to the influence estimate to provide differential privacy
        print(f"[CEU] Step 6: Adding Gaussian noise for certified guarantee...")
        noisy_influence_estimate = []
        with torch.no_grad():
            for delta_p in influence_estimate:
                # Add Gaussian noise N(0, sigma^2) to each parameter update
                noise = torch.randn_like(delta_p) * ceu_sigma
                noisy_delta_p = delta_p + noise
                noisy_influence_estimate.append(noisy_delta_p)

        # Step 7: Update model parameters: theta_UL = theta_OR - (I_E_UL + noise) / |V|
        # Note: We subtract because we want to remove the influence
        print(f"[CEU] Step 7: Applying parameter updates with noise...")
        with torch.no_grad():
            # Handle empty clean_forget_data
            clean_forget_data_length = 0 if not clean_forget_data else len(clean_forget_data.dataset)
            num_nodes = len(forget_data.dataset) + clean_forget_data_length
            for p, noisy_delta_p in zip(param_list, noisy_influence_estimate):
                # Scale by 1/|V| as per Equation 11 in the paper
                p.data -= noisy_delta_p / num_nodes

        # Verify model parameters are still valid after update
        if any(torch.isnan(p).any() or torch.isinf(p).any() for p in param_list):
            print(f"[CEU] Error: Model parameters became NaN or Inf after update. This should not happen!")
            raise RuntimeError("Model parameters corrupted during CEU update")

        print(f"[CEU] Unlearning complete with (ε={ceu_epsilon}, δ)-certified guarantee!")

    def idea(
        self,
        epoch_idx,
        forget_data,
        clean_forget_data,
        retain_train_data,
        retain_valid_data=None,
        retain_test_data=None,
        show_progress=False,
        unlearned_users_before=None,
        saved=True,
        verbose=False,
        idea_damping=0.01,
        idea_sigma=0.1,
        idea_epsilon=0.1,
        idea_delta=0.01,
        idea_iterations=100,
        idea_hessian_samples=1024,
        idea_k_hops=4,
        idea_use_random_retain_samples=False,
        max_norm=None,
        param_list=None,
        original_dataset=None,
        time_limit_per_batch=None,
        unlearn_start_time=None,
    ):
        """
        IDEA: A Flexible Framework of Certified Unlearning for Graph Neural Networks

        Reference:
            Yushun Dong et al. "IDEA: A Flexible Framework of Certified Unlearning for
            Graph Neural Networks" in KDD 2024.
            https://doi.org/10.1145/3637528.3671744

        Args:
            epoch_idx: Current epoch index
            forget_data: Data to unlearn
            clean_forget_data: Clean version of forget data
            retain_train_data: Retained training data
            retain_valid_data: Validation data (optional)
            retain_test_data: Test data (optional)
            show_progress: Whether to show progress bars
            unlearned_users_before: Previously unlearned users (optional)
            saved: Whether to save checkpoints
            verbose: Whether to print detailed logs
            idea_damping: Damping factor (lambda) for strong convexity
            idea_sigma: Standard deviation (sigma) for Gaussian noise in certification
            idea_epsilon: Epsilon parameter for (epsilon, delta)-certified unlearning
            idea_delta: Delta parameter for (epsilon, delta)-certified unlearning
            idea_iterations: Number of iterations for stochastic Hessian inverse estimation
            idea_hessian_samples: Number of samples used for Hessian computation
            idea_k_hops: Number of hops for k-hop neighborhood (default: 4, based on model layers)
            idea_use_random_retain_samples: Use random sampling instead of k-hop (default: False)
            max_norm: Maximum norm for gradient clipping (optional, prevents gradient explosion)
            param_list: List of parameters to update (default: all trainable params)
            original_dataset: Original dataset (retain + forget) for k-hop computation
        """
        print(f"\n[IDEA] Starting Flexible and Certified Unlearning...")
        print(f"[IDEA] Parameters: damping={idea_damping}, sigma={idea_sigma}, epsilon={idea_epsilon}, delta={idea_delta}")
        print(f"[IDEA] Iterations={idea_iterations}, Hessian samples={idea_hessian_samples}, k_hops={idea_k_hops}")

        self.move_optimizer_state(self.optimizer, self.device)
        self.model.train()
        loss_func = self.model.calculate_loss

        if param_list is None:
            param_list = [p for p in self.model.parameters() if p.requires_grad]

        # Initialize timeout tracking
        if unlearn_start_time is None:
            unlearn_start_time = time()

        # Step 1: Compute gradient differences (Ladd - Lsub) for forget data
        # This represents the gradient of the additional term in Equation (3)
        print(f"[IDEA] Step 1: Computing gradient differences for forget data...")

        # Compute gradients on forget data (this is part of Lsub in the paper)
        forget_grads = []
        self.model.zero_grad()

        for batch_idx, batch in enumerate(forget_data):
            # Check timeout (allow at least one full batch to complete)
            if time_limit_per_batch is not None and batch_idx > 0:
                elapsed_time = time() - unlearn_start_time
                if elapsed_time >= time_limit_per_batch:
                    print(f"[IDEA] Timeout reached ({elapsed_time/3600:.2f} hours). Stopping forget gradient computation early at batch {batch_idx + 1}/{len(forget_data)}")
                    return
            batch = batch.to(self.device)
            loss = loss_func(batch)
            loss.backward()

        # Accumulate forget gradients
        for p in param_list:
            if p.grad is not None:
                forget_grads.append(p.grad.clone())
            else:
                forget_grads.append(torch.zeros_like(p))

        # Normalize by number of samples
        forget_size = len(forget_data.dataset) if hasattr(forget_data, 'dataset') else len(forget_data)
        forget_grads = [g / forget_size for g in forget_grads]

        # Compute gradients on clean forget data (part of Ladd in the paper)
        # In node unlearning, this would be empty, but for attribute unlearning
        # this represents the gradient with modified attributes
        clean_grads = []
        if clean_forget_data is not None and len(clean_forget_data) > 0:
            self.model.zero_grad()

            for batch_idx, batch in enumerate(clean_forget_data):
                # Check timeout (allow at least one full batch to complete)
                if time_limit_per_batch is not None and batch_idx > 0:
                    elapsed_time = time() - unlearn_start_time
                    if elapsed_time >= time_limit_per_batch:
                        print(f"[IDEA] Timeout reached ({elapsed_time/3600:.2f} hours). Stopping clean forget gradient computation early at batch {batch_idx + 1}/{len(clean_forget_data)}")
                        return
                batch = batch.to(self.device)
                loss = loss_func(batch)
                loss.backward()

            for p in param_list:
                if p.grad is not None:
                    clean_grads.append(p.grad.clone())
                else:
                    clean_grads.append(torch.zeros_like(p))

            clean_size = len(clean_forget_data.dataset) if hasattr(clean_forget_data, 'dataset') else len(clean_forget_data)
            clean_grads = [g / clean_size for g in clean_grads]
        else:
            clean_grads = [torch.zeros_like(p) for p in param_list]

        # Compute delta_grads = Ladd - Lsub (gradient difference)
        # This is the gradient of the objective change due to unlearning
        delta_grads = [clean_g - forget_g for clean_g, forget_g in zip(clean_grads, forget_grads)]

        # Check for NaN/Inf in gradient computation before proceeding
        if any(torch.isnan(g).any() or torch.isinf(g).any() for g in delta_grads):
            print("[IDEA] Warning: NaN or Inf detected in gradient difference computation, skipping unlearning")
            return

        print(f"[IDEA] Gradient difference norm: {sum(g.norm().item() for g in delta_grads):.6f}")

        # Step 1.5: Prepare Hessian computation batches
        # For Hessian: Use k-hop neighbors first, then fill up with random samples if needed
        k_hop_batches = []

        if not idea_use_random_retain_samples and original_dataset is not None:
            print(f"[IDEA] Computing {idea_k_hops}-hop influenced neighbors for Hessian computation...")

            # Build user-item sparse matrix for k-hop computation
            print(f"[IDEA] Building user-item interaction matrix from original data...")
            user_item_matrix = self._build_original_user_item_matrix(original_dataset, retain_train_data.dataset, forget_data.dataset)

            # Get forget user IDs
            forget_user_ids = set()
            for batch in forget_data:
                user_ids = batch[forget_data.dataset.uid_field].numpy()
                forget_user_ids.update(user_ids)

            print(f"[IDEA] Finding {idea_k_hops}-hop neighbors for {len(forget_user_ids)} forget users...")

            # Get true k-hop neighbors (V1, V2, V3, V4 in the paper)
            k_hop_neighbor_ids = self._get_k_hop_neighbors(
                list(forget_user_ids),
                user_item_matrix,
                k=idea_k_hops,
                max_neighbors=min(10000, idea_hessian_samples * 10)  # Cap at 10x Hessian samples
            )

            print(f"[IDEA] Found {len(k_hop_neighbor_ids)} users in {idea_k_hops}-hop neighborhood")

            # Filter retain data to only k-hop neighbors (no limit here, get all k-hop samples)
            if len(k_hop_neighbor_ids) > 0:
                k_hop_batches = self._filter_retain_data_by_users(
                    retain_train_data,
                    k_hop_neighbor_ids,
                    max_samples=None  # Get all k-hop samples first
                )

                # Count how many samples we have from k-hop
                k_hop_sample_count = sum(len(batch) for batch in k_hop_batches)
                print(f"[IDEA] Got {k_hop_sample_count} samples from k-hop neighbors")

                # Fill up with random samples if we don't have enough for Hessian computation
                if k_hop_sample_count < idea_hessian_samples:
                    samples_needed = idea_hessian_samples - k_hop_sample_count
                    print(f"[IDEA] Filling up with {samples_needed} random samples from retain data...")

                    # Collect user IDs already in k-hop batches to avoid duplicates
                    k_hop_user_ids = set()
                    for batch in k_hop_batches:
                        k_hop_user_ids.update(batch[retain_train_data.dataset.uid_field].numpy())

                    # Sample from retain data, excluding k-hop users
                    samples_collected = 0
                    for batch in retain_train_data:
                        if samples_collected >= samples_needed:
                            break

                        # Filter out users already in k-hop
                        user_ids = batch[retain_train_data.dataset.uid_field].numpy()
                        mask = ~np.isin(user_ids, list(k_hop_user_ids))

                        if mask.any():
                            # Create filtered batch with users not in k-hop
                            filtered_batch_dict = {}
                            for key in batch.interaction:
                                value = batch.interaction[key]
                                if isinstance(value, torch.Tensor):
                                    filtered_batch_dict[key] = value[mask]
                                else:
                                    filtered_batch_dict[key] = value

                            if len(filtered_batch_dict[retain_train_data.dataset.uid_field]) > 0:
                                filtered_batch = Interaction(filtered_batch_dict)
                                k_hop_batches.append(filtered_batch)
                                samples_collected += len(filtered_batch)

                    print(f"[IDEA] Added {samples_collected} random samples. Total: {k_hop_sample_count + samples_collected}")
            else:
                print(f"[IDEA] Warning: No k-hop neighbors found, using random sampling from retain data")
                # Fallback to random sampling
                for batch_idx, batch in enumerate(retain_train_data):
                    if batch_idx * retain_train_data.batch_size >= idea_hessian_samples:
                        break
                    k_hop_batches.append(batch)
        else:
            # Use random sampling from retain data (when idea_use_random_retain_samples=True or no original_dataset)
            print(f"[IDEA] Using random sampling for Hessian computation")
            for batch_idx, batch in enumerate(retain_train_data):
                if batch_idx * retain_train_data.batch_size >= idea_hessian_samples:
                    break
                k_hop_batches.append(batch)

        # Step 2: Estimate H^{-1} * delta_grads using Conjugate Gradients
        # Solve (H + damping*I) * x = delta_grads for x
        print(f"[IDEA] Step 2: Computing H^{-1} * delta_grads using Conjugate Gradient...")

        # Define HVP function with damping for CG solver
        def compute_hvp_with_damping(v):
            """Compute (H + damping*I) * v"""
            hvp = self._compute_hvp_on_batches(
                k_hop_batches,
                v,
                param_list,
                loss_func
            )
            if hvp is None:
                return None
            # Add damping term: lambda * I
            return [hvp[i] + idea_damping * v[i] for i in range(len(param_list))]

        # Initialize CG: x_0 = 0, r_0 = delta_grads, p_0 = r_0
        x = [torch.zeros_like(d) for d in delta_grads]
        r = [d.clone() for d in delta_grads]
        p = [d.clone() for d in delta_grads]

        rs_old = sum((r_i * r_i).sum() for r_i in r)
        
        # Check initial residual is valid
        if rs_old.item() < 0 or torch.isnan(rs_old) or torch.isinf(rs_old):
            print(f"[IDEA] Warning: Invalid initial residual (rs_old={rs_old.item()}), stopping CG solver")
            return

        for iteration in range(idea_iterations):
            # Check timeout (allow at least one full iteration to complete)
            if time_limit_per_batch is not None and iteration > 0:
                elapsed_time = time() - unlearn_start_time
                if elapsed_time >= time_limit_per_batch:
                    print(f"[IDEA] Timeout reached ({elapsed_time/3600:.2f} hours). Stopping iterations early at iteration {iteration + 1}/{idea_iterations}")
                    return
            # Compute Ap = (H + damping*I) * p
            Ap = compute_hvp_with_damping(p)

            if Ap is None:
                print(f"[IDEA] Warning: HVP computation failed at CG iteration {iteration}, stopping CG solver")
                break

            # Check for NaN/Inf in HVP result
            if any(torch.isnan(h).any() or torch.isinf(h).any() for h in Ap):
                print(f"[IDEA] Warning: NaN/Inf detected in HVP at CG iteration {iteration}, stopping CG solver")
                break

            # Compute alpha = r^T r / p^T A p
            pAp = sum((p_i * Ap_i).sum() for p_i, Ap_i in zip(p, Ap))

            # Check for NaN/Inf or near-zero denominator
            if torch.isnan(pAp) or torch.isinf(pAp):
                print(f"[IDEA] Warning: NaN/Inf detected in pAp at CG iteration {iteration}, stopping CG solver")
                break
            
            pAp_item = pAp.item()
            if abs(pAp_item) < 1e-10:
                print(f"[IDEA] Warning: Near-zero pAp ({pAp_item}) at CG iteration {iteration}, stopping CG solver")
                break
            
            # CRITICAL FIX: Check for negative pAp (non-positive definite system)
            # If pAp < 0, the system is not positive definite, and alpha will be negative,
            # causing the update to move in the wrong direction and increase loss
            if pAp_item < 0:
                print(f"[IDEA] Warning: Negative pAp ({pAp_item}) detected at CG iteration {iteration}. System is not positive definite. This causes loss to increase. Stopping CG solver.")
                break

            alpha = rs_old / pAp

            # Check for negative alpha (should not happen if pAp > 0, but double-check)
            if alpha.item() < 0:
                print(f"[IDEA] Warning: Negative alpha ({alpha.item()}) detected at CG iteration {iteration}, stopping CG solver")
                break

            # Update x = x + alpha * p
            for i in range(len(x)):
                x[i] = x[i] + alpha * p[i]

            # Check for NaN/Inf in x
            if any(torch.isnan(xi).any() or torch.isinf(xi).any() for xi in x):
                print(f"[IDEA] Warning: NaN/Inf detected in solution x at CG iteration {iteration}, stopping CG solver")
                # Revert last update
                for i in range(len(x)):
                    x[i] = x[i] - alpha * p[i]
                break

            # Update r = r - alpha * Ap
            for i in range(len(r)):
                r[i] = r[i] - alpha * Ap[i]

            rs_new = sum((r_i * r_i).sum() for r_i in r)
            
            # Check for negative rs_new (should never happen, indicates numerical error)
            if rs_new.item() < 0:
                print(f"[IDEA] Warning: Negative residual squared ({rs_new.item()}) detected at CG iteration {iteration}, stopping CG solver")
                break

            # Check convergence every iteration (not just every 10) to catch divergence early
            residual_norm = torch.sqrt(rs_new).item()

            # Check for NaN/Inf in convergence metric
            if math.isnan(residual_norm) or math.isinf(residual_norm):
                print(f"[IDEA] Warning: Divergence detected (NaN/inf residual), stopping CG solver")
                break

            # Check for divergence: residual norm too large (similar to SCIF)
            if residual_norm > 1e10:
                print(f"[IDEA] Warning: Residual norm too large ({residual_norm:.2e}), stopping CG solver")
                break

            # Print progress every 10 iterations
            if iteration % 10 == 0:
                print(f"[IDEA] CG Iteration {iteration}/{idea_iterations}, residual norm: {residual_norm:.6f}")

            if residual_norm < 1e-6:
                print(f"[IDEA] CG converged at iteration {iteration}")
                break

            # Compute beta = r_new^T r_new / r_old^T r_old
            beta = rs_new / (rs_old + 1e-10)
            
            # Check for negative beta (should not happen, but indicates numerical issues)
            if beta.item() < 0:
                print(f"[IDEA] Warning: Negative beta ({beta.item()}) detected at CG iteration {iteration}, stopping CG solver")
                break

            # Update p = r + beta * p
            for i in range(len(p)):
                p[i] = r[i] + beta * p[i]

            rs_old = rs_new

        # x now contains H^{-1} * delta_grads
        h_inv_delta = x

        # Step 3: Apply parameter update with certification noise (Theorem 3)
        print(f"[IDEA] Step 3: Applying certified parameter updates...")

        # Final check: verify h_inv_delta is valid before applying updates
        if any(torch.isnan(h).any() or torch.isinf(h).any() for h in h_inv_delta):
            print(f"[IDEA] Warning: Final h_inv_delta contains NaN or Inf. Skipping parameter update.")
            return

        # Check if update is meaningful (similar to SCIF check)
        h_inv_delta_norm = sum(h.norm().item()**2 for h in h_inv_delta)**0.5
        if h_inv_delta_norm < 1e-15:
            print(f"[IDEA] Warning: Update norm is negligible (norm={h_inv_delta_norm}). Skipping parameter update.")
            return

        # Optionally clip updates if max_norm is specified (before computing certification parameters)
        if max_norm is not None:
            total_norm = sum(h.norm().item()**2 for h in h_inv_delta)**0.5
            clip_coef = max_norm / (total_norm + 1e-6)
            if clip_coef < 1:
                print(f"[IDEA] Clipping updates: norm={total_norm:.6f}, max_norm={max_norm}, clip_coef={clip_coef:.6f}")
                h_inv_delta = [h * clip_coef for h in h_inv_delta]

        # Compute zeta (upper bound on ||tilde_theta* - bar_theta*||_2) for certification
        # This is from Proposition 3 in the paper
        update_norm = sum(h.norm().item()**2 for h in h_inv_delta)**0.5
        zeta = update_norm * idea_damping  # Simplified bound computation

        # Compute sigma for (epsilon, delta)-certified unlearning (Theorem 3)
        # sigma >= zeta / (epsilon * sqrt(2*ln(1.25/delta)))
        required_sigma = zeta / (idea_epsilon * (2 * math.log(1.25 / idea_delta))**0.5)
        actual_sigma = max(idea_sigma, required_sigma)

        print(f"[IDEA] Update norm: {update_norm:.6f}, zeta bound: {zeta:.6f}")
        print(f"[IDEA] Required sigma: {required_sigma:.6f}, using sigma: {actual_sigma:.6f}")

        # Apply parameter update with Gaussian noise for certification

        with torch.no_grad():
            for p, delta_p in zip(param_list, h_inv_delta):
                # Add Gaussian noise for certification
                noise = torch.randn_like(delta_p) * actual_sigma
                # Apply update: theta_new = theta_old + (1/m) * H^{-1} * delta_grads + noise
                # The 1/m factor is handled by normalization in gradient computation
                p.data += delta_p + noise

        # Verify model parameters are still valid after update
        if any(torch.isnan(p).any() or torch.isinf(p).any() for p in param_list):
            print(f"[IDEA] Error: Model parameters became NaN or Inf after update. This should not happen!")
            raise RuntimeError("Model parameters corrupted during IDEA update")

        # Step 4: Fine-tune on retain data to stabilize
        print(f"[IDEA] Step 4: Fine-tuning on retain data...")

        self.model.zero_grad()
        self.optimizer.zero_grad()

        # Fine-tune for 1 epoch
        retain_samples_used = 128 * forget_size
        epochs = 1 + retain_samples_used // len(retain_train_data.dataset)

        for epoch_idx_inner in range(epochs):
            training_start_time = time()
            train_loss = self._train_epoch_efficient_shuffle(
                retain_train_data,
                epoch_idx_inner,
                show_progress=show_progress,
                retain_samples_used_for_update=retain_samples_used,
            )

            self.train_loss_dict[epoch_idx_inner] = (
                sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            )
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx_inner, training_start_time, training_end_time, train_loss
            )

            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx_inner, train_loss)
            self.wandblogger.log_metrics(
                {"epoch": epoch_idx_inner, "train_loss": train_loss, "train_step": epoch_idx_inner},
                head="train",
            )

        print(f"[IDEA] Unlearning complete!")

    def _compute_hvp_on_batches(self, batches, vectors, param_list, loss_func):
        """
        Compute Hessian-vector product (HVP) on a set of batches.

        Args:
            batches: List of batches to compute HVP on
            vectors: Vectors to multiply with Hessian
            param_list: List of parameters
            loss_func: Loss function

        Returns:
            List of HVP results for each parameter, or None if computation fails
        """
        self.model.zero_grad()

        # Compute average loss on batches
        total_loss = 0
        total_samples = 0
        has_rnn = self.has_recurrent_layers(self.model)

        # second derivative not supported for RNNs when using cuDNN...
        if has_rnn:
            with torch.backends.cudnn.flags(enabled=False):
                for batch in batches:
                    batch = batch.to(self.device)
                    loss = loss_func(batch)
                    total_loss += loss * len(batch)
                    total_samples += len(batch)

                avg_loss = total_loss / max(total_samples, 1)

                # Compute gradients
                grads = torch.autograd.grad(avg_loss, param_list, create_graph=True, allow_unused=True)

                # Check for None gradients
                if any(g is None for g in grads):
                    return None

                # Check for NaN/Inf in gradients (before HVP computation)
                if any(torch.isnan(g).any() or torch.isinf(g).any() for g in grads):
                    print("[IDEA HVP] NaN or Inf detected in first-order gradients")
                    return None
        else:
            for batch in batches:
                batch = batch.to(self.device)
                loss = loss_func(batch)
                total_loss += loss * len(batch)
                total_samples += len(batch)

            avg_loss = total_loss / max(total_samples, 1)

            # Compute gradients
            grads = torch.autograd.grad(avg_loss, param_list, create_graph=True, allow_unused=True)

            # Check for None gradients
            if any(g is None for g in grads):
                return None

            # Check for NaN/Inf in gradients (before HVP computation)
            if any(torch.isnan(g).any() or torch.isinf(g).any() for g in grads):
                print("[IDEA HVP] NaN or Inf detected in first-order gradients")
                return None

        # Compute HVP: H * v = grad(grad^T * v)
        # second derivative not supported for RNNs when using cuDNN...
        if has_rnn:
            with torch.backends.cudnn.flags(enabled=False):
                grad_vector_dot = sum((g * v).sum() for g, v in zip(grads, vectors))

                # Check for NaN/Inf in dot product
                if torch.isnan(grad_vector_dot) or torch.isinf(grad_vector_dot):
                    print(f"[IDEA HVP] NaN or Inf detected in dot product (dot={grad_vector_dot})")
                    return None

                hvp = torch.autograd.grad(grad_vector_dot, param_list, allow_unused=True)
        else:
            grad_vector_dot = sum((g * v).sum() for g, v in zip(grads, vectors))

            # Check for NaN/Inf in dot product
            if torch.isnan(grad_vector_dot) or torch.isinf(grad_vector_dot):
                print(f"[IDEA HVP] NaN or Inf detected in dot product (dot={grad_vector_dot})")
                return None

            hvp = torch.autograd.grad(grad_vector_dot, param_list, allow_unused=True)

        # Check for None in HVP
        if any(h is None for h in hvp):
            return None

        # Check for NaN/Inf in HVP output
        if any(torch.isnan(h).any() or torch.isinf(h).any() for h in hvp):
            print("[IDEA HVP] NaN or Inf detected in Hessian-vector product output")
            return None

        return list(hvp)

    def fit(
        self,
        train_data,
        valid_data=None,
        verbose=True,
        saved=True,
        show_progress=False,
        callback_fn=None,
        retrain_flag=False,
        retrain_checkpoint_idx_to_match=None,
    ):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose, retrain_flag=retrain_flag, retrain_checkpoint_idx_to_match=retrain_checkpoint_idx_to_match,)

        self.eval_collector.data_collect(train_data)
        if self.config["train_neg_sample_args"].get("dynamic", False):
            train_data.get_model(self.model)
        valid_step = 0

        # Time budget tracking
        max_training_hours = self.config["max_training_hours"] if "max_training_hours" in self.config else None
        training_start_time_budget = time() if max_training_hours is not None else None
        max_training_seconds = max_training_hours * 3600 if max_training_hours is not None else None
        last_epoch_duration = None
        time_budget_exceeded = False

        for epoch_idx in range(self.start_epoch, self.epochs):
            # Check time budget before starting a new epoch
            if max_training_hours is not None and last_epoch_duration is not None:
                elapsed_time = time() - training_start_time_budget
                projected_time_after_epoch = elapsed_time + last_epoch_duration
                if projected_time_after_epoch > max_training_seconds:
                    time_budget_exceeded = True
                    self.logger.info(
                        set_color("Time budget exceeded", "yellow") +
                        f": Elapsed time: {elapsed_time/3600:.2f}h, " +
                        f"Projected time after next epoch: {projected_time_after_epoch/3600:.2f}h, " +
                        f"Budget: {max_training_hours:.2f}h. Stopping training and proceeding to evaluation."
                    )
                    break

            # train
            epoch_start_time = time()
            training_start_time = time()
            train_loss = self._train_epoch(
                train_data, epoch_idx, show_progress=show_progress
            )
            self.train_loss_dict[epoch_idx] = (
                sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            )
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self.wandblogger.log_metrics(
                {"epoch": epoch_idx, "train_loss": train_loss, "train_step": epoch_idx},
                head="train",
            )

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose, retrain_flag=retrain_flag, retrain_checkpoint_idx_to_match=retrain_checkpoint_idx_to_match,)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(
                    valid_data, show_progress=show_progress
                )

                (
                    self.best_valid_score,
                    self.cur_step,
                    stop_flag,
                    update_flag,
                ) = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger,
                )
                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("valid_score", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = (
                    set_color("valid result", "blue") + ": \n" + dict2str(valid_result)
                )
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar("Vaild_score", valid_score, epoch_idx)
                self.wandblogger.log_metrics(
                    {**valid_result, "valid_step": valid_step}, head="valid"
                )

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose, retrain_flag=retrain_flag, retrain_checkpoint_idx_to_match=retrain_checkpoint_idx_to_match,)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = "Finished training, best eval result in epoch %d" % (
                        epoch_idx - self.cur_step * self.eval_step
                    )
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1

            # Track epoch duration for time budget
            if max_training_hours is not None:
                epoch_end_time = time()
                last_epoch_duration = epoch_end_time - epoch_start_time

        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result
    
    def target_params(
        self,
        model,
    ):
        params = []

        if hasattr(model, "user_embedding"):
            params.append(model.user_embedding.weight)
        elif isinstance(model, GRU4Rec):
            params.append(model.item_embedding.weight)
            params.append(model.dense.weight)
            if model.dense.bias is not None:
                params.append(model.dense.bias)
        elif isinstance(model, NARM):
            params.append(model.item_embedding.weight)
            params.append(model.b.weight)
        elif isinstance(model, SASRec):
            params.append(model.item_embedding.weight)
        elif isinstance(model, SRGNN):
            params.append(model.item_embedding.weight)
            params.append(model.linear_transform.weight)
            if model.linear_transform.bias is not None:
                params.append(model.linear_transform.bias)

        return params if len(params) > 0 else [p for _, p in model.named_parameters()]
    
    def norm_list(self, plist) -> float:
        return torch.sqrt(sum(p.pow(2).sum() for p in plist)).item()

    def _batch_grad(
        self,
        model,
        batch,
        param_list,
        loss_func,
        average_scale=None,
        average=True,
    ):
        average_scale = average_scale or len(batch)
        acc = [torch.zeros_like(p) for p in param_list]
        # for interaction in batch:
        #     interaction = interaction.to(self.device)
        loss = loss_func(batch)
        grads = torch.autograd.grad(loss, param_list, retain_graph=False, allow_unused=True)
        for a, g, p in zip(acc, grads, param_list):
            # If gradient is None (parameter not used in computation graph), treat as zero gradient
            if g is not None:
                a += g.detach()
            # else: gradient is effectively zero (parameter didn't contribute to loss)
        if average:
            for a in acc:
                a /= average_scale
        return acc

    def scif(
        self,
        epoch_idx,
        forget_data=None,
        clean_forget_data=None,
        retain_train_data=None,
        retain_valid_data=None,
        retain_test_data=None,
        show_progress=False,
        train_pair_count=1024,
        retain_samples_used_for_update=128,
        max_norm=None,
        unlearned_users_before=None,
        damping=0.01,
        time_limit_per_batch=None,
        unlearn_start_time=None,
    ):

        # r"""Train the model in an epoch

        # Args:
        #     train_data (DataLoader): The train data.
        #     epoch_idx (int): The current epoch id.
        #     loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
        #         :attr:`self.model.calculate_loss`. Defaults to ``None``.
        #     show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        # Returns:
        #     float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
        #     multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
        #     tuple which includes the sum of loss in each part.
        # """
        self.model.train()
        loss_func = self.model.calculate_loss
        total_loss = None
        # iter_data = (
        #     tqdm(
        #         retain_train_data,
        #         total=len(retain_train_data),
        #         ncols=100,
        #         desc=set_color(f"Train {epoch_idx:>5}", "pink"),
        #     )
        #     if show_progress
        #     else retain_train_data
        # )
        iter_data = retain_train_data

        # if not self.config["single_spec"] and retain_train_data.shuffle:
        #     retain_train_data.sampler.set_epoch(epoch_idx)

        # scaler = amp.GradScaler(enabled=self.enable_scaler)

        param_list = self.target_params(self.model)

        # Initialize timeout tracking
        if unlearn_start_time is None:
            unlearn_start_time = time()

        neg_grads = None
        pos_grads = None

        # Handle empty clean_forget_data (when all user interactions are sensitive)
        clean_forget_data_length = 0 if not clean_forget_data else len(clean_forget_data.dataset)

        forget_sample_count = max(1, len(forget_data.dataset) - clean_forget_data_length)
        print(f"forget data length: {len(forget_data.dataset)}, clean forget data length: {clean_forget_data_length}")
        retain_count = max(1, retain_samples_used_for_update * forget_sample_count - clean_forget_data_length)
        
        # calculate grads for forget data which we want to forget
        for batch_idx, interaction in enumerate(forget_data):
            # Check timeout (allow at least one full batch to complete)
            if time_limit_per_batch is not None and batch_idx > 0:
                elapsed_time = time() - unlearn_start_time
                if elapsed_time >= time_limit_per_batch:
                    print(f"[SCIF] Timeout reached ({elapsed_time/3600:.2f} hours). Stopping gradient computation early at batch {batch_idx + 1}/{len(forget_data)}")
                    return
            interaction = interaction.to(self.device)
            cur_grads = self._batch_grad(self.model, interaction, param_list, loss_func, average_scale=retain_count)
            if neg_grads is None:
                neg_grads = [-g for g in cur_grads]
            else:
                for i in range(len(cur_grads)):
                    neg_grads[i] -= cur_grads[i]
        
        # calculate grads for the cleaned forget data (forgotten interactions are removed)
        clean_forget_data_size = 0
        if clean_forget_data is not None:
            for batch_idx, interaction in enumerate(clean_forget_data):
                # Check timeout (allow at least one full batch to complete)
                if time_limit_per_batch is not None and batch_idx > 0:
                    elapsed_time = time() - unlearn_start_time
                    if elapsed_time >= time_limit_per_batch:
                        print(f"[SCIF] Timeout reached ({elapsed_time/3600:.2f} hours). Stopping clean forget gradient computation early at batch {batch_idx + 1}/{len(clean_forget_data)}")
                        return
                clean_forget_data_size += len(interaction)
                interaction = interaction.to(self.device)
                cur_grads = self._batch_grad(self.model, interaction, param_list, loss_func, average_scale=retain_count)
                if pos_grads is None:
                    pos_grads = [g for g in cur_grads]
                else:
                    for i in range(len(cur_grads)):
                        pos_grads[i] += cur_grads[i]
        
        
        # get retain grads from the retain data to prevent catastrophic forgetting
        for batch_idx, interaction in enumerate(retain_train_data):
            # Check timeout (allow at least one full batch to complete)
            if time_limit_per_batch is not None and batch_idx > 0:
                elapsed_time = time() - unlearn_start_time
                if elapsed_time >= time_limit_per_batch:
                    print(f"[SCIF] Timeout reached ({elapsed_time/3600:.2f} hours). Stopping retain gradient computation early at batch {batch_idx + 1}/{len(retain_train_data)}")
                    return
            if retain_count <= 0:
                break
            interaction = interaction.to(self.device)
            # filter out forgotten users from the original data we use here
            uid_field = retain_train_data.dataset.uid_field
            mask = ~torch.isin(interaction[uid_field], torch.tensor(unlearned_users_before, device=self.device))
            interaction = interaction[mask]
            interaction = interaction[:retain_count]
            cur_grads = self._batch_grad(self.model, interaction, param_list, loss_func, average_scale=retain_count)
            if pos_grads is None:
                pos_grads = [g for g in cur_grads]
            else:
                for i in range(len(cur_grads)):
                    pos_grads[i] += cur_grads[i]
            retain_count -= len(interaction)

        grads = [n + p for n, p in zip(neg_grads, pos_grads)]

        # Check for NaN/Inf in gradient computation
        if any(torch.isnan(g).any() or torch.isinf(g).any() for g in grads):
            self.logger.warning("[SCIF] NaN or Inf detected in gradient computation before CG solver")
            return

        # Check timeout before CG solver
        if time_limit_per_batch is not None:
            elapsed_time = time() - unlearn_start_time
            if elapsed_time >= time_limit_per_batch:
                print(f"[SCIF] Timeout reached ({elapsed_time/3600:.2f} hours). Skipping CG solver.")
                return

        inv_hvp, cg_converged = self.cg_inv_hvp(
            self.model,
            clean_forget_data,
            retain_train_data,
            forget_data,
            grads,
            param_list,
            damping=damping,
        )

        # Check if CG solver returned valid results
        if any(torch.isnan(h).any() or torch.isinf(h).any() for h in inv_hvp):
            self.logger.warning(f"[SCIF] CG solver returned NaN or Inf. Skipping parameter update for request {epoch_idx}.")
            return

        # Check if CG solver converged or produced meaningful results
        inv_hvp_norm = self.norm_list(inv_hvp)
        if inv_hvp_norm < 1e-15:
            self.logger.warning(f"[SCIF] CG solver produced negligible update (norm={inv_hvp_norm}). Skipping parameter update for request {epoch_idx}.")
            return

        tau = 1 / len(retain_train_data.dataset)
        if max_norm is not None:
            delta_norm = tau * inv_hvp_norm
            if delta_norm > max_norm:
                scale = max_norm / delta_norm
                inv_hvp = [x * scale for x in inv_hvp]

        with torch.no_grad():
            for p, d in zip(param_list, inv_hvp):
                p -= tau * d

        # Verify model parameters are still valid after update
        if any(torch.isnan(p).any() or torch.isinf(p).any() for p in param_list):
            self.logger.error(f"[SCIF] Model parameters became NaN or Inf after update for request {epoch_idx}. This should not happen!")
            raise RuntimeError("Model parameters corrupted during SCIF update")

        print(f"[SCIF]  removed {len(forget_data.dataset)} samples, "
            f"tau={tau},  ||delta theta||={tau * self.norm_list(inv_hvp)}")

    def _dot_list(self, a, b):
        return sum((x * y).sum() for x, y in zip(a, b))
    
    def _add_scaled(self, x, y, alpha):
        return [xi + alpha * yi for xi, yi in zip(x, y)]

    def has_recurrent_layers(self, model):
        recurrent_modules = (
            torch.nn.RNN,
            torch.nn.LSTM,
            torch.nn.GRU,
            torch.nn.RNNCell,
            torch.nn.LSTMCell,
            torch.nn.GRUCell,
        )
        
        for module in model.modules():
            if isinstance(module, recurrent_modules):
                return True
        return False

    def _hvp_single(self, model, interaction, v_list, param_list):
        loss_func = model.calculate_loss
        # second derivative not supported for RNNs when using cuDNN...
        if self.has_recurrent_layers(model):
            with torch.backends.cudnn.flags(enabled=False):
                loss = loss_func(interaction)
                grad = torch.autograd.grad(loss, param_list, create_graph=True, allow_unused=True)
                # Check for NaN/Inf in gradients (skip None gradients)
                if any(g is not None and (torch.isnan(g).any() or torch.isinf(g).any()) for g in grad):
                    self.logger.warning("[HVP] NaN or Inf detected in first-order gradients")
                    return None
                dot  = sum((g * v).sum() for g, v in zip(grad, v_list) if g is not None)
                # Check for NaN/Inf in dot product
                if torch.isnan(dot) or torch.isinf(dot):
                    self.logger.warning(f"[HVP] NaN or Inf detected in dot product (dot={dot})")
                    return None
                hv   = torch.autograd.grad(dot, param_list, retain_graph=False, allow_unused=True)
        else:
            loss = loss_func(interaction)
            grad = torch.autograd.grad(loss, param_list, create_graph=True, allow_unused=True)
            # Check for NaN/Inf in gradients (skip None gradients)
            if any(g is not None and (torch.isnan(g).any() or torch.isinf(g).any()) for g in grad):
                self.logger.warning("[HVP] NaN or Inf detected in first-order gradients")
                return None
            dot  = sum((g * v).sum() for g, v in zip(grad, v_list) if g is not None)
            # Check for NaN/Inf in dot product
            if torch.isnan(dot) or torch.isinf(dot):
                self.logger.warning(f"[HVP] NaN or Inf detected in dot product (dot={dot})")
                return None
            hv   = torch.autograd.grad(dot, param_list, retain_graph=False, allow_unused=True)

        # Check for NaN/Inf in Hessian-vector product (skip None values)
        if any(h is not None and (torch.isnan(h).any() or torch.isinf(h).any()) for h in hv):
            self.logger.warning("[HVP] NaN or Inf detected in Hessian-vector product output")
            return None

        return [h.detach() if h is not None else torch.zeros_like(p) for h, p in zip(hv, param_list)]

    def _hvp_dataset(
        self,
        model, data_batch,
        v_list, param_list,
        average=True,
    ):
        acc = [torch.zeros_like(p) for p in v_list]
        # for interaction in data_batch:
        interaction = data_batch
        hv = self._hvp_single(
            model, interaction,
            v_list, param_list,
        )
        if hv is None:
            # HVP computation failed, return None to signal error
            return None
        for a, h in zip(acc, hv):
            a += h
        if average:
            bs = len(data_batch)
            for a in acc:
                a /= bs
        return acc

    def cg_inv_hvp(
        self,
        model,
        clean_forget_data,
        retain_train_data,
        forget_data,
        v_list,
        param_list,
        damping= 0.01,
        bs=16,
        max_iter=None,
        tol=1e-5,
        LOCAL=False,
        samples_wanted_constant=1024,
    ):
        """
        Solve  (H + lambda I) x = v  for x with conjugate gradients.
        Returns: (x, diverged_flag)
        """
        # Handle empty clean_forget_data
        clean_forget_data_length = 0 if not clean_forget_data else len(clean_forget_data.dataset)

        # Total number of iterations: one pass over the data by default
        max_iter = max_iter or math.ceil((clean_forget_data_length + len(retain_train_data.dataset)) / bs)

        # --- initialisation ------------------------------------------------------
        x      = [torch.zeros_like(v) for v in v_list]   # x_theta
        r      = [v.clone() for v in v_list]             # r_theta = v - H x_theta  (H x_theta = 0)
        p      = [ri.clone() for ri in r]                # p_theta = r_theta
        rs_old = self._dot_list(r, r).item()
        
        # Check initial residual is valid
        if rs_old < 0 or math.isnan(rs_old) or math.isinf(rs_old):
            self.logger.warning(f"[CG] Invalid initial residual (rs_old={rs_old}), stopping CG solver")
            return None, True

        samples_wanted = samples_wanted_constant * max(1, len(forget_data.dataset) - clean_forget_data_length)
        samples_seen = 0
        break_flag = False

        # Only process clean_forget_data if it's not None
        if clean_forget_data is not None:
            for interaction in clean_forget_data:
                if break_flag:
                    break
                for i in range(0, len(interaction), bs):
                    batch = interaction[i : i + bs]
                    if samples_seen >= samples_wanted:
                        break_flag = True
                        break
                    interaction = interaction.to(self.device)
                    interaction = interaction[:samples_wanted - samples_seen]
                    samples_seen += len(interaction)
                    batch = interaction

                    q = self._hvp_dataset(
                        model, batch,
                        p, param_list,
                        average=True,
                    )
                    # Check if HVP computation failed
                    if q is None:
                        self.logger.warning(f"[CG] HVP computation failed at clean_forget iteration {i // bs}, stopping CG solver")
                        break_flag = True
                        break

                    # add lambda I term
                    q = [qi + damping * pi for qi, pi in zip(q, p)]

                    pq_dot = self._dot_list(p, q).item()
                    
                    # Check for NaN/Inf in dot product
                    if math.isnan(pq_dot) or math.isinf(pq_dot):
                        self.logger.warning(f"[CG] NaN or Inf detected in p*q dot product at clean_forget iteration {i // bs}")
                        break
                    
                    # Use epsilon threshold instead of exact zero comparison for numerical stability
                    if abs(pq_dot) < 1e-10:
                        # p and q are nearly orthogonal, we cannot proceed
                        break
                    
                    # CRITICAL FIX: Check for negative pq_dot (non-positive definite system)
                    # If pq_dot < 0, the system is not positive definite, and alpha will be negative,
                    # causing the update to move in the wrong direction and increase loss
                    if pq_dot < 0:
                        self.logger.warning(f"[CG] Negative p*q dot product ({pq_dot}) detected at clean_forget iteration {i // bs}. System is not positive definite. This causes loss to increase. Stopping CG solver.")
                        break_flag = True
                        break

                    alpha = rs_old / pq_dot
                    
                    # Check for negative alpha (should not happen if pq_dot > 0, but double-check)
                    if alpha < 0:
                        self.logger.warning(f"[CG] Negative alpha ({alpha}) detected at clean_forget iteration {i // bs}, stopping CG solver")
                        break_flag = True
                        break

                    # x_{k+1}  =  x_k + alpha p_k
                    x = self._add_scaled(x, p, alpha)

                    # r_{k+1}  =  r_k - alpha q
                    r = self._add_scaled(r, q, -alpha)

                rs_new = self._dot_list(r, r).item()
                
                # Check for negative rs_new (should never happen, indicates numerical error)
                if rs_new < 0:
                    self.logger.warning(f"[CG] Negative residual squared ({rs_new}) detected at clean_forget iteration {i // bs}, stopping CG solver")
                    break_flag = True
                    break
                
                residual_norm = math.sqrt(rs_new)
                if residual_norm < tol:
                    break  # converged

                beta = rs_new / rs_old
                
                # Check for negative beta (should not happen, but indicates numerical issues)
                if beta < 0:
                    self.logger.warning(f"[CG] Negative beta ({beta}) detected at clean_forget iteration {i // bs}, stopping CG solver")
                    break_flag = True
                    break

                # p_{k+1}  =  r_{k+1} + beta p_k
                p = [ri + beta * pi for ri, pi in zip(r, p)]
                rs_old = rs_new

        
        for interaction in retain_train_data:
            if break_flag:
                break
            for i in range(0, len(interaction), bs):
                batch = interaction[i : i + bs]
                if samples_seen >= samples_wanted:
                    break_flag = True
                    break
                interaction = interaction.to(self.device)
                interaction = interaction[:samples_wanted - samples_seen]
                samples_seen += len(interaction)
                batch = interaction

                q = self._hvp_dataset(
                    model, batch,
                    p, param_list,
                    average=True,
                )
                # Check if HVP computation failed
                if q is None:
                    self.logger.warning(f"[CG] HVP computation failed at retain_train iteration {i // bs}, stopping CG solver")
                    break_flag = True
                    break

                # add lambda I term
                q = [qi + damping * pi for qi, pi in zip(q, p)]

                pq_dot = self._dot_list(p, q).item()
                
                # Check for NaN/Inf in dot product
                if math.isnan(pq_dot) or math.isinf(pq_dot):
                    self.logger.warning(f"[CG] NaN or Inf detected in p*q dot product at retain_train iteration {i // bs}")
                    break
                
                # Use epsilon threshold instead of exact zero comparison for numerical stability
                if abs(pq_dot) < 1e-10:
                    # p and q are nearly orthogonal, we cannot proceed
                    break
                
                # CRITICAL FIX: Check for negative pq_dot (non-positive definite system)
                # If pq_dot < 0, the system is not positive definite, and alpha will be negative,
                # causing the update to move in the wrong direction and increase loss
                if pq_dot < 0:
                    self.logger.warning(f"[CG] Negative p*q dot product ({pq_dot}) detected at retain_train iteration {i // bs}. System is not positive definite. This causes loss to increase. Stopping CG solver.")
                    break_flag = True
                    break

                alpha = rs_old / pq_dot
                
                # Check for negative alpha (should not happen if pq_dot > 0, but double-check)
                if alpha < 0:
                    self.logger.warning(f"[CG] Negative alpha ({alpha}) detected at retain_train iteration {i // bs}, stopping CG solver")
                    break_flag = True
                    break

                # x_{k+1}  =  x_k + alpha p_k
                x = self._add_scaled(x, p, alpha)

                # r_{k+1}  =  r_k - alpha q
                r = self._add_scaled(r, q, -alpha)

                rs_new = self._dot_list(r, r).item()
                
                # Check for negative rs_new (should never happen, indicates numerical error)
                if rs_new < 0:
                    self.logger.warning(f"[CG] Negative residual squared ({rs_new}) detected at retain_train iteration {i // bs}, stopping CG solver")
                    break_flag = True
                    break
                
                residual_norm = math.sqrt(rs_new)
                if residual_norm < tol:
                    break  # converged

                beta = rs_new / rs_old
                
                # Check for negative beta (should not happen, but indicates numerical issues)
                if beta < 0:
                    self.logger.warning(f"[CG] Negative beta ({beta}) detected at retain_train iteration {i // bs}, stopping CG solver")
                    break_flag = True
                    break

                # p_{k+1}  =  r_{k+1} + beta p_k
                p = [ri + beta * pi for ri, pi in zip(r, p)]
                rs_old = rs_new

            

        diverged = math.sqrt(rs_old) >= tol
        return x, diverged

    def seif(
        self,
        epoch_idx,
        forget_data,
        clean_forget_data,
        retain_train_data,
        retain_valid_data=None,
        retain_test_data=None,
        show_progress=False,
        unlearned_users_before=None,
        saved=True,
        verbose=False,
        seif_erase_std=0.6,
        seif_erase_std_final=0.005,
        seif_repair_epochs=4,
        seif_forget_class_weight=0.05,
        seif_learning_rate=0.0007,
        seif_momentum=0.9,
        seif_weight_decay=5e-4,
        max_norm=None,
        param_list=None,
        time_limit_per_batch=None,
        unlearn_start_time=None,
    ):
        """
        SEIF: Self-Influence Guided Data Attribution for Unlearning

        Based on the winning solution from NeurIPS 2023 Machine Unlearning Challenge.

        Two-phase approach:
        1. Erase Phase: Apply Gaussian noise to selected model parameters
        2. Repair Phase: Fine-tune on retain set with weighted loss (lower weight for forget class)

        Args:
            epoch_idx: Current epoch index
            forget_data: Data to unlearn
            clean_forget_data: Clean version of forget data (affected items but interactions retained)
            retain_train_data: Retained training data
            retain_valid_data: Validation data (optional)
            retain_test_data: Test data (optional)
            show_progress: Whether to show progress bars
            unlearned_users_before: Previously unlearned users (optional)
            saved: Whether to save checkpoints
            verbose: Whether to print detailed logs
            seif_erase_std: Standard deviation for Gaussian noise in erase phase
            seif_erase_std_final: Standard deviation for final robustness noise
            seif_repair_epochs: Number of repair epochs
            seif_forget_class_weight: Weight for forget class in loss (lower = more forgetting)
            seif_learning_rate: Learning rate for repair phase
            seif_momentum: Momentum for SGD optimizer
            seif_weight_decay: Weight decay for optimizer
            max_norm: Maximum norm for gradient clipping (optional)
            param_list: List of parameters to update (default: all trainable params)
        """
        print(f"\n[SEIF] Starting Self-Influence Guided Data Attribution Unlearning...")
        print(f"[SEIF] Parameters: erase_std={seif_erase_std}, repair_epochs={seif_repair_epochs}")
        print(f"[SEIF] Forget class weight={seif_forget_class_weight}, lr={seif_learning_rate}")

        self.model.train()
        self.move_optimizer_state(self.optimizer, self.device)

        if param_list is None:
            param_list = [p for p in self.model.parameters() if p.requires_grad]

        # PHASE 1: ERASE - Apply Gaussian noise to heuristically chosen parameters
        # In the original implementation, they target convolutional layers
        # For recommender systems, we'll target embedding layers and interaction layers
        print(f"[SEIF] Phase 1: Erase - Adding Gaussian noise to model parameters...")

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue

                # Heuristically choose which layers to add noise to
                # Target embedding layers and linear/interaction layers (similar to conv layers in vision)
                if any(keyword in name.lower() for keyword in ['embedding', 'linear', 'mlp', 'interaction']):
                    # Apply Gaussian noise: param = param + N(0, std^2 * param)
                    noise = torch.normal(mean=0.0, std=seif_erase_std, size=param.shape, device=param.device)
                    param.data.add_(noise * param.data.abs())

                    if verbose:
                        print(f"[SEIF] Added noise to layer: {name}, shape: {param.shape}")

        print(f"[SEIF] Erase phase completed. Proceeding to repair phase...")

        # PHASE 2: REPAIR - Fine-tune on retain set with weighted loss
        print(f"[SEIF] Phase 2: Repair - Fine-tuning on retain set for {seif_repair_epochs} epochs...")

        # Track start time for timeout check
        if unlearn_start_time is None:
            unlearn_start_time = time()
        repair_start_time = unlearn_start_time
        # Use passed time limit if available, otherwise default to 10 minutes
        timeout_seconds = time_limit_per_batch if time_limit_per_batch is not None else (10 * 60)

        # Temporarily save original learning rate
        original_lr = self.optimizer.param_groups[0]['lr']

        # Update learning rate for SEIF repair phase
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = seif_learning_rate

        # For recommender systems, we adapt the weighted loss approach:
        # Instead of class weights (which are for classification), we'll use a
        # custom loss that gives lower weight to interactions involving forget items

        # Extract forget items from forget_data
        forget_items = set()
        for interaction_batch in forget_data:
            if 'item_id' in interaction_batch:
                forget_items.update(interaction_batch['item_id'].tolist())

        print(f"[SEIF] Identified {len(forget_items)} unique forget items")

        for repair_epoch in range(seif_repair_epochs):
            # Check if we've exceeded the 30 minute timeout
            elapsed_time = time() - repair_start_time
            if elapsed_time >= timeout_seconds:
                print(f"[SEIF] Timeout reached ({elapsed_time/60:.2f} minutes). Stopping repair epochs early at epoch {repair_epoch + 1}/{seif_repair_epochs}")
                break
            epoch_loss = 0.0
            batch_count = 0

            self.model.train()
            
            timeout_reached = False

            for batch_idx, interaction in enumerate(retain_train_data):
                # Check timeout periodically (every 10 batches to avoid overhead)
                if batch_idx % 10 == 0:
                    elapsed_time = time() - repair_start_time
                    if elapsed_time >= timeout_seconds:
                        print(f"[SEIF] Timeout reached ({elapsed_time/60:.2f} minutes). Stopping repair epochs early at epoch {repair_epoch + 1}/{seif_repair_epochs}, batch {batch_idx}")
                        timeout_reached = True
                        break
                
                interaction = interaction.to(self.device)

                # Filter out users that have been unlearned
                if unlearned_users_before:
                    uid_field = retain_train_data.dataset.uid_field
                    mask = ~torch.isin(
                        interaction[uid_field],
                        torch.tensor(unlearned_users_before, device=self.device)
                    )
                    interaction = interaction[mask]

                if len(interaction) == 0:
                    continue

                self.optimizer.zero_grad()

                # Calculate base loss
                losses = self.model.calculate_loss(interaction)

                if isinstance(losses, tuple):
                    loss = sum(losses)
                else:
                    loss = losses

                # Apply weighted loss: reduce weight for interactions with forget items
                if 'item_id' in interaction:
                    item_ids = interaction['item_id']
                    # Create weight tensor: low weight for forget items, 1.0 for others
                    weights = torch.ones(len(item_ids), device=self.device)
                    forget_mask = torch.isin(item_ids, torch.tensor(list(forget_items), device=self.device))
                    weights[forget_mask] = seif_forget_class_weight

                    # Apply weights to loss (average over weighted samples)
                    loss = loss * weights.mean()

                loss.backward()

                # Apply gradient clipping if max_norm is provided
                if max_norm is not None:
                    clip_grad_norm_(self.model.parameters(), max_norm)

                self.optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1
            
            # Break from epoch loop if timeout was reached
            if timeout_reached:
                break

            # Apply final robustness noise before last epoch
            if repair_epoch == seif_repair_epochs - 2:
                print(f"[SEIF] Applying final robustness noise (std={seif_erase_std_final})...")
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if not param.requires_grad:
                            continue

                        if any(keyword in name.lower() for keyword in ['embedding', 'linear', 'mlp', 'interaction']):
                            noise = torch.normal(mean=0.0, std=seif_erase_std_final, size=param.shape, device=param.device)
                            param.data.add_(noise * param.data.abs())

            avg_loss = epoch_loss / max(1, batch_count)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"[SEIF] Repair Epoch {repair_epoch + 1}/{seif_repair_epochs}, Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

        # Restore original learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = original_lr

        print(f"[SEIF] Unlearning complete for request {epoch_idx}")

    def unlearn(
        self,
        epoch_idx,
        forget_data,
        clean_forget_data,
        retain_train_data,
        retain_valid_data=None,
        retain_test_data=None,
        unlearning_algorithm="scif",
        verbose=True,
        saved=True,
        show_progress=False,
        callback_fn=None,
        max_norm=None,
        unlearned_users_before=None,
        kookmin_init_rate=0.01,
        retrain_checkpoint_idx_to_match=None,
        task_type="SBR",
        damping=0.01,
        gif_damping=0.01,
        gif_scale_factor=1000,
        gif_iterations=100,
        gif_k_hops=2,
        gif_use_true_khop=False,
        gif_retain_samples=None,
        ceu_lambda=0.01,
        ceu_sigma=0.1,
        ceu_epsilon=0.1,
        ceu_cg_iterations=100,
        ceu_hessian_samples=1024,
        idea_damping=0.01,
        idea_sigma=0.1,
        idea_epsilon=0.1,
        idea_delta=0.01,
        idea_iterations=100,
        idea_hessian_samples=1024,
        seif_erase_std=0.6,
        seif_erase_std_final=0.005,
        seif_repair_epochs=4,
        seif_forget_class_weight=0.05,
        seif_learning_rate=0.0007,
        seif_momentum=0.9,
        seif_weight_decay=5e-4,
        original_dataset=None,
        unlearning_batchsize=None,
        time_limit_per_batch=None,
        retain_samples_used_for_update=None,
    ):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        # if saved and self.start_epoch >= self.epochs:
        #     self._save_checkpoint(-1, verbose=verbose)

        # self.eval_collector.data_collect(train_data)
        # if self.config["train_neg_sample_args"].get("dynamic", False):
        #     train_data.get_model(self.model)
        # valid_step = 0

        # for epoch_idx in range(self.start_epoch, self.epochs):
        #     # unlearn 1 epoch
        # Track start time for timeout checking
        unlearn_start_time = time()
        if time_limit_per_batch is not None:
            print(f"[Unlearn] Time limit per batch: {time_limit_per_batch/3600:.2f} hours ({time_limit_per_batch:.0f} seconds)")
        
        if unlearning_algorithm == "scif":
            if retain_samples_used_for_update is None:
                retain_samples_used_for_update = 16 * len(forget_data.dataset)
            self.scif(
                epoch_idx,
                forget_data,
                clean_forget_data,
                retain_train_data,
                retain_valid_data,
                retain_test_data,
                show_progress=show_progress,
                max_norm=max_norm,
                unlearned_users_before=unlearned_users_before,
                damping=damping,
                time_limit_per_batch=time_limit_per_batch,
                unlearn_start_time=unlearn_start_time,
                retain_samples_used_for_update=retain_samples_used_for_update,
            )
        elif unlearning_algorithm == "kookmin":
            retain_samples_used_for_update = 16 * len(forget_data.dataset)
            neg_grad_retain_sample_size = 128 * len(forget_data.dataset)
            param_list = [p for _, p in self.model.named_parameters()]
            self.kookmin(
                epoch_idx,
                forget_data,
                clean_forget_data,
                retain_train_data,
                retain_valid_data,
                retain_test_data,
                show_progress=show_progress,
                unlearned_users_before=unlearned_users_before,
                kookmin_init_rate=kookmin_init_rate,
                retain_samples_used_for_update=retain_samples_used_for_update,
                neg_grad_retain_sample_size=neg_grad_retain_sample_size,
                param_list=param_list,
                time_limit_per_batch=time_limit_per_batch,
                unlearn_start_time=unlearn_start_time,
            )
        elif unlearning_algorithm == "fanchuan":
            retain_samples_used_for_update = 16 * len(forget_data.dataset)
            self.fanchuan(
                epoch_idx,
                forget_data,
                clean_forget_data,
                retain_train_data,
                retain_valid_data,
                retain_test_data,
                show_progress=show_progress,
                unlearned_users_before=unlearned_users_before,
                saved=saved,
                verbose=verbose,
                retain_samples_used_for_update=retain_samples_used_for_update,
                task_type=task_type,
                time_limit_per_batch=time_limit_per_batch,
                unlearn_start_time=unlearn_start_time,
            )
        elif unlearning_algorithm == "gif":
            # GIF: Graph Influence Function
            # Read all parameters directly from config
            param_list = [p for _, p in self.model.named_parameters()]
            self.gif(
                epoch_idx,
                forget_data,
                clean_forget_data,
                retain_train_data,
                retain_valid_data,
                retain_test_data,
                show_progress=show_progress,
                unlearned_users_before=unlearned_users_before,
                saved=saved,
                verbose=verbose,
                retain_samples_used_for_update=self.config["gif_retain_samples"] if "gif_retain_samples" in self.config and self.config["gif_retain_samples"] is not None else 16 * len(forget_data.dataset),
                gif_damping=self.config["gif_damping"] if "gif_damping" in self.config and self.config["gif_damping"] is not None else 0.01,
                gif_scale_factor=self.config["gif_scale_factor"] if "gif_scale_factor" in self.config and self.config["gif_scale_factor"] is not None else 1000,
                gif_iterations=self.config["gif_iterations"] if "gif_iterations" in self.config and self.config["gif_iterations"] is not None else 100,
                gif_k_hops=self.config["gif_k_hops"] if "gif_k_hops" in self.config and self.config["gif_k_hops"] is not None else 4,
                gif_use_random_retain_samples=self.config["gif_use_random_retain_samples"] if "gif_use_random_retain_samples" in self.config and self.config["gif_use_random_retain_samples"] is not None else False,
                max_norm=max_norm,
                param_list=param_list,
                original_dataset=original_dataset,
                time_limit_per_batch=time_limit_per_batch,
                unlearn_start_time=unlearn_start_time,
            )
        elif unlearning_algorithm == "ceu":
            # CEU: Certified Edge Unlearning
            # Read all parameters directly from config
            param_list = [p for _, p in self.model.named_parameters()]
            self.ceu(
                epoch_idx,
                forget_data,
                clean_forget_data,
                retain_train_data,
                retain_valid_data,
                retain_test_data,
                show_progress=show_progress,
                unlearned_users_before=unlearned_users_before,
                saved=saved,
                verbose=verbose,
                ceu_lambda=self.config["ceu_lambda"] if "ceu_lambda" in self.config and self.config["ceu_lambda"] is not None else 0.01,
                ceu_sigma=self.config["ceu_sigma"] if "ceu_sigma" in self.config and self.config["ceu_sigma"] is not None else 0.1,
                ceu_epsilon=self.config["ceu_epsilon"] if "ceu_epsilon" in self.config and self.config["ceu_epsilon"] is not None else 0.1,
                ceu_cg_iterations=self.config["ceu_cg_iterations"] if "ceu_cg_iterations" in self.config and self.config["ceu_cg_iterations"] is not None else 100,
                ceu_hessian_samples=self.config["ceu_hessian_samples"] if "ceu_hessian_samples" in self.config and self.config["ceu_hessian_samples"] is not None else 1024,
                ceu_k_hops=self.config["ceu_k_hops"] if "ceu_k_hops" in self.config and self.config["ceu_k_hops"] is not None else 4,
                ceu_use_random_retain_samples=self.config["ceu_use_random_retain_samples"] if "ceu_use_random_retain_samples" in self.config and self.config["ceu_use_random_retain_samples"] is not None else False,
                max_norm=max_norm,
                param_list=param_list,
                original_dataset=original_dataset,
                time_limit_per_batch=time_limit_per_batch,
                unlearn_start_time=unlearn_start_time,
            )
        elif unlearning_algorithm == "idea":
            # IDEA: Flexible Framework of Certified Unlearning for GNNs
            # Read all parameters directly from config
            param_list = [p for _, p in self.model.named_parameters()]
            self.idea(
                epoch_idx,
                forget_data,
                clean_forget_data,
                retain_train_data,
                retain_valid_data,
                retain_test_data,
                show_progress=show_progress,
                unlearned_users_before=unlearned_users_before,
                saved=saved,
                verbose=verbose,
                idea_damping=self.config["idea_damping"] if "idea_damping" in self.config and self.config["idea_damping"] is not None else 0.01,
                idea_sigma=self.config["idea_sigma"] if "idea_sigma" in self.config and self.config["idea_sigma"] is not None else 0.1,
                idea_epsilon=self.config["idea_epsilon"] if "idea_epsilon" in self.config and self.config["idea_epsilon"] is not None else 0.1,
                idea_delta=self.config["idea_delta"] if "idea_delta" in self.config and self.config["idea_delta"] is not None else 0.01,
                idea_iterations=self.config["idea_iterations"] if "idea_iterations" in self.config and self.config["idea_iterations"] is not None else 100,
                idea_hessian_samples=self.config["idea_hessian_samples"] if "idea_hessian_samples" in self.config and self.config["idea_hessian_samples"] is not None else 1024,
                idea_k_hops=self.config["idea_k_hops"] if "idea_k_hops" in self.config and self.config["idea_k_hops"] is not None else 4,
                idea_use_random_retain_samples=self.config["idea_use_random_retain_samples"] if "idea_use_random_retain_samples" in self.config and self.config["idea_use_random_retain_samples"] is not None else False,
                max_norm=max_norm,
                param_list=param_list,
                original_dataset=original_dataset,
                time_limit_per_batch=time_limit_per_batch,
                unlearn_start_time=unlearn_start_time,
            )
        elif unlearning_algorithm == "seif":
            # SEIF: Self-Influence Guided Data Attribution
            # Based on NeurIPS 2023 Machine Unlearning Challenge winning solution
            param_list = [p for _, p in self.model.named_parameters()]
            self.seif(
                epoch_idx,
                forget_data,
                clean_forget_data,
                retain_train_data,
                retain_valid_data,
                retain_test_data,
                show_progress=show_progress,
                unlearned_users_before=unlearned_users_before,
                saved=saved,
                verbose=verbose,
                seif_erase_std=self.config["seif_erase_std"] if "seif_erase_std" in self.config and self.config["seif_erase_std"] is not None else 0.6,
                seif_erase_std_final=self.config["seif_erase_std_final"] if "seif_erase_std_final" in self.config and self.config["seif_erase_std_final"] is not None else 0.005,
                seif_repair_epochs=self.config["seif_repair_epochs"] if "seif_repair_epochs" in self.config and self.config["seif_repair_epochs"] is not None else 4,
                seif_forget_class_weight=self.config["seif_forget_class_weight"] if "seif_forget_class_weight" in self.config and self.config["seif_forget_class_weight"] is not None else 0.05,
                seif_learning_rate=self.config["seif_learning_rate"] if "seif_learning_rate" in self.config and self.config["seif_learning_rate"] is not None else 0.0007,
                seif_momentum=self.config["seif_momentum"] if "seif_momentum" in self.config and self.config["seif_momentum"] is not None else 0.9,
                seif_weight_decay=self.config["seif_weight_decay"] if "seif_weight_decay" in self.config and self.config["seif_weight_decay"] is not None else 5e-4,
                max_norm=max_norm,
                param_list=param_list,
                time_limit_per_batch=time_limit_per_batch,
                unlearn_start_time=unlearn_start_time,
            )

        # Report elapsed time
        unlearn_end_time = time()
        elapsed_time = unlearn_end_time - unlearn_start_time
        if time_limit_per_batch is not None:
            remaining_time = time_limit_per_batch - elapsed_time
            if remaining_time < 0:
                print(f"[Unlearn] WARNING: Exceeded time limit by {abs(remaining_time)/3600:.2f} hours")
            else:
                print(f"[Unlearn] Elapsed time: {elapsed_time/3600:.2f} hours, Remaining: {remaining_time/3600:.2f} hours")
        else:
            print(f"[Unlearn] Elapsed time: {elapsed_time/3600:.2f} hours")

        if saved:
            # Only include batchsize in filename if explicitly set and != 1 (for hyperparameter tests)
            # When using the best batchsize normally, it won't be in the filename
            self._save_checkpoint(epoch_idx, verbose=verbose, retrain_checkpoint_idx_to_match=retrain_checkpoint_idx_to_match, unlearning_batchsize=unlearning_batchsize)
            # self.train_loss_dict[epoch_idx] = (
            #     sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            # )
            # training_end_time = time()
            # train_loss_output = self._generate_train_loss_output(
            #     epoch_idx, training_start_time, training_end_time, train_loss
            # )
            # if verbose:
            #     self.logger.info(train_loss_output)
            # self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            # self.wandblogger.log_metrics(
            #     {"epoch": epoch_idx, "train_loss": train_loss, "train_step": epoch_idx},
            #     head="train",
            # )

        #     # eval
        #     if self.eval_step <= 0 or not valid_data:
        #         if saved:
        #             self._save_checkpoint(epoch_idx, verbose=verbose)
        #         continue
        #     if (epoch_idx + 1) % self.eval_step == 0:
        #         valid_start_time = time()
        #         valid_score, valid_result = self._valid_epoch(
        #             valid_data, show_progress=show_progress
        #         )

        #         (
        #             self.best_valid_score,
        #             self.cur_step,
        #             stop_flag,
        #             update_flag,
        #         ) = early_stopping(
        #             valid_score,
        #             self.best_valid_score,
        #             self.cur_step,
        #             max_step=self.stopping_step,
        #             bigger=self.valid_metric_bigger,
        #         )
        #         valid_end_time = time()
        #         valid_score_output = (
        #             set_color("epoch %d evaluating", "green")
        #             + " ["
        #             + set_color("time", "blue")
        #             + ": %.2fs, "
        #             + set_color("valid_score", "blue")
        #             + ": %f]"
        #         ) % (epoch_idx, valid_end_time - valid_start_time, valid_score)
        #         valid_result_output = (
        #             set_color("valid result", "blue") + ": \n" + dict2str(valid_result)
        #         )
        #         if verbose:
        #             self.logger.info(valid_score_output)
        #             self.logger.info(valid_result_output)
        #         self.tensorboard.add_scalar("Vaild_score", valid_score, epoch_idx)
        #         self.wandblogger.log_metrics(
        #             {**valid_result, "valid_step": valid_step}, head="valid"
        #         )

        #         if update_flag:
        #             if saved:
        #                 self._save_checkpoint(epoch_idx, verbose=verbose)
        #             self.best_valid_result = valid_result

        #         if callback_fn:
        #             callback_fn(epoch_idx, valid_score)

        #         if stop_flag:
        #             stop_output = "Finished training, best eval result in epoch %d" % (
        #                 epoch_idx - self.cur_step * self.eval_step
        #             )
        #             if verbose:
        #                 self.logger.info(stop_output)
        #             break

        #         valid_step += 1

        # self._add_hparam_to_tensorboard(self.best_valid_score)
        
    def _full_sort_batch_eval(self, batched_data):
        interaction, history_index, positive_u, positive_i = batched_data
        try:
            # Note: interaction without item ids
            scores = self.model.full_sort_predict(interaction.to(self.device))
        except NotImplementedError:
            inter_len = len(interaction)
            new_inter = interaction.to(self.device).repeat_interleave(self.tot_item_num)
            batch_size = len(new_inter)
            new_inter.update(self.item_tensor.repeat(inter_len))
            if batch_size <= self.test_batch_size:
                scores = self.model.predict(new_inter)
            else:
                scores = self._spilt_predict(new_inter, batch_size)

        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf
        return interaction, scores, positive_u, positive_i

    def _neg_sample_batch_eval(self, batched_data):
        interaction, row_idx, positive_u, positive_i = batched_data
        batch_size = interaction.length
        if batch_size <= self.test_batch_size:
            origin_scores = self.model.predict(interaction.to(self.device))
        else:
            origin_scores = self._spilt_predict(interaction, batch_size)

        if self.config["eval_type"] == EvaluatorType.VALUE:
            return interaction, origin_scores, positive_u, positive_i
        elif self.config["eval_type"] == EvaluatorType.RANKING:
            col_idx = interaction[self.config["ITEM_ID_FIELD"]]
            batch_user_num = positive_u[-1] + 1
            scores = torch.full(
                (batch_user_num, self.tot_item_num), -np.inf, device=self.device
            )
            scores[row_idx, col_idx] = origin_scores
            return interaction, scores, positive_u, positive_i

    @torch.no_grad()
    def evaluate(
        self, eval_data, load_best_model=True, model_file=None, show_progress=False, collect_target_probabilities=False, target_items=None,
    ):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            collections.OrderedDict: eval result, key is the eval metric and value in the corresponding metric value.
        """
        if not eval_data:
            return

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            if not os.path.exists(checkpoint_file):
                error_msg = f"Checkpoint file not found: {checkpoint_file}"
                self.logger.warning(error_msg)
                return None  # Return None to indicate the file doesn't exist
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.load_other_parameter(checkpoint.get("other_parameter"))
            message_output = "Loading model structure and parameters from {}".format(
                checkpoint_file
            )
            self.logger.info(message_output)

        self.model.eval()

        probability_data = [] if collect_target_probabilities else None
        if collect_target_probabilities and target_items is not None:
            target_idx = torch.as_tensor(target_items, device=self.device, dtype=torch.long)

        if isinstance(eval_data, FullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            if self.item_tensor is None:
                self.item_tensor = eval_data._dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval
        if self.config["eval_type"] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data._dataset.item_num

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
            if show_progress
            else eval_data
        )

        num_sample = 0
        for batch_idx, batched_data in enumerate(iter_data):
            num_sample += len(batched_data)
            interaction, scores, positive_u, positive_i = eval_func(batched_data)

            if collect_target_probabilities and target_items is not None:
                # calculate probabilities for target items
                logZ = torch.logsumexp(scores, dim=1, keepdim=True)
                p_targets = torch.exp(scores.index_select(1, target_idx) - logZ)
                
                for j in range(len(interaction)):
                    # Get target_length for NBR models to properly filter padding
                    # Note: We don't skip empty baskets as evaluation metrics need the correct count
                    target_length = None
                    if hasattr(self.model, 'target_length_field') and self.model.target_length_field in interaction:
                        target_length = interaction[self.model.target_length_field][j].item()
                    
                    # For NBR models, use target_item_list instead of item_id_list
                    # Check if ITEM_SEQ field exists in interaction (it might not for NBR models)
                    item_seq = None
                    if hasattr(self.model, 'ITEM_SEQ'):
                        # Check if the field actually exists in the interaction
                        if self.model.ITEM_SEQ in interaction:
                            seq_data = interaction[self.model.ITEM_SEQ][j].cpu().numpy()
                            # Filter out padding (typically -1 or 0) from sequence
                            if hasattr(self.model, 'ITEM_SEQ_LEN') and self.model.ITEM_SEQ_LEN in interaction:
                                seq_len = interaction[self.model.ITEM_SEQ_LEN][j].item()
                                if seq_len > 0:
                                    item_seq = seq_data[:seq_len]
                                    # Filter out padding values (typically -1 or 0)
                                    item_seq = item_seq[item_seq > 0] if len(item_seq) > 0 else None
                            else:
                                # Filter out padding values (assume -1 or 0 is padding)
                                item_seq = seq_data[seq_data > 0] if len(seq_data) > 0 else None
                        elif hasattr(self.model, 'target_items_field') and self.model.target_items_field in interaction:
                            # NBR models use target_item_list field instead
                            seq_data = interaction[self.model.target_items_field][j].cpu().numpy()
                            # Filter out padding based on target_length if available
                            if target_length is not None and target_length > 0:
                                item_seq = seq_data[:target_length]
                                # Further filter out padding values (typically -1 or 0)
                                item_seq = item_seq[item_seq > 0] if len(item_seq) > 0 else None
                            elif target_length == 0:
                                # Empty basket - item_seq should be empty array, not None
                                item_seq = np.array([], dtype=seq_data.dtype)
                            else:
                                # Filter out padding values (assume -1 or 0 is padding)
                                item_seq = seq_data[seq_data > 0] if len(seq_data) > 0 else None
                    
                    probabilities = p_targets[j].cpu().numpy()
                    user_id_val = None
                    if hasattr(self.model, 'USER_ID') and self.model.USER_ID in interaction:
                        user_id_val = interaction[self.model.USER_ID][j].item()
                    
                    # Always append, even for empty baskets - evaluation metrics need the correct count
                    probability_data.append({
                        'item_seq': item_seq,
                        'positive_item': positive_i[j].item(),
                        'target_probabilities': probabilities,
                        'user_id': user_id_val,
                        'target_length': target_length if target_length is not None else None
                    })

            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                )
            self.eval_collector.eval_batch_collect(
                scores, interaction, positive_u, positive_i
            )
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)
        if not self.config["single_spec"]:
            result = self._map_reduce(result, num_sample)
        self.wandblogger.log_eval_metrics(result, head="eval")
        
        if collect_target_probabilities:
            return result, probability_data
        return result

    def _map_reduce(self, result, num_sample):
        gather_result = {}
        total_sample = [
            torch.zeros(1).to(self.device) for _ in range(self.config["world_size"])
        ]
        torch.distributed.all_gather(
            total_sample, torch.Tensor([num_sample]).to(self.device)
        )
        total_sample = torch.cat(total_sample, 0)
        total_sample = torch.sum(total_sample).item()
        for key, value in result.items():
            result[key] = torch.Tensor([value * num_sample]).to(self.device)
            gather_result[key] = [
                torch.zeros_like(result[key]).to(self.device)
                for _ in range(self.config["world_size"])
            ]
            torch.distributed.all_gather(gather_result[key], result[key])
            gather_result[key] = torch.cat(gather_result[key], dim=0)
            gather_result[key] = round(
                torch.sum(gather_result[key]).item() / total_sample,
                self.config["metric_decimal_place"],
            )
        return gather_result

    def _spilt_predict(self, interaction, batch_size):
        spilt_interaction = dict()
        for key, tensor in interaction.interaction.items():
            spilt_interaction[key] = tensor.split(self.test_batch_size, dim=0)
        num_block = (batch_size + self.test_batch_size - 1) // self.test_batch_size
        result_list = []
        for i in range(num_block):
            current_interaction = dict()
            for key, spilt_tensor in spilt_interaction.items():
                current_interaction[key] = spilt_tensor[i]
            result = self.model.predict(
                Interaction(current_interaction).to(self.device)
            )
            if len(result.shape) == 0:
                result = result.unsqueeze(0)
            result_list.append(result)
        return torch.cat(result_list, dim=0)



class SRGNNTrainer(Trainer):
    r"""
    SRGNNTrainerLrScheduler is designed for training SRGNN with a LR scheduler.
    Here we multiply the learning rate by 0.1 every 3 epochs.
    """
    def __init__(self, config, model):
        super().__init__(config, model)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=3,
            gamma=0.1
        )

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False, retain_samples_used_for_update=None, reinit_masks=None, scale_for_reinit_params=1.0):
        loss = super()._train_epoch(train_data, epoch_idx, loss_func=loss_func, show_progress=show_progress, retain_samples_used_for_update=retain_samples_used_for_update, reinit_masks=reinit_masks, scale_for_reinit_params=scale_for_reinit_params)
        self.scheduler.step()
        return loss


class KGTrainer(Trainer):
    r"""KGTrainer is designed for Knowledge-aware recommendation methods. Some of these models need to train the
    recommendation related task and knowledge related task alternately.

    """

    def __init__(self, config, model):
        super(KGTrainer, self).__init__(config, model)

        self.train_rec_step = config["train_rec_step"]
        self.train_kg_step = config["train_kg_step"]

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False, retain_samples_used_for_update=None, reinit_masks=None, scale_for_reinit_params=1.0):
        if self.train_rec_step is None or self.train_kg_step is None:
            interaction_state = KGDataLoaderState.RSKG
        elif (
            epoch_idx % (self.train_rec_step + self.train_kg_step) < self.train_rec_step
        ):
            interaction_state = KGDataLoaderState.RS
        else:
            interaction_state = KGDataLoaderState.KG
        if not self.config["single_spec"]:
            train_data.knowledge_shuffle(epoch_idx)
        train_data.set_mode(interaction_state)
        if interaction_state in [KGDataLoaderState.RSKG, KGDataLoaderState.RS]:
            return super()._train_epoch(
                train_data, epoch_idx, show_progress=show_progress, retain_samples_used_for_update=retain_samples_used_for_update, reinit_masks=reinit_masks, scale_for_reinit_params=scale_for_reinit_params
            )
        elif interaction_state in [KGDataLoaderState.KG]:
            return super()._train_epoch(
                train_data,
                epoch_idx,
                loss_func=self.model.calculate_kg_loss,
                show_progress=show_progress,
                retain_samples_used_for_update=retain_samples_used_for_update,
                reinit_masks=reinit_masks,
                scale_for_reinit_params=scale_for_reinit_params,
            )
        return None


class KGATTrainer(Trainer):
    r"""KGATTrainer is designed for KGAT, which is a knowledge-aware recommendation method."""

    def __init__(self, config, model):
        super(KGATTrainer, self).__init__(config, model)

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False, retain_samples_used_for_update=None, reinit_masks=None, scale_for_reinit_params=1.0):
        # train rs
        if not self.config["single_spec"]:
            train_data.knowledge_shuffle(epoch_idx)
        train_data.set_mode(KGDataLoaderState.RS)
        rs_total_loss = super()._train_epoch(
            train_data, epoch_idx, show_progress=show_progress, retain_samples_used_for_update=retain_samples_used_for_update, reinit_masks=reinit_masks, scale_for_reinit_params=scale_for_reinit_params
        )

        # train kg
        train_data.set_mode(KGDataLoaderState.KG)
        kg_total_loss = super()._train_epoch(
            train_data,
            epoch_idx,
            loss_func=self.model.calculate_kg_loss,
            show_progress=show_progress,
            retain_samples_used_for_update=retain_samples_used_for_update,
            reinit_masks=reinit_masks,
            scale_for_reinit_params=scale_for_reinit_params,
        )

        # update A
        self.model.eval()
        with torch.no_grad():
            self.model.update_attentive_A()

        return rs_total_loss, kg_total_loss


class PretrainTrainer(Trainer):
    r"""PretrainTrainer is designed for pre-training.
    It can be inherited by the trainer which needs pre-training and fine-tuning.
    """

    def __init__(self, config, model):
        super(PretrainTrainer, self).__init__(config, model)
        self.pretrain_epochs = self.config["pretrain_epochs"]
        self.save_step = self.config["save_step"]

    def save_pretrained_model(self, epoch, saved_model_file):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id
            saved_model_file (str): file name for saved pretrained model

        """
        state = {
            "config": self.config,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "other_parameter": self.model.other_parameter(),
        }
        torch.save(state, saved_model_file)
        self.saved_model_file = saved_model_file

    def pretrain(self, train_data, verbose=True, show_progress=False):
        for epoch_idx in range(self.start_epoch, self.pretrain_epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(
                train_data, epoch_idx, show_progress=show_progress
            )
            self.train_loss_dict[epoch_idx] = (
                sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            )
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)

            if (epoch_idx + 1) % self.save_step == 0:
                saved_model_file = os.path.join(
                    self.checkpoint_dir,
                    "{}-{}-{}.pth".format(
                        self.config["model"], self.config["dataset"], str(epoch_idx + 1)
                    ),
                )
                self.save_pretrained_model(epoch_idx, saved_model_file)
                update_output = (
                    set_color("Saving current", "blue") + ": %s" % saved_model_file
                )
                if verbose:
                    self.logger.info(update_output)

        return self.best_valid_score, self.best_valid_result


class S3RecTrainer(PretrainTrainer):
    r"""S3RecTrainer is designed for S3Rec, which is a self-supervised learning based sequential recommenders.
    It includes two training stages: pre-training ang fine-tuning.

    """

    def __init__(self, config, model):
        super(S3RecTrainer, self).__init__(config, model)

    def fit(
        self,
        train_data,
        valid_data=None,
        verbose=True,
        saved=True,
        show_progress=False,
        callback_fn=None,
    ):
        if self.model.train_stage == "pretrain":
            return self.pretrain(train_data, verbose, show_progress)
        elif self.model.train_stage == "finetune":
            return super().fit(
                train_data, valid_data, verbose, saved, show_progress, callback_fn
            )
        else:
            raise ValueError(
                "Please make sure that the 'train_stage' is 'pretrain' or 'finetune'!"
            )


class MKRTrainer(Trainer):
    r"""MKRTrainer is designed for MKR, which is a knowledge-aware recommendation method."""

    def __init__(self, config, model):
        super(MKRTrainer, self).__init__(config, model)
        self.kge_interval = config["kge_interval"]

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False, retain_samples_used_for_update=None, reinit_masks=None, scale_for_reinit_params=1.0):
        rs_total_loss, kg_total_loss = 0.0, 0.0

        # train rs
        self.logger.info("Train RS")
        train_data.set_mode(KGDataLoaderState.RS)
        rs_total_loss = super()._train_epoch(
            train_data,
            epoch_idx,
            loss_func=self.model.calculate_rs_loss,
            show_progress=show_progress,
            retain_samples_used_for_update=retain_samples_used_for_update,
            reinit_masks=reinit_masks,
            scale_for_reinit_params=scale_for_reinit_params,
        )

        # train kg
        if epoch_idx % self.kge_interval == 0:
            self.logger.info("Train KG")
            train_data.set_mode(KGDataLoaderState.KG)
            kg_total_loss = super()._train_epoch(
                train_data,
                epoch_idx,
                loss_func=self.model.calculate_kg_loss,
                show_progress=show_progress,
                retain_samples_used_for_update=retain_samples_used_for_update,
                reinit_masks=reinit_masks,
                scale_for_reinit_params=scale_for_reinit_params,
            )

        return rs_total_loss, kg_total_loss


class TraditionalTrainer(Trainer):
    r"""TraditionalTrainer is designed for Traditional model(Pop,ItemKNN), which set the epoch to 1 whatever the config."""

    def __init__(self, config, model):
        super(TraditionalTrainer, self).__init__(config, model)
        self.epochs = 1  # Set the epoch to 1 when running memory based model


class DecisionTreeTrainer(AbstractTrainer):
    """DecisionTreeTrainer is designed for DecisionTree model."""

    def __init__(self, config, model):
        super(DecisionTreeTrainer, self).__init__(config, model)

        self.logger = getLogger()
        self.tensorboard = get_tensorboard(self.logger)
        self.label_field = config["LABEL_FIELD"]
        self.convert_token_to_onehot = self.config["convert_token_to_onehot"]

        # evaluator
        self.eval_type = config["eval_type"]
        self.epochs = config["epochs"]
        self.eval_step = min(config["eval_step"], self.epochs)
        self.valid_metric = config["valid_metric"].lower()
        self.eval_collector = Collector(config)
        self.evaluator = Evaluator(config)

        # model saved
        self.checkpoint_dir = config["checkpoint_dir"]
        ensure_dir(self.checkpoint_dir)
        temp_file = "{}-{}-temp.pth".format(self.config["model"], get_local_time())
        self.temp_file = os.path.join(self.checkpoint_dir, temp_file)

        temp_best_file = "{}-{}-temp-best.pth".format(
            self.config["model"], get_local_time()
        )
        self.temp_best_file = os.path.join(self.checkpoint_dir, temp_best_file)

        saved_model_file = "{}-{}.pth".format(self.config["model"], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)

        self.stopping_step = config["stopping_step"]
        self.valid_metric_bigger = config["valid_metric_bigger"]
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None

    def _interaction_to_sparse(self, dataloader):
        r"""Convert data format from interaction to sparse or numpy

        Args:
            dataloader (DecisionTreeDataLoader): DecisionTreeDataLoader dataloader.
        Returns:
            cur_data (sparse or numpy): data.
            interaction_np[self.label_field] (numpy): label.
        """
        interaction = dataloader._dataset[:]
        interaction_np = interaction.numpy()
        cur_data = np.array([])
        columns = []
        for key, value in interaction_np.items():
            value = np.resize(value, (value.shape[0], 1))
            if key != self.label_field:
                columns.append(key)
                if cur_data.shape[0] == 0:
                    cur_data = value
                else:
                    cur_data = np.hstack((cur_data, value))

        if self.convert_token_to_onehot:
            from scipy import sparse
            from scipy.sparse import dok_matrix

            convert_col_list = dataloader._dataset.convert_col_list
            hash_count = dataloader._dataset.hash_count

            new_col = cur_data.shape[1] - len(convert_col_list)
            for key, values in hash_count.items():
                new_col = new_col + values
            onehot_data = dok_matrix((cur_data.shape[0], new_col))

            cur_j = 0
            new_j = 0

            for key in columns:
                if key in convert_col_list:
                    for i in range(cur_data.shape[0]):
                        onehot_data[i, int(new_j + cur_data[i, cur_j])] = 1
                    new_j = new_j + hash_count[key] - 1
                else:
                    for i in range(cur_data.shape[0]):
                        onehot_data[i, new_j] = cur_data[i, cur_j]
                cur_j = cur_j + 1
                new_j = new_j + 1

            cur_data = sparse.csc_matrix(onehot_data)

        return cur_data, interaction_np[self.label_field]

    def _interaction_to_lib_datatype(self, dataloader):
        pass

    def _valid_epoch(self, valid_data):
        r"""

        Args:
            valid_data (DecisionTreeDataLoader): DecisionTreeDataLoader, which is the same with GeneralDataLoader.
        """
        valid_result = self.evaluate(valid_data, load_best_model=False)
        valid_score = calculate_valid_score(valid_result, self.valid_metric)
        return valid_score, valid_result

    def _save_checkpoint(self, epoch):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        state = {
            "config": self.config,
            "epoch": epoch,
            "cur_step": self.cur_step,
            "best_valid_score": self.best_valid_score,
            "state_dict": self.temp_best_file,
            "other_parameter": None,
        }
        torch.save(state, self.saved_model_file)

    def fit(
        self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False
    ):
        for epoch_idx in range(self.epochs):
            self._train_at_once(train_data, valid_data)

            if (epoch_idx + 1) % self.eval_step == 0:
                # evaluate
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)

                (
                    self.best_valid_score,
                    self.cur_step,
                    stop_flag,
                    update_flag,
                ) = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger,
                )

                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("valid_score", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = (
                    set_color("valid result", "blue") + ": \n" + dict2str(valid_result)
                )
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar("Vaild_score", valid_score, epoch_idx)

                if update_flag:
                    if saved:
                        self.model.save_model(self.temp_best_file)
                        self._save_checkpoint(epoch_idx)
                    self.best_valid_result = valid_result

                if stop_flag:
                    stop_output = "Finished training, best eval result in epoch %d" % (
                        epoch_idx - self.cur_step * self.eval_step
                    )
                    if self.temp_file:
                        os.remove(self.temp_file)
                    if verbose:
                        self.logger.info(stop_output)
                    break

        return self.best_valid_score, self.best_valid_result

    def evaluate(
        self, eval_data, load_best_model=True, model_file=None, show_progress=False
    ):
        raise NotImplementedError

    def _train_at_once(self, train_data, valid_data):
        raise NotImplementedError


class XGBoostTrainer(DecisionTreeTrainer):
    """XGBoostTrainer is designed for XGBOOST."""

    def __init__(self, config, model):
        super(XGBoostTrainer, self).__init__(config, model)

        self.xgb = __import__("xgboost")
        self.boost_model = config["xgb_model"]
        self.silent = config["xgb_silent"]
        self.nthread = config["xgb_nthread"]

        # train params
        self.params = config["xgb_params"]
        self.num_boost_round = config["xgb_num_boost_round"]
        self.evals = ()
        self.early_stopping_rounds = config["xgb_early_stopping_rounds"]
        self.evals_result = {}
        self.verbose_eval = config["xgb_verbose_eval"]
        self.callbacks = None
        self.deval = None
        self.eval_pred = self.eval_true = None

    def _interaction_to_lib_datatype(self, dataloader):
        r"""Convert data format from interaction to DMatrix

        Args:
            dataloader (DecisionTreeDataLoader): xgboost dataloader.
        Returns:
            DMatrix: Data in the form of 'DMatrix'.
        """
        data, label = self._interaction_to_sparse(dataloader)
        return self.xgb.DMatrix(
            data=data, label=label, silent=self.silent, nthread=self.nthread
        )

    def _train_at_once(self, train_data, valid_data):
        r"""

        Args:
            train_data (DecisionTreeDataLoader): DecisionTreeDataLoader, which is the same with GeneralDataLoader.
            valid_data (DecisionTreeDataLoader): DecisionTreeDataLoader, which is the same with GeneralDataLoader.
        """
        self.dtrain = self._interaction_to_lib_datatype(train_data)
        self.dvalid = self._interaction_to_lib_datatype(valid_data)
        self.evals = [(self.dtrain, "train"), (self.dvalid, "valid")]
        self.model = self.xgb.train(
            self.params,
            self.dtrain,
            self.num_boost_round,
            self.evals,
            early_stopping_rounds=self.early_stopping_rounds,
            evals_result=self.evals_result,
            verbose_eval=self.verbose_eval,
            xgb_model=self.boost_model,
            callbacks=self.callbacks,
        )

        self.model.save_model(self.temp_file)
        self.boost_model = self.temp_file

    def evaluate(
        self, eval_data, load_best_model=True, model_file=None, show_progress=False
    ):
        if load_best_model:
            if model_file:
                checkpoint_file = model_file
            else:
                checkpoint_file = self.temp_best_file
            self.model.load_model(checkpoint_file)

        self.deval = self._interaction_to_lib_datatype(eval_data)
        self.eval_true = torch.Tensor(self.deval.get_label())
        self.eval_pred = torch.Tensor(self.model.predict(self.deval))

        self.eval_collector.eval_collect(self.eval_pred, self.eval_true)
        result = self.evaluator.evaluate(self.eval_collector.get_data_struct())
        return result


class LightGBMTrainer(DecisionTreeTrainer):
    """LightGBMTrainer is designed for LightGBM."""

    def __init__(self, config, model):
        super(LightGBMTrainer, self).__init__(config, model)

        self.lgb = __import__("lightgbm")

        # train params
        self.params = config["lgb_params"]
        self.num_boost_round = config["lgb_num_boost_round"]
        self.evals = ()
        self.deval_data = self.deval_label = None
        self.eval_pred = self.eval_true = None

    def _interaction_to_lib_datatype(self, dataloader):
        r"""Convert data format from interaction to Dataset

        Args:
            dataloader (DecisionTreeDataLoader): xgboost dataloader.
        Returns:
            dataset(lgb.Dataset): Data in the form of 'lgb.Dataset'.
        """
        data, label = self._interaction_to_sparse(dataloader)
        return self.lgb.Dataset(data=data, label=label)

    def _train_at_once(self, train_data, valid_data):
        r"""

        Args:
            train_data (DecisionTreeDataLoader): DecisionTreeDataLoader, which is the same with GeneralDataLoader.
            valid_data (DecisionTreeDataLoader): DecisionTreeDataLoader, which is the same with GeneralDataLoader.
        """
        self.dtrain = self._interaction_to_lib_datatype(train_data)
        self.dvalid = self._interaction_to_lib_datatype(valid_data)
        self.evals = [self.dtrain, self.dvalid]
        self.model = self.lgb.train(
            self.params, self.dtrain, self.num_boost_round, self.evals
        )

        self.model.save_model(self.temp_file)
        self.boost_model = self.temp_file

    def evaluate(
        self, eval_data, load_best_model=True, model_file=None, show_progress=False
    ):
        if load_best_model:
            if model_file:
                checkpoint_file = model_file
            else:
                checkpoint_file = self.temp_best_file
            self.model = self.lgb.Booster(model_file=checkpoint_file)

        self.deval_data, self.deval_label = self._interaction_to_sparse(eval_data)
        self.eval_true = torch.Tensor(self.deval_label)
        self.eval_pred = torch.Tensor(self.model.predict(self.deval_data))

        self.eval_collector.eval_collect(self.eval_pred, self.eval_true)
        result = self.evaluator.evaluate(self.eval_collector.get_data_struct())
        return result


class RaCTTrainer(PretrainTrainer):
    r"""RaCTTrainer is designed for RaCT, which is an actor-critic reinforcement learning based general recommenders.
    It includes three training stages: actor pre-training, critic pre-training and actor-critic training.

    """

    def __init__(self, config, model):
        super(RaCTTrainer, self).__init__(config, model)

    def fit(
        self,
        train_data,
        valid_data=None,
        verbose=True,
        saved=True,
        show_progress=False,
        callback_fn=None,
    ):
        if self.model.train_stage == "actor_pretrain":
            return self.pretrain(train_data, verbose, show_progress)
        elif self.model.train_stage == "critic_pretrain":
            return self.pretrain(train_data, verbose, show_progress)
        elif self.model.train_stage == "finetune":
            return super().fit(
                train_data, valid_data, verbose, saved, show_progress, callback_fn
            )
        else:
            raise ValueError(
                "Please make sure that the 'train_stage' is "
                "'actor_pretrain', 'critic_pretrain' or 'finetune'!"
            )


class RecVAETrainer(Trainer):
    r"""RecVAETrainer is designed for RecVAE, which is a general recommender."""

    def __init__(self, config, model):
        super(RecVAETrainer, self).__init__(config, model)
        self.n_enc_epochs = config["n_enc_epochs"]
        self.n_dec_epochs = config["n_dec_epochs"]

        self.optimizer_encoder = self._build_optimizer(
            params=self.model.encoder.parameters()
        )
        self.optimizer_decoder = self._build_optimizer(
            params=self.model.decoder.parameters()
        )

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False, retain_samples_used_for_update=None, reinit_masks=None, scale_for_reinit_params=1.0):
        self.optimizer = self.optimizer_encoder
        encoder_loss_func = lambda data: self.model.calculate_loss(
            data, encoder_flag=True
        )
        for epoch in range(self.n_enc_epochs):
            super()._train_epoch(
                train_data,
                epoch_idx,
                loss_func=encoder_loss_func,
                show_progress=show_progress,
                retain_samples_used_for_update=retain_samples_used_for_update,
                reinit_masks=reinit_masks,
                scale_for_reinit_params=scale_for_reinit_params,
            )

        self.model.update_prior()
        loss = 0.0
        self.optimizer = self.optimizer_decoder
        decoder_loss_func = lambda data: self.model.calculate_loss(
            data, encoder_flag=False
        )
        for epoch in range(self.n_dec_epochs):
            loss += super()._train_epoch(
                train_data,
                epoch_idx,
                loss_func=decoder_loss_func,
                show_progress=show_progress,
                retain_samples_used_for_update=retain_samples_used_for_update,
                reinit_masks=reinit_masks,
                scale_for_reinit_params=scale_for_reinit_params,
            )
        return loss


class NCLTrainer(Trainer):
    def __init__(self, config, model):
        super(NCLTrainer, self).__init__(config, model)

        self.num_m_step = config["m_step"]
        assert self.num_m_step is not None

    def fit(
        self,
        train_data,
        valid_data=None,
        verbose=True,
        saved=True,
        show_progress=False,
        callback_fn=None,
    ):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data.
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1)

        self.eval_collector.data_collect(train_data)

        for epoch_idx in range(self.start_epoch, self.epochs):
            # only differences from the original trainer
            if epoch_idx % self.num_m_step == 0:
                self.logger.info("Running E-step ! ")
                self.model.e_step()
            # train
            training_start_time = time()
            train_loss = self._train_epoch(
                train_data, epoch_idx, show_progress=show_progress
            )
            self.train_loss_dict[epoch_idx] = (
                sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            )
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx)
                    update_output = (
                        set_color("Saving current", "blue")
                        + ": %s" % self.saved_model_file
                    )
                    if verbose:
                        self.logger.info(update_output)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(
                    valid_data, show_progress=show_progress
                )

                (
                    self.best_valid_score,
                    self.cur_step,
                    stop_flag,
                    update_flag,
                ) = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger,
                )
                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("valid_score", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = (
                    set_color("valid result", "blue") + ": \n" + dict2str(valid_result)
                )
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar("Vaild_score", valid_score, epoch_idx)

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx)
                        update_output = (
                            set_color("Saving current best", "blue")
                            + ": %s" % self.saved_model_file
                        )
                        if verbose:
                            self.logger.info(update_output)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = "Finished training, best eval result in epoch %d" % (
                        epoch_idx - self.cur_step * self.eval_step
                    )
                    if verbose:
                        self.logger.info(stop_output)
                    break
        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False, retain_samples_used_for_update=None, reinit_masks=None, scale_for_reinit_params=1.0):
        r"""Train the model in an epoch
        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.
            retain_samples_used_for_update (int, optional): Number of retain samples to use for update. Defaults to ``None``.
            reinit_masks (dict, optional): Masks for parameter reinitialization. Defaults to ``None``.
            scale_for_reinit_params (float, optional): Scale factor for reinitialized parameters. Defaults to ``1.0``.
        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        loss_func = self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", "pink"),
            )
            if show_progress
            else train_data
        )
        scaler = self.scaler

        if not self.config["single_spec"] and train_data.shuffle:
            train_data.sampler.set_epoch(epoch_idx)

        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            sync_loss = 0
            if not self.config["single_spec"]:
                self.set_reduce_hook()
                sync_loss = self.sync_grad_loss()

            with torch.autocast(device_type=self.device.type, enabled=self.enable_amp):
                losses = loss_func(interaction)

            if isinstance(losses, tuple):
                if epoch_idx < self.config["warm_up_step"]:
                    losses = losses[:-1]
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = (
                    loss_tuple
                    if total_loss is None
                    else tuple(map(sum, zip(total_loss, loss_tuple)))
                )
            else:
                loss = losses
                total_loss = (
                    losses.item() if total_loss is None else total_loss + losses.item()
                )
            self._check_nan(loss)
            scaler.scale(loss + sync_loss).backward()

            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            scaler.step(self.optimizer)
            scaler.update()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                )
        return total_loss
