import torch
from recbole.data.dataset.sequential_dataset import SequentialDataset
from recbole.data.interaction import Interaction
from recbole.data.dataset.dataset import Dataset
import os
import pandas as pd


class SequentialUnlearningDataset(SequentialDataset):
    """
    Stream the raw interactions once.  
    Whenever the current row matches a (session, item, time) triple
    that should be unlearned, emit exactly one (prefix → next_item) pair.
    """

    def __init__(self, config):
        super().__init__(config)                         # only grabs field names

        self.uid_col  = self.uid_field                  # e.g. "session_id"
        self.iid_col  = self.iid_field                  # e.g. "item_id"
        self.time_col = self.time_field                 # e.g. "timestamp"
        self.max_len  = config["MAX_ITEM_LIST_LENGTH"]

        unlearning_samples_path = os.path.join(
            config["data_path"], f"{config['dataset']}_unlearn_pairs_{config['unlearning_sample_selection_method']}_seed_{config['unlearn_sample_selection_seed']}_unlearning_fraction_{float(config['unlearning_fraction'])}.inter"
        )
        self.unlearning_samples = pd.read_csv(unlearning_samples_path, sep="\t")

        inter_path = os.path.join(
            config["data_path"], f"{config['dataset']}.inter"
        )
        if not os.path.isfile(inter_path):
            raise FileNotFoundError(f"Couldn't find {inter_path}")
        raw_inter_df = pd.read_csv(inter_path, sep="\t")

        # Dataframe MUST already be sorted by (uid, time) ascending
        self.raw_df = raw_inter_df[[self.uid_col, self.iid_col, self.time_col]]
        unlearn_session_ids = set(self.unlearning_samples["userid:token"])
        # filter out sessions not needed for unlearning this specific forget set
        self.raw_df = self.raw_df[self.raw_df[self.uid_col] in unlearn_session_ids]

    # ---------- helpers -----------------------------------------------------

    def _row(self, prefix, target):
        """Build one Interaction row from prefix list and target item id."""
        L = len(prefix)
        padded = torch.zeros(self.max_len, dtype=torch.long)
        padded[-L:] = torch.tensor(prefix[-self.max_len:], dtype=torch.long)
        return Interaction({
            self.ITEM_SEQ:     padded.unsqueeze(0),          # [1, max_len]
            self.ITEM_SEQ_LEN: torch.tensor([L]),            # [1]
            self.POS_ITEM_ID:  torch.tensor([target])        # [1]
        })

    # ---------- RecBole hook -------------------------------------------------

    def build(self, eval_setting=None):
        examples   = []
        cur_sess   = None
        prefix_seq = []          # running list of item_ids in current session
        unlearn_df_idx = 0

        # single pass over the (already session-sorted) dataframe
        for uid, iid, ts in self.raw_df.itertuples(index=False):
            if uid != cur_sess:                # new session begins
                cur_sess, prefix_seq = uid, []

            if (uid, iid, ts) == tuple(self.unlearn_samples.iloc[unlearn_df_idx]) and prefix_seq:
                examples.append(self._row(prefix_seq, iid))
                unlearn_df_idx += 1

            prefix_seq.append(iid)             # extend prefix for next rows

        if not examples:
            raise RuntimeError("No unlearning triples were hit.")

        self.inter_feat = Interaction.cat(examples, axis=0)
        return Dataset.build(self, eval_setting)
