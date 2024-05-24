import time

import pytorch_lightning as pl
import torch
from overrides import overrides
from torch.utils.data.dataloader import DataLoader

from src.data_modules.dataset.contrastive_dataset import SimBERTDataset


class ContrastiveBERTDataModule(pl.LightningDataModule):
    def __init__(self, cfg, tokenizer, collate_fn, printer):
        super().__init__()
        self.cfg = cfg
        self.train_loader = None
        self.val_loader = None
        self.tokenizer = tokenizer
        self.collate_fn = collate_fn
        self.printer = printer

    @overrides
    def setup(self, stage=None):
        self.train_loader = self._get_train_loader()
        self.val_loader = self._get_val_loader(mode="val")

    @overrides
    def train_dataloader(self):
        return self.train_loader

    @overrides
    def val_dataloader(self):
        return self.val_loader

    def _get_val_loader(self, mode):
        return self._get_val_test_loaders(mode="val")

    def _get_test_loader(self):
        return self._get_val_test_loaders(mode="val")

    def _get_train_loader(self):
        start_time = time.time()
        dataset = SimBERTDataset(
            corpus_path=self.cfg.corpus_path,
            tokenizer=self.tokenizer,
            seq_len=self.cfg.max_seq_length,
            padding=self.cfg.padding,
            encoding="utf-8",
            corpus_lines=0,
            on_memory=self.cfg.on_memory
        )
        data_loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=False,
            collate_fn=self.collate_fn,
            prefetch_factor=self.cfg.prefetch_factor,
        )

        elapsed_time = time.time() - start_time
        self.printer(
            f"Elapsed time for loading training data: {elapsed_time}", flush=True
        )

        return data_loader

    def _get_val_test_loaders(self, mode):
        dataset = SimBERTDataset(
            corpus_path=self.cfg.eval_corpus_path,
            tokenizer=self.tokenizer,
            seq_len=self.cfg.max_seq_length,
            padding=self.cfg.padding,
            encoding="utf-8",
            corpus_lines=0,
            on_memory=self.cfg.on_memory,
            mode=mode,
        )

        data_loader = DataLoader(
            dataset,
            batch_size=self.cfg.eval_batch_size,
            shuffle=False,
            num_workers=self.cfg.eval_num_workers,
            pin_memory=False,
            collate_fn=self.collate_fn,
        )
        return data_loader
