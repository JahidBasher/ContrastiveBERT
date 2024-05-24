import os
import torch
from pytorch_lightning import Trainer
# from pytorch_lightning.plugins import MixedPrecisionPlugin
from pytorch_lightning.loggers import TensorBoardLogger
from src.trainer.training import ContrastiveBERT
from src.data_modules.tokenizer.tokenizers import MultilingualContrastiveTokenizer
from src.data_modules.data_module import ContrastiveBERTDataModule
from src.data_modules.collate_fn import collate_fn
from src.model import BERT, BERTLMC
from src.callbacks import get_callbacks, find_optimal_lr


from config import Config


def main(cfg):
    # Initialize tokenizer
    m_tokenizer = MultilingualContrastiveTokenizer(
        tokenizer_path=cfg.tokenizer_path,
        vocab_size=cfg.vocab_size,
        max_token_length=cfg.max_token_length,
        pad_token=cfg.pad_token,
        unk_token=cfg.unk_token,
        start_token=cfg.start_token,
        end_token=cfg.end_token,
    )
    m_tokenizer.load_from_disk(cfg.tokenizer_model_path)

    # Initialize data module
    data_module = ContrastiveBERTDataModule(
        cfg=cfg,
        tokenizer=m_tokenizer,
        collate_fn=collate_fn,
        printer=print
    )

    # Initialize BERT model
    bert = BERT(
        m_tokenizer.vocab_size,
        hidden=cfg.hidden_dim,
        n_layers=cfg.layers,
        attn_heads=cfg.attn_heads,
    )
    cbert_model = BERTLMC(bert, m_tokenizer.vocab_size)

    # Initialize Lightning logger
    logger = TensorBoardLogger(
        save_dir=cfg.logger_save_dir, version=cfg.logger_version, name=cfg.logger_name
    )

    # Initialize ContrastiveBERT Lightning module
    cbert = ContrastiveBERT(cfg=cfg, model=cbert_model)

    # Initialize Trainer
    trainer = Trainer(
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        devices=cfg.devices,
        accelerator=cfg.accelerator,
        logger=logger,
        max_epochs=cfg.max_epochs,
        precision=cfg.precision,
        callbacks=get_callbacks(cfg.output_path, save_interval_steps=cfg.save_interval),
        val_check_interval=cfg.save_interval,
        enable_model_summary=True
        # num_sanity_val_steps=0  # Uncomment if needed
    )
    # Train the model
    trainer.fit(cbert, datamodule=data_module, ckpt_path=cfg.pretrained_path)


if __name__ == "__main__":
    # Instantiate configuration
    cfg = Config()
    main(cfg)
