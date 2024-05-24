# pl_callbacks.py

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    # ProgressBar,
    RichProgressBar,
    # GPUStatsMonitor,
    StochasticWeightAveraging,
)
from pytorch_lightning.tuner import Tuner


def find_optimal_lr(trainer, model):
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model)
    return lr_finder.suggestion()


def get_callbacks(path, metric="Evalloss", save_interval_steps=1000):
    # Model Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor=metric,
        dirpath=path,
        filename="bert-{epoch:02d}-{step:d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
        every_n_train_steps=save_interval_steps,  # Save every n steps
        save_last=True,  # Optionally save the last checkpoint
    )
    early_stopping_callback = EarlyStopping(
        monitor=metric, patience=3, verbose=True, mode="min"
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    rich_progress_bar = RichProgressBar()
    # gpu_stats = GPUStatsMonitor()
    swa = StochasticWeightAveraging(
        swa_epoch_start=5,  # Start SWA after 5 epochs (example value)
        annealing_epochs=10,  # Anneal the learning rates over 10 epochs (example value)
        annealing_strategy='cos',  # Anneal using cosine annealing (example value)
        swa_lrs=0.05  # Example value, specify the SWA learning rates here
    )

    return [
        checkpoint_callback,
        # early_stopping_callback,
        lr_monitor,
        rich_progress_bar,
        # gpu_stats,
        swa,
    ]
