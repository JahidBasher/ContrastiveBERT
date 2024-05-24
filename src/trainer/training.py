import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning import LightningModule
from src.trainer.schedulers import ScheduledOptimLR


class ContrastiveBERT(LightningModule):
    def __init__(self, cfg, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.model = model
        self.time_tracker = None
        self.criterion = nn.NLLLoss(ignore_index=0).to(self.cfg.device)
        self.contrastive_criterion = nn.CrossEntropyLoss().to(self.cfg.device)
        self.train_step_outputs = []
        self.validation_step_outputs = []

    def setup(self, stage):
        self.time_tracker = time.time()

    def configure_optimizers(self):
        optimizer = Adam(
            self.model.parameters(),
            lr=self.cfg.lr,
            betas=self.cfg.betas,
            weight_decay=self.cfg.weight_decay,
        )

        scheduler = ScheduledOptimLR(
            optimizer, self.cfg.hidden_dim, n_warmup_steps=self.cfg.warmup_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "metric_to_track",
                "name": "learning_rate",
                "interval": "step",
            },
        }

    def training_step(self, batch, batch_idx):
        loss_dict = self.forward_batch(batch)
        self.log_dict(loss_dict)
        self.train_step_outputs.append(loss_dict)
        return loss_dict

    def on_train_epoch_end(self):
        avg_loss, avg_contrastive_loss, avg_mlm_loss = (
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
        )

        for step_out in self.train_step_outputs:
            avg_loss += step_out["loss"]
            avg_contrastive_loss += step_out["StepContrastiveLoss"]
            avg_mlm_loss += step_out["StepMLMLoss"]

        log_dict = {
            "EpochLoss": avg_loss,
            "EpochContrastiveLoss": avg_contrastive_loss,
            "EpochMLMLoss": avg_mlm_loss,
        }
        self.print(log_dict)
        self.train_step_outputs.clear()
        self.log_dict(log_dict)

    def validation_step(self, batch, batch_idx):
        loss_dict = self.forward_batch(batch, eval_mode=True)
        self.log_dict(loss_dict)
        self.validation_step_outputs.append(loss_dict)
        return loss_dict

    def on_validation_epoch_end(self):
        Top1Accuracy, Top5Accuracy = 0, 0

        for step_out in self.validation_step_outputs:
            Top1Accuracy += step_out.get("StepTop1Accuracy", 0)
            Top5Accuracy += step_out.get("StepTop5Accuracy", 0)

        accuracy = {
            "Top1Accuracy": round(Top1Accuracy / len(self.validation_step_outputs), 4),
            "Top5Accuracy": round(Top5Accuracy / len(self.validation_step_outputs), 4),
        }

        self.print(accuracy)
        self.validation_step_outputs.clear()
        self.log_dict(accuracy)

    def forward_batch(self, data, eval_mode=False):
        mask_lm_output, cls_representation = self.model.forward(
            data["bert_input"], data["segment_label"]
        )
        cls_logits, cls_labels = self.info_nce_loss(cls_representation)

        contrastive_loss = self.contrastive_criterion(cls_logits, cls_labels)
        mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])
        loss = contrastive_loss + mask_loss

        output = {}
        if eval_mode:
            top1_5 = get_accuracy(cls_logits, cls_labels, topk=(1, 5))
            output["EvalStepTop1Accuracy"] = round(top1_5[0].item(), 4)
            output["EvalStepTop5Accuracy"] = round(top1_5[1].item(), 4)

            output.update(
                {
                    "EvalStepContrastiveLoss": contrastive_loss,
                    "EvalStepMLMLoss": mask_loss,
                    "Evalloss": loss,
                }
            )
        else:
            output.update(
                {
                    "StepContrastiveLoss": contrastive_loss,
                    "StepMLMLoss": mask_loss,
                    "loss": loss,
                }
            )
        return output

    def info_nce_loss(self, features):
        batch_size = features.shape[0]
        assert batch_size % 2 == 0, "batch size must be even"

        labels = torch.cat(
            [torch.arange(batch_size // 2) for _ in range(self.cfg.n_views)], dim=0
        )
        labels = (
            (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(self.cfg.device)
        )
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.cfg.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1
        )

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.cfg.device)

        logits /= self.cfg.temperature
        return logits, labels


def get_accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

    return res
