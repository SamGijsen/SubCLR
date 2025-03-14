import torch
import numpy as np
import torch.distributed as dist

from torch import Tensor

from typing import List, Tuple
from utils.utils import score, majority_vote, load_DDP_state_dict
from base.base_trainer import BaseTrainer


class SV_Trainer(BaseTrainer):
    """
    Trainer class for supervised learning using two models.
    Referred to here as an `encoder` and `head` model.
    These are ran sequentially as in out=head(encoder(in)).

    sampling level: channel
    inference level: epoch
    """

    def __init__(self, current_setting: str, cfg: dict, sub_ids: list, to_opt: list, 
                 local_rank: int=0, device: str="cfg", DDP: bool=False):
        super().__init__(current_setting, cfg, sub_ids, to_opt, local_rank, device, DDP)

    def _construct_optimizer_scheduler(self, models: list): 
        """Sets self.optimizer and self.scheduler for models."""

        params_to_optimize = [] 
        for model, opt_flag in zip(models, self.to_opt):
            if opt_flag:
                params_to_optimize.extend(model.parameters())

        self.optimizer = torch.optim.Adam(
            params=params_to_optimize,
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=self.patience,
            factor=0.5
        )

    def _train_epoch(self, models: list, epoch: int) -> Tuple[List, List, float, list]:

        model = models[0]
        model.train()

        losses_this_epoch = []

        for batch in self.train_dataloader:
            self.optimizer.zero_grad()

            x, y = batch[0].to(self.device), batch[1].to(self.device)

            with torch.autocast("cuda", enabled=self.amp):

                if self.model_type == "EEG_ResNet":
                    out = model(x.view(-1, 1, self.in_channels, self.rep_dim))# .view(-1, 1)
                else:
                    out = model(x.permute(0,-1,1).unsqueeze(1))
                loss = self.loss_function(out, y)

            if self.amp: # use scaler
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            losses_this_epoch.append(loss.item()) 

        if self.DDP: # Reduce across GPUs
            loss_tensor = torch.tensor(np.array(losses_this_epoch)).mean().to(self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / self.wsize
        else:
            avg_loss = np.mean(losses_this_epoch)

        return [model], losses_this_epoch, avg_loss, []
    
    def validate(self, models: list, test: bool=False) -> Tuple[float, float, List, List]:
        """
        Validates on using self.val_dataloader.
        Return validation loss and evaluation metric.
        """
        loss_total = 0.
        ys_true, ys_pred = [], []

        model = models[0]

        if test: # Load best model in test-mode
            model = self._load_DDP_state_dict(model, self.model_save_path + "/model_0_best.pt")
            dataloader = self.test_dataloader
            subject_ids = self.sub_ids["test"]
        else:
            dataloader = self.val_dataloader
            subject_ids = self.sub_ids["val"]

        model.eval()

        with torch.no_grad():
            for batch in dataloader:

                x, y = batch[0].to(self.device), batch[1].to(self.device)

                with torch.autocast("cuda", enabled=self.amp):

                    #out = model(x.view(-1, 1, self.n_time_samples, self.in_channels))
                    if self.model_type == "EEG_ResNet":
                        out = model(x.view(-1, 1, self.in_channels, self.rep_dim))# .view(-1, 1)
                    else:
                        out = model(x.permute(0,-1,1).unsqueeze(1))
                    loss = self.loss_function(out, y)
                
                loss_total += loss.item()

                ys_true.extend(y.cpu().numpy())
                ys_pred.extend(out.cpu().numpy())

            loss_total /= len(dataloader)

        # Performance evaluation.
        ys_true = np.concatenate(ys_true)
        ys_pred = np.concatenate(ys_pred)

        if self.DDP: # Reduce across GPUs

            loss_total_tensor = torch.tensor(loss_total).to(self.device)
            dist.all_reduce(loss_total_tensor, op=dist.ReduceOp.SUM)
            loss_total = loss_total_tensor.item() / self.wsize

            ys_pred = torch.tensor(ys_pred, dtype=torch.float32).to(self.device)
            ys_pred_list = [torch.zeros(ys_pred.shape[0], dtype=torch.float32).to(self.device) 
                            for _ in range(self.wsize)]
            dist.all_gather(ys_pred_list, ys_pred)
            ys_pred = torch.cat((ys_pred_list), dim=0).cpu().numpy()
            
            ys_true = torch.tensor(ys_true, dtype=torch.float32).to(self.device)
            ys_true_list = [torch.zeros(ys_true.shape[0], dtype=torch.float32).to(self.device) 
                            for _ in range(self.wsize)]
            dist.all_gather(ys_true_list, ys_true)
            ys_true = torch.cat((ys_true_list), dim=0).cpu().numpy()

        sub_ys_true, sub_ys_pred, metrics = score(ys_true, ys_pred, subject_ids, self.n_classes, True, True)

        return loss_total, metrics, sub_ys_true, sub_ys_pred
           
    
