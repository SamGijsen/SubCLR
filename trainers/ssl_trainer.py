import torch
import torch.distributed as dist
import numpy as np
import math
import h5py
import os

from typing import List, Tuple
from base.base_trainer import BaseTrainer
from models.loss import BYOL_Loss, SubCLR_Loss, ContraWR_Loss
from utils.utils import score, filter_data
from torch import optim

class SSL_Trainer(BaseTrainer):
    def __init__(self, current_setting: str, cfg: dict, sub_ids: dict, to_opt: list, 
                 local_rank: int=0, device: str="cfg", DDP: bool=False):
        super().__init__(current_setting, cfg, sub_ids, to_opt, local_rank, device, DDP)

    def _construct_optimizer_scheduler(self, models: list): 
        """Sets self.optimizer and self.scheduler for models."""

        params_to_optimize = [] 
        for model, opt_flag in zip(models, self.to_opt):
            if opt_flag:
                #params_to_optimize.extend(self.define_param_groups(model, LARS=self.cfg["training"]["use_LARS"]))
                params_to_optimize.extend(model.parameters())

        lr = self.get_lr_warmup_cosinedecay(epoch=0)
        if self.cfg["training"]["use_LARS"]:
            self.optimizer = LARS(
                params_to_optimize,
                lr=lr,
                weight_decay=self.weight_decay,
                weight_decay_filter=exclude_bias_and_norm,
                lars_adaptation_filter=exclude_bias_and_norm
            )
        else:
            self.optimizer = torch.optim.Adam(
                params=params_to_optimize,
                lr=lr,
                weight_decay=self.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=self.patience)

    def define_param_groups(self, model, LARS=False):
        def exclude_from_wd_and_adaptation(name):
            # if 'bn' in name:
            #     return True
            # if LARS and 'bias' in name:
            #     return True
            return False

        param_groups = [
            {
                'params': [p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)],
                'weight_decay': self.weight_decay,
                'layer_adaptation': True,
            },
            {
                'params': [p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)],
                'weight_decay': 0.,
                'layer_adaptation': False,
            },
        ]
        return param_groups

    def _train_epoch(self, models: list, epoch: int) -> Tuple[List, List, float, list]:

        encoder = models[0]
        encoder.train()
        head = models[1]
        head.train()

        SubCLR = isinstance(self.loss_function, SubCLR_Loss)
        two_way = isinstance(self.loss_function, (BYOL_Loss, ContraWR_Loss))
        BYOL = isinstance(self.loss_function, BYOL_Loss)
                
        # determine input channels
        if self.cfg["training"]["inference_type"] == "channels":
            n_chan = 3 if self.cfg["model"]["convert_to_TF"] else 1
        elif self.cfg["training"]["inference_type"] == "epochs":
            n_chan = 0 if self.cfg["model"]["convert_to_TF"] else self.cfg["model"]["in_channels"] # TODO

        if two_way:
            encoder2 = models[2]
            head2 = models[3]
            encoder2.train()
            head2.train() 
            if BYOL:
                BYOL_mapping = models[4]
                BYOL_mapping.train()
        del models
        
        if self.DDP: # Ensure that epochs differ
            self.train_dataloader.sampler.set_epoch(epoch)

        losses_this_epoch, adv_losses_this_epoch = [], []
        
        for i, batch in enumerate(self.train_dataloader):

            self.optimizer.zero_grad()

            with torch.autocast("cuda", enabled=self.amp):

                if SubCLR:
                    x, labels = batch
                    x = x.view(-1, n_chan, self.n_time_samples).to(self.device)
                    z = head(encoder(x))
                    loss = self.loss_function(z, labels.to(self.device))

                elif isinstance(self.loss_function, ContraSub_Loss):
                    x, labels = batch
                    v1 = x[:,0,:].view(-1, 1, self.n_time_samples).to(self.device)
                    v2 = x[:,1,:].view(-1, 1, self.n_time_samples).to(self.device)

                    online = head(encoder(v1))
                    with torch.no_grad():
                        target = head(encoder(v2))
                    loss = self.loss_function(online, target, labels.to(self.device))    

                else: # Augmentation-based methods
                    if self.cfg["model"]["convert_to_TF"]: # B, n_chan, aug, T
                        v1 = batch[:,:,0,:].to(self.device)
                        v2 = batch[:,:,1,:].to(self.device)
                    else: # B, aug, T
                        v1 = batch[:,0,:].view(-1, n_chan, self.n_time_samples).to(self.device)
                        v2 = batch[:,1,:].view(-1, n_chan, self.n_time_samples).to(self.device)

                    if isinstance(self.loss_function, BYOL_Loss):
                        online_v1 = BYOL_mapping(head(encoder(v1)))
                        online_v2 = BYOL_mapping(head(encoder(v2)))
                        with torch.no_grad():
                            target_v1 = head2(encoder2(v1)).clone().detach()
                            target_v2 = head2(encoder2(v2)).clone().detach()
                        loss = self.loss_function(online_v1, target_v2) +  self.loss_function(online_v2, target_v1)

                    elif isinstance(self.loss_function, ContraWR_Loss):
                        online = head(encoder(v1))
                        with torch.no_grad():
                            target = head(encoder(v2))
                        loss = self.loss_function(online, target)     

                    else: # SimCLR and VICReg
                        z1 = head(encoder(v1))
                        z2 = head(encoder(v2))
                        loss = self.loss_function(z1, z2)
                            
            if self.amp: # use scaler
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            losses_this_epoch.append(loss.item())

        # exponential moving average
        if two_way:
            m = self.set_m(epoch) if BYOL else self.loss_function.m
            for enc_p_1, enc_p_2 in zip(encoder.parameters(), encoder2.parameters()):
                enc_p_2.data = enc_p_2.data * m + enc_p_1.data * (1. - m) 
            for head_p_1, head_p_2 in zip(head.parameters(), head2.parameters()):
                head_p_2.data = head_p_2.data * m + head_p_1.data * (1. - m) 
            models = [encoder, head, encoder2, head2]
            if BYOL:
                models.append(BYOL_mapping)
        else: 
            models = [encoder, head]

        if self.DDP: # Reduce across GPUs
            loss_tensor = torch.tensor(np.array(losses_this_epoch)).mean().to(self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / self.wsize
        else:
            avg_loss = np.mean(losses_this_epoch)

        return models, losses_this_epoch, avg_loss, adv_losses_this_epoch
    

    def train(self, models: list, setting):

        if setting == "SSL_PRE":
            self.train_SSL(models)
        else:
            self.train_SV(models)

    def forward(self, models: list, dataset, embedding_name: str):
        # Assumes we are using the EPOCH dataset.
        print(f"No existing embeddings detected. Generating them now as {embedding_name}")
        encoder = models[0]
        encoder = self._load_DDP_state_dict(encoder, self.cfg["model"]["pretrained_path"] + "/model_0_checkpoint.pt")
        dataloader = self.test_dataloader

        encoder.eval()

        prec = np.float16 if self.amp else np.float32 
        if self.cfg["training"]["inference_type"] == "channels":
            h_bank = np.empty((len(dataloader.dataset), self.in_channels, self.rep_dim), dtype=prec) 
            n_model_channels = 1
        elif self.cfg["training"]["inference_type"] == "epochs":
            h_bank = np.empty((len(dataloader.dataset), self.rep_dim), dtype=prec) 
            n_model_channels = self.in_channels
        #h_bank = np.empty((len(dataloader.dataset), self.rep_dim), dtype=prec) 
        #y_bank = np.empty((len(dataloader.dataset), 1), dtype=np.float32)

        total_samples = 0
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                with torch.autocast("cuda", enabled=self.amp):

                    if self.cfg["model"]["convert_to_TF"]:
                        h = encoder(filter_data(batch[0].view(-1, self.n_time_samples)).to(self.device))
                    else:
                        h = encoder(batch[0].view(-1, n_model_channels, self.n_time_samples).to(self.device)) 
                        #h = encoder(batch[0].view(-1, 1, self.in_channels, self.n_time_samples).to(self.device)) 
                    
                    if self.cfg["training"]["inference_type"] == "channels":
                        h = h.view(-1, self.in_channels, self.rep_dim)
                    #h = h.view(-1, self.rep_dim)
                    
                batch_samples = h.shape[0]
                h_bank[total_samples : total_samples + batch_samples] = h.cpu().numpy()
                total_samples += batch_samples

        # save as dataset
        if self.local_rank == 0:
            dataset_path = self.cfg["model"]["pretrained_path"] + embedding_name
            file = h5py.File(dataset_path, 'w')

            file.create_dataset('features', data=h_bank)
            for k, v in dataset.file.items():
                if k not in ["features", "dataset_mean", "dataset_std"]:
                    try:
                        file.create_dataset(k, data=v[dataset.test_epochs])
                    except:
                        file.create_dataset(k, data=v)

            file.close()

    def validate(self, models: list, test: bool=False) -> Tuple[float, float, List, List]:
        """
        Validates on using self.val_dataloader.
        Return validation loss and evaluation metric.
        """
        loss_total = 0.
        ys_true, ys_pred = [], []

        encoder = models[0]
        head = models[1]

        if test: # Load best model in test-mode
            encoder = self._load_DDP_state_dict(encoder, self.model_save_path + "/model_0_best.pt")
            head = self._load_DDP_state_dict(head, self.model_save_path + "/model_1_best.pt")
            dataloader = self.test_dataloader
            subject_ids = self.sub_ids["test"]
        else:
            dataloader = self.val_dataloader
            subject_ids = self.sub_ids["val"]

        encoder.eval()
        head.eval()

        with torch.no_grad():
            for batch in dataloader:

                x, y = batch[0].to(self.device), batch[1].to(self.device)

                with torch.autocast("cuda", enabled=self.amp):

                    z = encoder(x.view(-1, 1, self.n_time_samples))

                    out = head(z.view(-1, 1, self.in_channels, self.rep_dim)).view(-1, 1)

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

    def set_m(self, epoch):
        # Sets EMA coefficient for BYOL based on base_m (self.m)
        return 1 - (1 - self.m) * (math.cos(math.pi * epoch / self.num_epochs) + 1)/2
   

class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])

def exclude_bias_and_norm(p):
    return p.ndim == 1

