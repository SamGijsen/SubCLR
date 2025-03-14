import os
import numpy as np
import yaml
import torch
import torch.nn as nn
import time
import joblib
import math

from typing import Tuple, List
from abc import abstractmethod
from models.loss import VICReg_Loss, SimCLR_Loss, BYOL_Loss, SubCLR_Loss, ContraWR_Loss

class BaseTrainer:
    """
    Base class for all trainers.
    """
    def __init__(self, current_setting: str, cfg: dict, sub_ids: dict, to_opt: list, 
                 local_rank: int=0, device: str="cfg", DDP : bool=False):
        
        self.current_setting = current_setting
        self.cfg = cfg
        self.local_rank = local_rank
        self.DDP = DDP
        if device == "cfg":
            self.device = cfg["training"]["device"]
        else:
            self.device = device
        self.sub_ids = sub_ids
        self.to_opt = to_opt

        # Fetch model-specific hyperparameters
        self.model_name = cfg["model"]["model_name"]
        self.model_type = cfg["model"]["type"]
        self.in_channels = cfg["model"]["in_channels"]
        self.n_time_samples = cfg["model"]["n_time_samples"]
        self.n_classes = cfg["model"]["n_classes"]
        self.rep_dim = cfg["model"]["rep_dim"]

        # Fetch training-specific hyperparameters
        self.target = cfg["training"]["target"]
        self.amp = cfg["training"]["amp"]
        self.batch_size = cfg["training"]["batch_size"]
        self.num_epochs = cfg["training"]["num_epochs"]
        self.lr = cfg["training"]["learning_rate"]
        self.T = cfg["training"]["T"]
        self.m = cfg["training"]["m"]
        self.n_augs = cfg["training"]["n_augmentations"]
        self.loss_function = cfg["training"]["loss_function"]
        self._fetch_loss_function()
        self.weight_decay = cfg["training"]["weight_decay"]
        self.patience = cfg["training"]["patience"]
        self.warmup_epochs = cfg["training"]["warmup_epochs"]
        self.model_save_path = cfg["training"]["model_save_path"]
        self.wsize = cfg["training"]["world_size"]

        # cross-validation 
        self.fold = cfg["training"]["fold"]
        self.n_train = cfg["training"]["n_train"]
        self.hp_key = "".join([f"{key.replace(cfg['model']['model_name']+'__', '_')}_{value}" for key, value in cfg["training"]["hp_key"].items()])

        # Fetch dataset-specific hyperparameters
        self.sfreq = cfg["dataset"]["sfreq"]
        self.dataset_name = cfg["dataset"]["name"]
        self.out_file = f"fold_{self.fold}_ntrain_{self.n_train}_{self.model_name}_{self.hp_key}" 
        self.model_save_path = f"{self.model_save_path}/{self.model_name}/{self.current_setting}/{self.out_file}"

        # check if model_save_path exists
        if local_rank==0 and not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path, exist_ok=True)

        # use scaler if we're training in FP16
        self.scaler = torch.cuda.amp.GradScaler() if self.amp else None

        if current_setting in ["SV", "SSL_FT", "SSL_PRE"]:
            self.input_size = self.n_time_samples
        else:
            self.input_size = self.rep_dim

    @abstractmethod
    def _train_epoch(self, models: list, epoch: int) -> Tuple[List, float]:
        """
        Training logic for an epoch

        Returns list of models and training loss.
        """
        raise NotImplementedError
    
    @abstractmethod
    def _construct_optimizer_scheduler(self, models: list): 
        """
        Constructs self.optimizer and self.scheduler.
        """
        raise NotImplementedError
    
    @abstractmethod
    def validate(self, models: list, test: bool) -> Tuple[float, float, List, List]:
        """
        Validation scoring. Needs to return val loss and metric.
        """
        raise NotImplementedError

    def _filter_bn_params(self, module):
        """
        Filter function used to exclude batch normalization parameters from weight decay.
        """
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            return False 
        return True

    def set_dataloaders(self, dataloaders: list):
        """
        Assigns from a list containing [train_dl, val_dl, test_dl].
        """
        self.train_dataloader = dataloaders[0]
        self.val_dataloader = dataloaders[1]
        self.test_dataloader = dataloaders[2]

    def _load_DDP_state_dict(self, model, path):
        state_dict = torch.load(path, self.device)
        
        if self.DDP:
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = "module." + key
                new_state_dict[new_key] = value
            state_dict = new_state_dict

        model.load_state_dict(state_dict)

        return model

    def _save_models(self, models: list, model_suffix: str):
        """
        Save state dict(s) of model(s) while checking for DDP.
        """
        for index, model in enumerate(models):
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model = model.module

            name = f"/model_{index}_{model_suffix}.pt"

            try:
                torch.save(model.state_dict(), 
                        self.model_save_path + name)
            except:
                pass

    def _save(self, models: list, lin_probe: list, ep_loss: list, train_loss: list, val_loss: list, 
              val_metrics: list, lrs: list, y_true: list, y_pred: list):
        """
        Save state dict(s), losses, learning rates, and true & predicted labels.
        """
        self._save_models(models, model_suffix="checkpoint")

        to_save = dict( # save losses etc.
            ep_loss=ep_loss,
            train_loss=train_loss,
            val_loss=val_loss,
            val_metrics=val_metrics,
            lin_probe=lin_probe,
            lrs=lrs,
            y_true=y_true,
            y_pred=y_pred
            )
        np.save(self.model_save_path + "/losses.npy", to_save)

        # Save config as .yaml
        with open(self.model_save_path + "/config_" + self.model_name + ".yaml", "w") as file:
            yaml.dump(self.cfg, file)

    def _fetch_loss_function(self):
    
        if self.loss_function == "L1Loss":
            self.loss_function = nn.L1Loss()
        elif self.loss_function == "MSELoss":
            self.loss_function = nn.MSELoss()
        elif self.loss_function == "BCELoss":
            self.loss_function = nn.BCELoss()
        elif self.loss_function == "BCEWithLogitsLoss":
            self.loss_function = nn.BCEWithLogitsLoss()
        elif self.loss_function == "NLLLoss":
            self.loss_function = nn.NLLLoss()
        elif self.loss_function == "VICReg":
            self.loss_function = VICReg_Loss(self.device, self.DDP)
        elif self.loss_function == "SimCLR":
            self.loss_function = SimCLR_Loss(self.device, self.DDP, temp=self.T)
        elif self.loss_function == "BYOL":
            self.loss_function = BYOL_Loss(self.device, self.batch_size, self.DDP)
        elif self.loss_function == "SubCLR":
            self.loss_function = SubCLR_Loss(self.device, self.DDP, temp=self.T)
        elif self.loss_function == "ContraWR":
            self.loss_function = ContraWR_Loss(self.device, self.batch_size, self.DDP, temp=self.T)   
        else:
            raise ValueError("Loss function not implemented")
        
    def train(self, models: list, setting: str):
        if setting == "SSL_NL":
            self.input_size = self.rep_dim
        else:
            self.input_size = self.n_time_samples

        if setting == "SSL_PRE":
            self.train_SSL(models)
        else:
            self.train_SV(models)

    def train_SV(self, models: list):

        self._construct_optimizer_scheduler(models)

        # initialize required variables
        self.best_val_loss, self.best_val_met = float('inf'), float('inf')
        val_losses, val_metrics, train_losses, ep_losses, lrs, y_true, y_pred = [], [], [], [], [], [], []
        lr_reduced = 0

        for epoch in range(self.num_epochs):
            start_t = time.monotonic() if self.local_rank == 0 else None

            models, epoch_losses, avg_loss, _ = self._train_epoch(models, epoch)
            
            val_loss, val_met, val_yt, val_yp = self.validate(models, test=False)
            self.scheduler.step(val_loss)

            # track change in learning rate
            lrs.append(self.optimizer.param_groups[0]['lr'])
            if epoch > 1 and lrs[-2] != lrs[-1]:
                lr_reduced += 1

            if self.local_rank == 0:
                ep_losses.append(epoch_losses)
                train_losses.append(avg_loss)
                val_losses.append(val_loss)
                val_metrics.append(val_met[1])
                y_true.append(val_yt)
                y_pred.append(val_yp)
                
                # check progress 
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_val_met = val_met[1]
                    self._save_models(models, model_suffix="best")

                # Save progress & print update
                print(f"ep {epoch:03d} | Tloss: {avg_loss:.3f} | Vloss: {val_loss:.3f} "
                      f"| Vmet: {val_met[1]:.4f} | lr: {self.optimizer.param_groups[0]['lr']:.4f} "
                      f"| n bad: {self.scheduler.num_bad_epochs} | t {time.monotonic()-start_t:.0f}s") #
                self._save(models, [], ep_losses, train_losses, val_losses, val_metrics, lrs, y_true, y_pred)

            if lr_reduced > 1:
                break

    def train_SSL(self, models: list):

        if self.cfg["training"]["use_LARS"]:
            self.lr *= ((self.wsize*self.batch_size)/256)
        self._construct_optimizer_scheduler(models)

        # initialize required variables
        ep_losses, train_losses, adv_losses, lrs = [], [], [], []
        self.best_val_loss, self.best_train_loss = float('inf'), float('inf')
                
        for epoch in range(self.num_epochs):

            start_t = time.monotonic() if self.local_rank == 0 else None

            # set and track lr
            lr = self.get_lr_warmup_cosinedecay(epoch) if self.warmup_epochs else self.lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            lrs.append(self.optimizer.param_groups[0]['lr'])

            models, epoch_losses, avg_loss, adv_loss = self._train_epoch(models, epoch)    

            if self.local_rank == 0:
                ep_losses.append(epoch_losses)
                train_losses.append(avg_loss)
                adv_losses.append(adv_loss)
                
                # check progress 
                if avg_loss < self.best_train_loss:
                    self.best_train_loss = avg_loss
                    self.best_val_loss = avg_loss # placeholders
                    self.best_val_met = avg_loss
                    self._save_models(models, model_suffix="best")

                # Save progress & print update
                print(f"ep {epoch:03d} | Tloss: {avg_loss:.3f} | lr: {self.optimizer.param_groups[0]['lr']:.4f} "
                      f"| t {time.monotonic()-start_t:.0f}s") #
                self._save(models, adv_losses, ep_losses, train_losses, [], [], lrs, [], [])

                # if epoch % 25 == 0 and self.num_epochs > 20:
                #     self._save_models(models, model_suffix=f"epoch{epoch}")
        
    def get_lr_warmup_cosinedecay(self, epoch):
        min_lr = 1e-2 * self.lr 
        if epoch < self.warmup_epochs:
            return self.lr * epoch / self.warmup_epochs + min_lr 
        decay_ratio = (epoch - self.warmup_epochs) / (self.num_epochs - self.warmup_epochs)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (self.lr - min_lr)
