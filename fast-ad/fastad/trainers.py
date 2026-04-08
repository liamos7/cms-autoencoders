import time
import os
import torch
import numpy as np
from tqdm import tqdm

from torch.optim import Adam

from .utils import AverageMeter
from .utils import get_roc_auc_from_scores


class BaseTrainer:
    """Trainer for a conventional iterative training of model"""
    def __init__(self, n_epochs, val_interval=1, save_interval=1, device='cpu'):
        self.n_epochs = n_epochs
        self.val_interval = val_interval
        self.save_interval = save_interval
        self.device = device
        self.d_val_result = {}

    def train(self, model, opt, d_dataloaders, logger=None, logdir='', scheduler=None, clip_grad=None):
        best_val_loss = np.inf
        train_loader, val_loader = d_dataloaders['training'], d_dataloaders['validation']
        total_steps = self.n_epochs * len(train_loader)
        progress_bar = tqdm(total=total_steps, desc="Training")

        i_step = 0
        for i_epoch in range(self.n_epochs):
            for x, y in train_loader:
                i_step += 1
                model.train()
                x, y = x.to(self.device), y.to(self.device)

                d_train = model.train_step(x, optimizer=opt, clip_grad=clip_grad)
                
                # Batch-level scheduler step for Warmup/Cosine
                if scheduler is not None:
                    scheduler.step()

                logger.process_iter_train(d_train)
                progress_bar.update(1)

                progress_bar.set_postfix({
                    'Epoch': f"{i_epoch+1}/{self.n_epochs}",
                    'Loss': f"{d_train['loss']:.4f}",
                    'Pos Energy': d_train.get('energy/pos_energy_', None),
                    'Neg Energy': d_train.get('energy/neg_energy_', None),
                })

                if i_step % self.val_interval == 0:
                    # Log training metrics to tensorboard
                    d_train_summary = logger.summary_train(i_step)
                    
                    model.eval()
                    for i, (val_x, val_y) in enumerate(val_loader):
                        val_x = val_x.to(self.device)
                        val_y = val_y.to(self.device)

                        d_val = model.validation_step(val_x, y=val_y, show_image=(i==0), calc_roc_auc=True)
                        logger.process_iter_val(d_val)
                    d_val = logger.summary_val(i_step)
                    val_loss = d_val['loss/val_loss_']

                    # Update progress bar with validation info instead of printing
                    progress_bar.set_postfix({
                        'Epoch': f"{i_epoch+1}/{self.n_epochs}",
                        'Train Loss': f"{d_train['loss']:.4f}",
                        'Val Loss': f"{val_loss:.4f}",
                        'AUROC': f"{d_val.get('roc_auc_', 0):.4f}"
                    })
                    best_model = val_loss < best_val_loss

                    if i_step % self.save_interval == 0 or best_model:
                        self.save_model(model, logdir, best=best_model, i_iter=i_step)
                    if best_model:
                        progress_bar.write(f'Iter [{i_step:d}] best model saved {val_loss:.4f} <= {best_val_loss:.4f}')
                        best_val_loss = val_loss
            if i_epoch % self.save_interval == 0:
                self.save_model(model, logdir, best=False, i_epoch=i_epoch)

        progress_bar.close()
        return model, best_val_loss


    def save_model(self, model, logdir, best=False, i_iter=None, i_epoch=None):
        if best:
            pkl_name = "model_best.pkl"
        else:
            if i_iter is not None:
                pkl_name = "model_iter_{}.pkl".format(i_iter)
            else:
                pkl_name = "model_epoch_{}.pkl".format(i_epoch)
        state = {"epoch": i_epoch, "model_state": model.state_dict(), 'iter': i_iter}
        save_path = os.path.join(logdir, pkl_name)
        torch.save(state, save_path)
        print(f'Model saved: {pkl_name}')
