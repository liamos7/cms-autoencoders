"""
ae.py
=====
Autoencoders
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision.utils import make_grid
from .modules import IsotropicGaussian, IsotropicLaplace
from ..utils import get_roc_auc_from_scores


class AE(nn.Module):
    """autoencoder"""
    def __init__(self, encoder, decoder):
        """
        encoder, decoder : neural networks
        """
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.own_optimizer = False

    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z)

    def encode(self, x):
        z = self.encoder(x)
        return z
    
    def decode(self, z):
        return self.decoder(z)
    
    def reconstruct(self, x):
        """Alias for forward pass"""
        return self(x)

    def predict(self, x):
        """one-class anomaly prediction"""
        recon = self(x)
        if hasattr(self.decoder, 'error'):
            predict = self.decoder.error(x, recon)
        else:
            predict = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)
        return predict

    def predict_and_reconstruct(self, x):
        recon = self(x)
        if hasattr(self.decoder, 'error'):
            recon_err = self.decoder.error(x, recon)
        else:
            recon_err = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)
        return recon_err, recon
    
    
    def train_step(self, x, optimizer, clip_grad=None, **kwargs):
        optimizer.zero_grad()
        recon_error = self.predict(x)
        loss = recon_error.mean()
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_grad)
        optimizer.step()
        return {'loss': loss.item()}


    def validation_step(self, x, y, **kwargs):
        predict, recon = self.predict_and_reconstruct(x)
        
        has_inliers = (y==0).any()
        has_holdout = (y==1).any()

        d = {
            "loss": predict[y==0].mean().item() if has_inliers else float('nan'),
            "loss/loss_holdout_": predict[y==1].mean().item() if has_holdout else float('nan'),
            "loss/recon_error_": ((recon - x) ** 2)[y==0].mean().item() if has_inliers else float('nan'),
            "loss/recon_error_holdout_": ((recon - x) ** 2)[y==1].mean().item() if has_holdout else float('nan'),
            "predict": predict,
            "reconstruction": recon,
        }
        if kwargs.get('show_image', True):
            d.update({
                "input/input@": make_grid(x[y==0].detach().cpu(), nrow=10, value_range=(0, 1)) if has_inliers else None,
                "recon/recon@": make_grid(recon[y==0].detach().cpu(), nrow=10, value_range=(0, 1)) if has_inliers else None,
                "input/input_holdout@": make_grid(x[y==1].detach().cpu(), nrow=10, value_range=(0, 1)) if has_holdout else None,
                "recon/recon_holdout@": make_grid(recon[y==1].detach().cpu(), nrow=10, value_range=(0, 1)) if has_holdout else None,
            })
        if kwargs.get('calc_roc_auc', False) and has_inliers and has_holdout:
            d["roc_auc_"] = get_roc_auc_from_scores(
                predict[y==1].detach().cpu().numpy(),
                predict[y==0].detach().cpu().numpy()
            )
        return d
    

class NAE(AE):
    """
    Normalized Autoencoder (NAE) - Energy-based autoencoder with optional spherical latent space.
    
    NAE computes energy as normalized L2 reconstruction error and can be trained in two phases:
    1. Standard autoencoder training (reconstruction loss)
    2. Energy-based training with contrastive divergence (optional)
    
    For basic usage in fastad, we implement a simplified version focused on the
    autoencoder phase, with energy-based training as an optional extension.
    """
    
    def __init__(self, encoder, decoder, spherical=False, temperature=1.0, 
                 temperature_trainable=False):
        """
        Args:
            encoder: Neural network that maps input to latent space
            decoder: Neural network that maps latent to reconstruction  
            spherical: If True, normalize latent codes to unit sphere
            temperature: Temperature parameter for energy scaling
            temperature_trainable: If True, temperature is a learnable parameter
        """
        # super(NAE, self).__init__()
        super(NAE, self).__init__(encoder, decoder)
        self.own_optimizer = False

        self.encoder = encoder
        self.decoder = decoder
        self.spherical = spherical
        self.own_optimizer = False
        
        # Temperature parameter (log-parameterized for stability)
        temperature_log = np.log(temperature)
        if temperature_trainable:
            self.register_parameter('temperature_log', 
                                  nn.Parameter(torch.tensor(temperature_log, dtype=torch.float)))
        else:
            self.register_buffer('temperature_log', torch.tensor(temperature_log, dtype=torch.float))
        
    @property
    def temperature(self):
        """Temperature (always positive due to log parameterization)"""
        return torch.exp(self.temperature_log)
    
    def normalize(self, z):
        """Normalize latent codes to unit sphere if spherical=True"""
        if self.spherical:
            return z / z.view(len(z), -1).norm(dim=1, keepdim=True)
        return z
    
    def encode(self, x):
        """Encode input to latent space with optional spherical normalization"""
        z = self.encoder(x)
        return self.normalize(z)
    
    def energy(self, x):
        """
        Compute energy (normalized L2 reconstruction error per dimension).
        Lower energy = better reconstruction = more normal data.
        """
        recon = self(x)
        D = torch.tensor(np.prod(x.shape[1:]), dtype=torch.float, device=x.device)  # Number of dimensions
        error = ((x - recon) ** 2).view(len(x), -1).sum(dim=1)
        return error / D
    
    def predict(self, x):
        """
        Anomaly prediction using energy.
        Higher energy = more anomalous.
        """
        return self.energy(x)
    
    def predict_and_reconstruct(self, x):
        """Return both anomaly scores and reconstructions"""
        recon = self(x)
        D = torch.tensor(np.prod(x.shape[1:]), dtype=torch.float, device=x.device)
        energy = ((x - recon) ** 2).view(len(x), -1).sum(dim=1) / D
        return energy, recon
    
    def train_step(self, x, optimizer, clip_grad=None, **kwargs):
        """
        Training step - standard autoencoder reconstruction loss.
        This is the Phase 1 training from the reference implementation.
        """
        optimizer.zero_grad()
        
        # Compute reconstruction loss (energy)
        energy = self.energy(x)
        loss = energy.mean()
        
        # Optional L2 regularization
        if hasattr(self, 'l2_reg_weight') and self.l2_reg_weight > 0:
            l2_loss = sum(p.pow(2.0).sum() for p in self.parameters())
            loss = loss + self.l2_reg_weight * l2_loss
        
        loss.backward()
        
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_grad)
            
        optimizer.step()
        
        return {'loss': loss.item()}
    

    def sample(self, N, z_shape=None, device='cpu'):
        """
        Generate samples by sampling latent codes and decoding.
        For NAE, this is a basic implementation - the full version would use MCMC.
        """
        if z_shape is None:
            # Infer latent shape from encoder
            dummy_input = torch.randn(1, *self.get_input_shape()).to(device)
            with torch.no_grad():
                dummy_z = self.encode(dummy_input)
            z_shape = dummy_z.shape[1:]
        
        # Sample random latent codes
        if self.spherical:
            # Sample from unit sphere
            z = torch.randn(N, *z_shape).to(device)
            z = z / z.view(N, -1).norm(dim=1, keepdim=True)
        else:
            # Sample from standard Gaussian
            z = torch.randn(N, *z_shape).to(device)
            
        # Decode to samples
        with torch.no_grad():
            samples = self.decode(z)
            
        return samples
    
    def get_input_shape(self):
        """
        Helper to infer input shape - this is a placeholder.
        In practice, should be set during initialization or first forward pass.
        """
        # Default MNIST shape - should be configurable
        return (1, 18, 14)


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NAEWithEnergyTraining(NAE):
    """
    Stabilized NAE for Energy-Based Training.
    Inherits from NAE.
    """

    def __init__(
        self,
        encoder,
        decoder,
        spherical=False,
        temperature=1.0,
        temperature_trainable=False,
        gamma=1e-2,           # Weight for energy^2 regularization
        neg_lambda=1.0,      # Weight on negative energy term
        l2_weight=1e-8,      # L2 weight regularization
        z_step_size=0.005,     # Slightly reduced for stability
        z_noise_std=0.005,
        z_steps=30,
        z_use_metropolis=False,
        x_step_size=0.001,     # Reduced from 0.05 to prevent explosion
        x_noise_std=0.0005,
        x_steps=10,
        x_use_annealing=True,
        buffer_size=10000,
        buffer_prob=0.95,
        buffer_reinit_prob=0.05,
        latent_dim=20,       # Matches Teacher linear output
    ):
        super().__init__(encoder, decoder, spherical, temperature, temperature_trainable)
        
        self.gamma = gamma
        self.neg_lambda = neg_lambda
        self.l2_weight = l2_weight

        # Langevin params
        self.z_step_size = z_step_size
        self.z_noise_std = z_noise_std
        self.z_steps = z_steps
        self.z_use_metropolis = z_use_metropolis
        self.x_step_size = x_step_size
        self.x_noise_std = x_noise_std
        self.x_steps = x_steps
        self.x_use_annealing = x_use_annealing

        # Replay buffer
        self.buffer_size = buffer_size
        self.buffer_prob = buffer_prob
        self.buffer_reinit_prob = buffer_reinit_prob
        self.latent_dim = latent_dim
        self._replay_buffer = None
        self._buffer_ptr = 0

    def seed_buffer(self, loader, device, limit=None):
        """
        Seeds the buffer with uniform sphere samples to provide contrastive signal
        away from the data manifold.
        """
        print(f"[NAE] Seeding replay buffer with uniform sphere samples...")
        z = torch.randn(self.buffer_size, self.latent_dim)
        z = F.normalize(z, dim=-1)
        self._replay_buffer = z.detach()
        print(f"[NAE] Buffer seeded with {len(self._replay_buffer)} samples.")

    def _sample_latent_init(self, batch_size, device):
        if self._replay_buffer is None:
            # Fallback if seed_buffer wasn't called
            z = torch.randn(self.buffer_size, self.latent_dim)
            if self.spherical: z = F.normalize(z, dim=-1)
            self._replay_buffer = z

        idx = torch.randint(0, self.buffer_size, (batch_size,))
        buffer_samples = self._replay_buffer[idx].to(device)

        fresh = torch.randn(batch_size, self.latent_dim, device=device)
        if self.spherical:
            fresh = F.normalize(fresh, dim=-1)

        # Reinitialize some buffer samples to fresh noise to prevent mode collapse
        reinit_mask = (torch.rand(batch_size, device=device) < self.buffer_reinit_prob).unsqueeze(-1)
        buffer_samples = torch.where(reinit_mask, fresh, buffer_samples)

        use_buffer = (torch.rand(batch_size, device=device) < self.buffer_prob).unsqueeze(-1)
        return torch.where(use_buffer, buffer_samples, fresh).detach()

    def _update_buffer(self, z_final):
        z_final = z_final.detach().cpu()
        batch_size = len(z_final)
        end = self._buffer_ptr + batch_size
        if end <= self.buffer_size:
            self._replay_buffer[self._buffer_ptr:end] = z_final
        else:
            first = self.buffer_size - self._buffer_ptr
            self._replay_buffer[self._buffer_ptr:] = z_final[:first]
            self._replay_buffer[:end - self.buffer_size] = z_final[first:]
        self._buffer_ptr = end % self.buffer_size

    def train_step(self, x, optimizer, clip_grad=None, **kwargs):
        optimizer.zero_grad()
        
        # 1. Energy Computation
        pos_energy = self.energy(x)
        x_neg = self.langevin_sample(x)
        neg_energy = self.energy(x_neg)

        # 2. Loss Formulation
        # Contrastive term: minimize pos, maximize neg
        cd_loss = pos_energy.mean() - self.neg_lambda * neg_energy.mean()

        # Stabilizer: Penalize squared energy of BOTH to prevent ballooning
        # Loss = (E_pos - E_neg) + gamma * (E_pos^2 + E_neg^2)
        reg_loss = self.gamma * (pos_energy.pow(2).mean() + neg_energy.pow(2).mean())

        # Weight decay
        l2_loss = sum(p.pow(2.0).sum() for p in self.parameters()) * self.l2_weight

        loss = (cd_loss + reg_loss) / self.temperature + l2_loss
        
        if torch.isnan(loss):
            return {'loss': 0.0, 'warning': 'NaN detected'}

        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)
        optimizer.step()

        return {
            'loss': loss.item(),
            'energy/pos_energy_': pos_energy.mean().item(),
            'energy/neg_energy_': neg_energy.mean().item(),
            'energy/diff_': (pos_energy.mean() - neg_energy.mean()).item()
        }

    def langevin_sample(self, x_init):
        """Two-stage OMI sampling."""
        batch_size = x_init.shape[0]
        device = x_init.device

        # Stage 1: Latent Chain
        z = self._sample_latent_init(batch_size, device).requires_grad_(True)
        for _ in range(self.z_steps):
            e = self.energy(self.decode(z)).sum()
            grad = torch.autograd.grad(e, z)[0]
            with torch.no_grad():
                grad_norm = grad.view(batch_size, -1).norm(dim=1, keepdim=True) + 1e-8
                grad = grad / grad_norm
                z = z - self.z_step_size * grad + torch.randn_like(z) * self.z_noise_std
                if self.spherical: z = F.normalize(z, dim=-1)
            z = z.detach().requires_grad_(True)
        
        self._update_buffer(z)

        # Stage 2: Data Chain
        x = self.decode(z).detach().requires_grad_(True)
        for step in range(self.x_steps):
            e = self.energy(x).sum()
            grad = torch.autograd.grad(e, x)[0]
            with torch.no_grad():
                step_size = self.x_step_size
                if self.x_use_annealing:
                    step_size *= (1.0 - (step / self.x_steps) * 0.5)
                x = x - step_size * grad + torch.randn_like(x) * self.x_noise_std
            x = x.detach().requires_grad_(True)
            
        return x.detach()

    def validation_step(self, x, y, **kwargs):
        """Use base NAE validation (reconstruction-based ROC-AUC)."""
        return super().validation_step(x, y, **kwargs)

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def load_pretrained_nae(self, model_path):
        if not model_path:
            print("No pretrained NAE provided, training from scratch")
            return

        print(f"Loading pretrained NAE from {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        pretrained_state = checkpoint.get('model_state', checkpoint)

        missing_keys, unexpected_keys = self.load_state_dict(pretrained_state, strict=False)

        expected_missing = {
            'x_noise_std', 'x_steps', 'x_step_size',
            'z_noise_std', 'z_steps', 'z_step_size',
            'neg_lambda', 'gamma', 'l2_weight', 'l2_reg_weight',
            'use_two_stage'
        }
        critical_missing = [k for k in missing_keys
                            if not any(e in k for e in expected_missing)]

        if critical_missing:
            print(f"WARNING: Critical keys missing from checkpoint: {critical_missing}")
        else:
            print("Successfully loaded pretrained NAE weights.")

        # Reset replay buffer (it belongs to the new training run)
        self._replay_buffer = None
        self._buffer_ptr = 0
    

class VAE(AE):
    def __init__(self, encoder, decoder, n_sample=1, use_mean=False, pred_method='recon', sigma_trainable=False):
        super(VAE, self).__init__(encoder, IsotropicGaussian(decoder, sigma=1, sigma_trainable=sigma_trainable))
        self.n_sample = n_sample  # the number of samples to generate for anomaly detection
        self.use_mean = use_mean  # if True, does not sample from posterior distribution
        self.pred_method = pred_method  # which anomaly score to use
        self.z_shape = None
        
    def forward(self, x):
        z = self.encoder(x)
        z_sample = self.sample_latent(z)
        return self.decoder(z_sample)

    def sample_latent(self, z):
        half_chan = int(z.shape[1] / 2)
        mu, log_sig = z[:, :half_chan], z[:, half_chan:]
        if self.use_mean:
            return mu
        eps = torch.randn(*mu.shape, dtype=torch.float32)
        eps = eps.to(z.device)
        return mu + torch.exp(log_sig) * eps

    # def sample_marginal_latent(self, z_shape):
    #     return torch.randn(z_shape)

    def kl_loss(self, z):
        """analytic (positive) KL divergence between gaussians
        KL(q(z|x) | p(z))"""
        half_chan = int(z.shape[1] / 2)
        mu, log_sig = z[:, :half_chan], z[:, half_chan:]
        mu_sq = mu ** 2
        sig_sq = torch.exp(log_sig) ** 2
        kl = mu_sq + sig_sq - torch.log(sig_sq) - 1
        # return 0.5 * torch.mean(kl.view(len(kl), -1), dim=1)
        return 0.5 * torch.sum(kl.view(len(kl), -1), dim=1)

    def train_step(self, x, optimizer, clip_grad=None, **kwargs):
        optimizer.zero_grad()
        z = self.encoder(x)
        z_sample = self.sample_latent(z)
        nll = - self.decoder.log_likelihood(x, z_sample)
        kl_loss = self.kl_loss(z)
        loss = nll + kl_loss
        loss = loss.mean()
        nll = nll.mean()

        loss.backward()
        optimizer.step()
        return {'loss': nll.item(), 'vae/kl_loss_': kl_loss.mean(), 'vae/sigma_': self.decoder.sigma.item()}

    def predict(self, x):
        """one-class anomaly prediction using the metric specified by self.anomaly_score"""
        if self.pred_method == 'recon':
            return self.reconstruction_probability(x)
        elif self.pred_method == 'lik':
            return  - self.marginal_likelihood(x)  # negative log likelihood
        else:
            raise ValueError(f'{self.pred_method} should be recon or lik')

    def predict_and_reconstruct(self, x):
        recon = self(x)
        pred = self.predict(x)
        return pred, recon

    # def validation_step(self, x, y, **kwargs):
    #     z = self.encoder(x)
    #     z_sample = self.sample_latent(z)
    #     recon = self.decoder(z_sample)
        
    #     loss = ((recon - x) ** 2)[y==0].mean()
    #     loss_holdout = ((recon - x) ** 2)[y==1].mean()
    #     predict = - self.decoder.log_likelihood(x, z_sample)
        
    #     if kwargs.get('show_image', True):
    #         x_img = make_grid(x[y==0].detach().cpu(), nrow=10, value_range=(0, 1))
    #         recon_img = make_grid(recon[y==0].detach().cpu(), nrow=10, value_range=(0, 1))
    #         x_img_holdout = make_grid(x[y==1].detach().cpu(), nrow=10, value_range=(0, 1))
    #         recon_img_holdout = make_grid(recon[y==1].detach().cpu(), nrow=10, value_range=(0, 1))
    #     else:
    #         x_img, recon_img, x_img_holdout, recon_img_holdout = None, None, None, None
    #     if kwargs.get('calc_roc_auc', False):
    #         roc_auc = get_roc_auc_from_scores(
    #             predict[y==1].detach().cpu().numpy(),
    #             predict[y==0].detach().cpu().numpy()
    #         )
    #     else:
    #         roc_auc = None
    #     return {
    #         'loss': loss.item(), 'predict': predict, 'reconstruction': recon, 'roc_auc_': roc_auc, 'loss/loss_holdout_': loss_holdout.item(),
    #         'input/input@': x_img, 'reov/recon@': recon_img, 'input/input_holdout@': x_img_holdout, 'recon/recon_holdout@': recon_img_holdout,
    #     }


    def reconstruction_probability(self, x):
        l_score = []
        z = self.encoder(x)
        for i in range(self.n_sample):
            z_sample = self.sample_latent(z)
            recon_loss = - self.decoder.log_likelihood(x, z_sample)
            l_score.append(recon_loss)
        return torch.stack(l_score).mean(dim=0)

    def marginal_likelihood(self, x, n_sample=None):
        """marginal likelihood from importance sampling
        log P(X) = log int P(X|Z) * P(Z)/Q(Z|X) * Q(Z|X) dZ"""
        if n_sample is None:
            n_sample = self.n_sample

        # check z shape
        with torch.no_grad():
            z = self.encoder(x)

            l_score = []
            for i in range(n_sample):
                z_sample = self.sample_latent(z)
                log_recon = self.decoder.log_likelihood(x, z_sample)
                log_prior = self.log_prior(z_sample)
                log_posterior = self.log_posterior(z, z_sample)
                l_score.append(log_recon + log_prior - log_posterior)
        score = torch.stack(l_score)
        logN = torch.log(torch.tensor(n_sample, dtype=torch.float, device=x.device))
        return torch.logsumexp(score, dim=0) - logN

    def marginal_likelihood_naive(self, x, n_sample=None):
        if n_sample is None:
            n_sample = self.n_sample

        # check z shape
        z_dummy = self.encoder(x[[0]])
        z = torch.zeros(len(x), *list(z_dummy.shape[1:]), dtype=torch.float).to(x.device)

        l_score = []
        for i in range(n_sample):
            z_sample = self.sample_latent(z)
            recon_loss = - self.decoder.log_likelihood(x, z_sample)
            score = recon_loss
            l_score.append(score)
        score = torch.stack(l_score)
        return - torch.logsumexp(-score, dim=0)

    def elbo(self, x):
        l_score = []
        z = self.encoder(x)
        for i in range(self.n_sample):
            z_sample = self.sample_latent(z)
            recon_loss = - self.decoder.log_likelihood(x, z_sample)
            kl_loss = self.kl_loss(z)
            score = recon_loss + kl_loss
            l_score.append(score)
        return torch.stack(l_score).mean(dim=0)

    def log_posterior(self, z, z_sample):
        half_chan = int(z.shape[1] / 2)
        mu, log_sig = z[:, :half_chan], z[:, half_chan:]

        log_p = torch.distributions.Normal(mu, torch.exp(log_sig)).log_prob(z_sample)
        log_p = log_p.view(len(z), -1).sum(-1)
        return log_p

    def log_prior(self, z_sample):
        log_p = torch.distributions.Normal(torch.zeros_like(z_sample), torch.ones_like(z_sample)).log_prob(z_sample)
        log_p = log_p.view(len(z_sample), -1).sum(-1)
        return log_p

    def posterior_entropy(self, z):
        half_chan = int(z.shape[1] / 2)
        mu, log_sig = z[:, :half_chan], z[:, half_chan:]
        D = mu.shape[1]
        pi = torch.tensor(np.pi, dtype=torch.float32).to(z.device)
        term1 = D / 2
        term2 = D / 2 * torch.log(2 * pi)
        term3 = log_sig.view(len(log_sig), -1).sum(dim=-1)
        return term1 + term2 + term3

    def _set_z_shape(self, x):
        if self.z_shape is not None:
            return
        with torch.no_grad():
            dummy_z = self.encode(x[[0]])
        dummy_z = self.sample_latent(dummy_z)
        z_shape = dummy_z.shape
        self.z_shape = z_shape[1:]

    def sample_z(self, n_sample, device):
        z_shape = (n_sample,) + self.z_shape
        return torch.randn(z_shape, device=device, dtype=torch.float)

    def sample(self, n_sample, device):
        z = self.sample_z(n_sample, device)
        return {'sample_x': self.decoder.sample(z)}





import torch
from torch import nn


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        """
        Teacher computational complexity: 18.91 MMac
        Teacher number of parameters: .36 k
        """
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=1, padding="valid"),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(8, 16, 3, stride=1, padding="valid"),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(16, 16, 3, stride=1, padding="valid"),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(16, 16, 3, stride=2, padding="valid"),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(16, 8, 3, stride=2, padding="valid"),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Flatten(start_dim=1),
            nn.Linear(128, 20),
        )

        self.decoder = nn.Sequential(
            nn.Linear(20, 8 * 4 * 4),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Unflatten(dim=1, unflattened_size=(8, 4, 4)),
            nn.ConvTranspose2d(
                8, 16, 3, stride=2, padding=1, output_padding=1
            ),
            nn.LeakyReLU(negative_slope=0.3),
            nn.ConvTranspose2d(
                16, 16, 3, stride=2, padding=1, output_padding=1
            ),
            nn.LeakyReLU(negative_slope=0.3),
            nn.ConvTranspose2d(
                16, 16, 3, stride=2, padding=1, output_padding=1
            ),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(16, 8, 3, stride=1, padding="valid"),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(8, 1, 3, stride=1, padding="valid"),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x
