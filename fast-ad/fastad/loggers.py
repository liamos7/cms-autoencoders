import numpy as np

from .utils import AverageMeter


class BaseLogger:
    """BaseLogger that can handle most of the logging
    logging convention
    ------------------
    'loss' has to be exist in all training settings
    endswith('_') : scalar
    endswith('@') : image
    """
    def __init__(self, tb_writer):
        """tb_writer: tensorboard SummaryWriter"""
        self.writer = tb_writer
        self.train_loss_meter = AverageMeter()
        self.val_loss_meter = AverageMeter()
        self.d_train = {}
        self.d_val = {}
        self.has_val_loss = True  # If True, we assume that validation loss is available

    def process_iter_train(self, d_result):
        self.train_loss_meter.update(d_result['loss'])
        self.d_train = d_result

    def summary_train(self, i):
        self.d_train['loss/train_loss_'] = self.train_loss_meter.avg
        for key, val in self.d_train.items():
            if key.endswith('_'):
                self.writer.add_scalar(key, val, i)
            if key.endswith('@'):
                if val is not None:
                    self.writer.add_image(key, val, i)

        result = self.d_train
        self.d_train = {}
        self.train_loss_meter.reset()
        return result

    def process_iter_val(self, d_result):
        if self.has_val_loss:
            loss_val = d_result['loss']
            if loss_val == loss_val:
                self.val_loss_meter.update(loss_val)
            
        # Accumulate validation metrics across batches
        if not hasattr(self, '_val_accumulators'):
            self._val_accumulators = {}
            self._val_counts = {}

        for key, val in d_result.items():
            if key.endswith('_') and val is not None:
                scalar = float(val)
                if scalar != scalar:
                    continue
                if key not in self._val_accumulators:
                    self._val_accumulators[key] = 0.0
                    self._val_counts[key] = 0
                self._val_accumulators[key] += float(val)
                self._val_counts[key] += 1
            elif not key.endswith('_'):
                # Keep non-scalar values (like images) from the last batch
                self.d_val[key] = val

    def summary_val(self, i):
        print(f"summary_val: {i}")
        if self.has_val_loss:
            self.d_val['loss/val_loss_'] = self.val_loss_meter.avg
            
        # Add accumulated validation metrics
        if hasattr(self, '_val_accumulators'):
            for key, total in self._val_accumulators.items():
                count = self._val_counts[key]
                if count > 0:
                    self.d_val[key] = total / count
                    
        l_print_str = [f'Iter [{i:d}]']
        for key, val in self.d_val.items():
            if key.endswith('_'):
                self.writer.add_scalar(key, val, i)
                l_print_str.append(f'{key}: {val:.4f}')
            if key.endswith('@'):
                if val is not None:
                    self.writer.add_image(key, val, i)

        print_str = ' '.join(l_print_str)

        result = self.d_val
        result['print_str'] = print_str
        self.d_val = {}
        self.val_loss_meter.reset()
        
        # Reset accumulators for next validation
        if hasattr(self, '_val_accumulators'):
            self._val_accumulators.clear()
            self._val_counts.clear()
            
        return result


class NAELogger(BaseLogger):
    def __init__(self, tb_writer):
        super().__init__(tb_writer)
        self.train_loss_meter_nae = AverageMeter()
        self.val_loss_meter_nae = AverageMeter()
        self.d_train_nae = {}
        self.d_val_nae = {}

    def process_iter_train_nae(self, d_result):
        self.train_loss_meter.update(d_result['loss'])
        self.d_train_nae = d_result

    def summary_train_nae(self, i):
        d_result = self.d_train_nae
        writer = self.writer
        writer.add_scalar('nae/loss', d_result['loss'], i)
        writer.add_scalar('nae/energy_diff', d_result['pos_e'] - d_result['neg_e'], i)
        writer.add_scalar('nae/pos_e', d_result['pos_e'], i)
        writer.add_scalar('nae/neg_e', d_result['neg_e'], i)
        writer.add_scalar('nae/encoder_l2', d_result['encoder_norm'], i)
        writer.add_scalar('nae/decoder_l2', d_result['decoder_norm'], i)
        if 'neg_e_x0' in d_result:
            writer.add_scalar('nae/neg_e_x0', d_result['neg_e_x0'], i)
        if 'neg_e_z0' in d_result:
            writer.add_scalar('nae/neg_e_z0', d_result['neg_e_z0'], i)
        if 'temperature' in d_result:
            writer.add_scalar('nae/temperature', d_result['temperature'], i)
        if 'sigma' in d_result:
            writer.add_scalar('nae/sigma', d_result['sigma'], i)
        if 'delta_term' in d_result:
            writer.add_scalar('nae/delta_term', d_result['delta_term'], i)
        if 'gamma_term' in d_result:
            writer.add_scalar('nae/gamma_term', d_result['gamma_term'], i)


        '''images'''
        x_neg = d_result['x_neg']
        recon_neg = d_result['recon_neg']
        img_grid = make_grid(x_neg, nrow=10, range=(0, 1))
        writer.add_image('nae/sample', img_grid, i)
        img_grid = make_grid(recon_neg, nrow=10, range=(0, 1), normalize=True)
        writer.add_image('nae/sample_recon', img_grid, i)

        # to uint8 and save as array
        x_neg = (x_neg.permute(0,2,3,1).numpy() * 256.).clip(0, 255).astype('uint8')
        recon_neg = (recon_neg.permute(0,2,3,1).numpy() * 256.).clip(0, 255).astype('uint8')
        # save_image(img_grid, f'{writer.file_writer.get_logdir()}/nae_sample_{i}.png')
        np.save(f'{writer.file_writer.get_logdir()}/nae_neg_{i}.npy', x_neg)
        np.save(f'{writer.file_writer.get_logdir()}/nae_neg_recon_{i}.npy', recon_neg)


    def summary_val_nae(self, i, d_result):
        l_print_str = [f'Iter [{i:d}]']
        for key, val in d_result.items():
            self.writer.add_scalar(key, val, i)
            l_print_str.append(f'{key}: {val:.4f}')
        print_str = ' '.join(l_print_str)
        return print_str
