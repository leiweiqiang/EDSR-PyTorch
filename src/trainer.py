import os
import math
from decimal import Decimal

import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm

try:
    import quantization
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8
        
        # Handle Post-Training Quantization (PTQ)
        if self.args.test_only and getattr(self.args, 'quantize', '') == 'ptq' and QUANTIZATION_AVAILABLE:
            self.ckp.write_log('Performing Post-Training Quantization...')
            self._perform_ptq()

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        
        # For QAT, model should be in train mode
        # For regular training, also use train mode
        self.model.train()
        
        # If QAT is enabled, ensure model is prepared
        if getattr(self.args, 'quantize', '') == 'qat' and QUANTIZATION_AVAILABLE:
            if not self.model.is_qat_prepared:
                self.model.prepare_for_qat()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        
        # Convert QAT model to quantized model only for final/test-only eval
        should_convert = (
            getattr(self.args, 'quantize', '') == 'qat' and
            QUANTIZATION_AVAILABLE and
            self.model.is_qat_prepared and
            not self.model.is_quantized and
            (
                self.args.test_only or
                (epoch + 1) >= self.args.epochs
            )
        )
        if should_convert:
            self.ckp.write_log('Converting QAT model to quantized model...')
            # 确保量化后端被设置
            import torch.backends.quantized as quantized_backends
            backend = getattr(self.model, 'quantize_backend', 'fbgemm')
            quantized_backends.engine = backend
            self.ckp.write_log(f'Setting quantization backend to: {backend}')
            self.model.convert_to_quantized()
        
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        any_eval = False
        for idx_data, d in enumerate(self.loader_test):
            if len(d) == 0:
                self.ckp.write_log(
                    'Warning: empty test set for {} (check --data_range/dir_data)'.format(
                        d.dataset.name
                    )
                )
                self.ckp.write_log(
                    'Debug: dir_data={}, data_range={}, ext={}, data_test={}'.format(
                        self.args.dir_data,
                        self.args.data_range,
                        self.args.ext,
                        self.args.data_test
                    )
                )
                continue
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                any_eval = True
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            if any_eval:
                self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))
            else:
                self.ckp.write_log('Warning: no evaluation performed; skipping best-model check')
                self.ckp.save(self, epoch, is_best=False)

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        # 量化模型必须在 CPU 上运行
        if getattr(self.model, 'is_quantized', False):
            device = torch.device('cpu')
        elif self.args.cpu:
            device = torch.device('cpu')
        else:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs
    
    def _perform_ptq(self):
        """
        Perform Post-Training Quantization (PTQ).
        """
        if not QUANTIZATION_AVAILABLE:
            self.ckp.write_log('Warning: Quantization module not available. Skipping PTQ.')
            return
        
        calibration_samples = getattr(self.args, 'calibration_samples', 100)
        backend = getattr(self.args, 'quantize_backend', 'fbgemm')
        
        self.ckp.write_log('Calibrating model with {} samples...'.format(calibration_samples))
        
        # Move model to CPU for quantization (quantization typically works on CPU)
        original_device = next(self.model.model.parameters()).device
        self.model.model = self.model.model.cpu()
        
        # Use training loader for calibration if available, otherwise use test loader
        calibration_loader = self.loader_train if self.loader_train else self.loader_test
        
        # Perform quantization
        self.model.model = quantization.quantize_model(
            self.model.model,
            calibration_loader=calibration_loader,
            num_samples=calibration_samples,
            backend=backend
        )
        
        self.model.is_quantized = True
        self.ckp.write_log('Post-Training Quantization completed.')
        
        # Log model size comparison
        try:
            original_size = quantization.get_model_size(self.model.model, quantized=False)
            quantized_size = quantization.get_model_size(self.model.model, quantized=True)
            compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
            self.ckp.write_log('Model size: {:.2f} MB -> {:.2f} MB (compression: {:.2f}x)'.format(
                original_size, quantized_size, compression_ratio
            ))
        except Exception as e:
            self.ckp.write_log('Warning: Could not calculate model size: {}'.format(e))

