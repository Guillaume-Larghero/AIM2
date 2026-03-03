"""TensorBoard logging utilities."""

from torch.utils.tensorboard import SummaryWriter


class Logger:
    """TensorBoard logger with nested metric support."""

    def __init__(self, config):
        self.config = config
        if config.system.USE_TENSORBOARD:
            self.writer = SummaryWriter(log_dir=config.paths.LOG_DIR)
            print(f"TensorBoard logging to: {config.paths.LOG_DIR}")
        else:
            self.writer = None

    def log_metrics(self, metrics, step, prefix=''):
        """Log a (possibly nested) dict of scalar metrics."""
        if self.writer is None:
            return
        self._log_dict(metrics, step, prefix)

    def _log_dict(self, d, step, prefix=''):
        for key, value in d.items():
            if isinstance(value, dict):
                new_prefix = f"{prefix}/{key}" if prefix else key
                self._log_dict(value, step, new_prefix)
            elif isinstance(value, (int, float)):
                full_key = f"{prefix}/{key}" if prefix else key
                self.writer.add_scalar(full_key, value, step)

    def log_learning_rate(self, optimizer, step):
        """Log learning rates for each parameter group."""
        if self.writer is None:
            return
        for idx, pg in enumerate(optimizer.param_groups):
            name = pg.get('name', f'group_{idx}')
            self.writer.add_scalar(f'learning_rate/{name}', pg['lr'], step)

    def close(self):
        if self.writer is not None:
            self.writer.close()