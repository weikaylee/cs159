"""DataModuleFromConfig — PyTorch Lightning DataModule that instantiates
train / validation / test / predict datasets from an OmegaConf config dict.

Used by main.py:
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.set_generator(f, g)
    trainer.predict(model, data)
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from sgm.util import instantiate_from_config


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        train=None,
        validation=None,
        test=None,
        predict=None,
        num_workers: int = 4,
        wrap: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.wrap = wrap

        self._configs = {}
        if train      is not None: self._configs["train"]      = train
        if validation is not None: self._configs["validation"] = validation
        if test       is not None: self._configs["test"]       = test
        if predict    is not None: self._configs["predict"]    = predict

        self.datasets = {}
        self._generators = (None, None)

    # ── lifecycle ──────────────────────────────────────────────────────────

    def prepare_data(self):
        """Instantiate all configured datasets."""
        for split, cfg in self._configs.items():
            self.datasets[split] = instantiate_from_config(cfg)

    def set_generator(self, f, g):
        """Accept the two torch.Generators passed by main.py."""
        self._generators = (f, g)

    # ── dataloaders ────────────────────────────────────────────────────────

    def _loader(self, split: str, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            self.datasets[split],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            drop_last=shuffle,
            pin_memory=False,
        )

    def train_dataloader(self):
        return self._loader("train", shuffle=True)

    def val_dataloader(self):
        return self._loader("validation", shuffle=False)

    def test_dataloader(self):
        return self._loader("test", shuffle=False)

    def predict_dataloader(self):
        return self._loader("predict", shuffle=False)
