from ..datasets import OODNLVR2Dataset
from .datamodule_base import BaseDataModule


class OODNLVR2DataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return OODNLVR2Dataset

    @property
    def dataset_name(self):
        return "ood_nlvr2"
