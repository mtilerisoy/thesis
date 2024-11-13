from .vg_caption_datamodule import VisualGenomeCaptionDataModule
from .f30k_caption_karpathy_datamodule import F30KCaptionKarpathyDataModule
from .coco_caption_karpathy_datamodule import CocoCaptionKarpathyDataModule
from .conceptual_caption_datamodule import ConceptualCaptionDataModule
from .sbu_datamodule import SBUCaptionDataModule
from .vqav2_datamodule import VQAv2DataModule
from .nlvr2_datamodule import NLVR2DataModule
from .snli_datamodule import SNLIDataModule

from .ood_itr_datamodule import OODITRDataModule
from .ood_vqav2_datamodule import OODVQAv2DataModule
from .ood_nlvr2_datamodule import OODNLVR2DataModule

# Mapping from dataset name to DataModule class
# This is used in config.py to instantiate the DataModule
_datamodules = {
    "vg": VisualGenomeCaptionDataModule,
    "f30k": F30KCaptionKarpathyDataModule,
    "coco": CocoCaptionKarpathyDataModule,
    "gcc": ConceptualCaptionDataModule,
    "sbu": SBUCaptionDataModule,
    "vqa": VQAv2DataModule,
    "nlvr2": NLVR2DataModule,
    "snli": SNLIDataModule,

    # Custom OOD DataModules
    "ood_itr": OODITRDataModule,
    "ood_vqa": OODVQAv2DataModule,
    "ood_nlvr2": OODNLVR2DataModule,
}
