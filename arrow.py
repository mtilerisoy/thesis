from vilt.utils import write_coco_karpathy
from vilt.utils import write_conceptual_caption
from vilt.utils import write_f30k_karpathy
from vilt.utils import write_nlvr2
from vilt.utils import write_sbu
from vilt.utils import write_vg
from vilt.utils import write_vqa

from vilt.utils import write_ood_itr
from vilt.utils import write_ood_vqa
from vilt.utils import write_ood_nlvr2

# # Example usages:

# # In distribution datasets
# write_nlvr2.make_arrow(root="/data-4/users/mileriso/datasets/NLVR2", dataset_root="/data-4/users/mileriso/datasets/NLVR2/arrows")
# write_vqa.make_arrow(root="/data-4/users/mileriso/datasets/VQAv2", dataset_root="/data-4/users/mileriso/datasets/VQAv2/arrows")

# # OOD datasets
# write_ood_vqa.make_arrow(root="/data-4/users/mileriso/datasets/OOD/", json_file="/data-4/users/mileriso/datasets//OOD/vqa_vlue_test.json", dataset_root="/data-4/users/mileriso/datasets/OOD/arrows")
# write_ood_nlvr2.make_arrow(root="/data-4/users/mileriso/datasets/OOD/", dataset_root="/data-4/users/mileriso/datasets/OOD/arrows")