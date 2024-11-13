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

# # Example usage
# write_coco_karpathy.make_arrow(root="/data-4/users/mileriso/datasets/COCO", dataset_root="/data-4/users/mileriso/datasets/COCO/arrows")
# write_f30k_karpathy.make_arrow(root="/data-4/users/mileriso/datasets/Flickr30k", dataset_root="/data-4/users/mileriso/datasets/Flickr30k/arrows")
# # write_conceptual_caption.make_arrow(root="../data/conceptual_caption", dataset_root="../data/conceptual_caption/arrows")
# # write_sbu.make_arrow(root="../data/SBU", dataset_root="../data/SBU/arrows")
# # write_vg.make_arrow(root="../data/VG", dataset_root="../data/VG/arrows")

# write_nlvr2.make_arrow(root="/data-4/users/mileriso/datasets/NLVR2", dataset_root="/data-4/users/mileriso/datasets/NLVR2/arrows")

# write_vqa.make_arrow(root="/data-4/users/mileriso/datasets/VQAv2", dataset_root="/data-4/users/mileriso/datasets/VQAv2/arrows")

# write_ood_itr.make_arrow(root="/data-4/users/mileriso/datasets/OOD/", json_file="/data-4/users/mileriso/datasets//OOD/itr_vlue_test.json", dataset_root="/data-4/users/mileriso/datasets/OOD/arrows")

# write_ood_vqa.make_arrow(root="/data-4/users/mileriso/datasets/OOD/", json_file="/data-4/users/mileriso/datasets//OOD/vqa_vlue_test.json", dataset_root="/data-4/users/mileriso/datasets/OOD/arrows")

# write_ood_nlvr2.make_arrow(root="/data-4/users/mileriso/datasets/OOD/", dataset_root="/data-4/users/mileriso/datasets/OOD/arrows")