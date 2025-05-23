# Dataset Preparation
We utilize two in-domain datsets: Visual Question Answering v2 (VQAv2), and Natural Language for Visual Reasoning 2 (NLVR2), and one out-of-distribution dataset: VLUE benchmark.

We do not distribute datasets because of the license issue.
Please download the datasets by yourself.
The codebase uses `pyarrow` to serialize the datasets, conversion scripts are located in `vilt/utils/write_*.py`.
Please organize the datasets as described below and run `make_arrow` functions to convert the dataset to pyarrow binary file.

## VQAv2
https://visualqa.org/download.html

Download COCO [2014 train images](http://images.cocodataset.org/zips/train2014.zip), [2014 val images](http://images.cocodataset.org/zips/val2014.zip), [2015 test images](http://images.cocodataset.org/zips/test2015.zip), annotations ([train](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip), [val](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip)), and questions ([train](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip), [val](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip), [test](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip))

    root
    ├── train2014            
    │   ├── COCO_train2014_000000000009.jpg                
    |   └── ...
    ├── val2014              
    |   ├── COCO_val2014_000000000042.jpg
    |   └── ...  
    ├── test2015              
    |   ├── COCO_test2015_000000000001.jpg
    |   └── ...         
    ├── v2_OpenEnded_mscoco_train2014_questions.json
    ├── v2_OpenEnded_mscoco_val2014_questions.json
    ├── v2_OpenEnded_mscoco_test2015_questions.json
    ├── v2_OpenEnded_mscoco_test-dev2015_questions.json
    ├── v2_mscoco_train2014_annotations.json
    └── v2_mscoco_val2014_annotations.json

```python
from vilt.utils.write_vqa import make_arrow
make_arrow(dataset_root_path, dataset_root_path/arrows_root)
```

## NLVR2
Clone the [repository](https://github.com/lil-lab/nlvr) and sign the [request form](https://goo.gl/forms/yS29stWnFWzrDBFH3) to download the images.

    root
    ├── images/train           
    │   ├── 0                  
    │   │   ├── train-10108-0-img0.png   
    │   │   └── ...
    │   ├── 1                  
    │   │   ├── train-10056-0-img0.png       
    │   │   └── ...
    │   └── ...
    ├── dev       
    │   ├── dev-0-0-img0.png
    |   └── ...
    ├── test1     
    │   ├── test1-0-0-img0.png
    |   └── ...
    ├── nlvr
    ├── nlvr2
    └── README.md

```python
from vilt.utils.write_nlvr2 import make_arrow
make_arrow(dataset_root_path, dataset_root_path/arrows_root)
```

## VLUE Benchmark
The [VLUE benchmark](https://vlue-benchmark.github.io/leaderboard.html) uses images from the [MaRVL Dataset](https://marvl-challenge.github.io). The images can be accessed by following this [link](https://marvl-challenge.github.io/download) and following the procedure on the website. The annotations (*.json files) can be found within the ```data/``` folder under the root of this repository and should be copied to the root of the VLUE dataset folder as descirbed in the folder structure.

The folder structure should be as follows:

    root
    ├── arrows/
    ├── id/images/           
    │   ├── 10-xxx                  
    │   │   ├── 10-10.jpg
    │   │   ├── 10-11.jpg      
    │   │   └── ...
    │   ├── 11-xxx                  
    │   │   ├── 11-10.jpg
    │   │   └── 11-11.jpg
    │   │   └── ...
    │   └── ...
    ├── sw/images/           
    │   ├── 10-xxx                       
    │   │   └── ...
    │   ├── 11-xxx
    │   │   └── ...
    │   └── ...
    ├── ta/images/           
    │   ├── 10-xxx                    
    │   │   └── ...
    │   ├── 11-xxx
    │   │   └── ...
    │   └── ...
    ├── tr/images/           
    │   ├── 10-xxx                     
    │   │   └── ...
    │   ├── 11-xxx
    │   │   └── ...
    │   └── ...
    ├── zh/images/           
    │   ├── 10-xxx                   
    │   │   └── ...
    │   ├── 11-xxx
    │   │   └── ...
    │   └── ...
    ├── caption_vlue_test.json
    ├── caption_vlue_test_gt.json
    ├── nlvr2_vlue_test.json
    └── vqa_vlue_test.json

```python
from vilt.utils import write_ood_nlvr2
from vilt.utils import write_ood_vqa

write_ood_nlvr2(nlvr2_root, nlvr2_root/arows)
write_ood_vqa(vqav2_root, vqav2_root/arrows)
```

Throught this study we used NLVR2 `\cite{suhr2018corpus}`, VQA 2.0 `\cite{goyal2017making}` datasets, and VLUE benchmark `\cite{zhou2022vlue}`.


```
@article{suhr2018corpus,
  title={A corpus for reasoning about natural language grounded in photographs},
  author={Suhr, Alane and Zhou, Stephanie and Zhang, Ally and Zhang, Iris and Bai, Huajun and Artzi, Yoav},
  journal={arXiv preprint arXiv:1811.00491},
  year={2018}
}

@inproceedings{goyal2017making,
  title={Making the v in vqa matter: Elevating the role of image understanding in visual question answering},
  author={Goyal, Yash and Khot, Tejas and Summers-Stay, Douglas and Batra, Dhruv and Parikh, Devi},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={6904--6913},
  year={2017}
}

@article{zhou2022vlue,
      author    = {Wangchunshu Zhou and Yan Zeng and Shizhe Diao and Xinsong Zhang},
      title     = {VLUE: A Multi-Task Benchmark for Evaluating Vision-Language Models},
      journal   = {CoRR},
      volume    = {abs/2205.15237},
      year      = {2022},
      archivePrefix = {arXiv},
      eprint    = {2205.15237}
}
```