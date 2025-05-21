# Evaluation
The results will vary a bit based on the batch size selection because the models do a batched-inference. This yields padded image batch that would be inconsistently embedded while performing linear image patch projection.

## Inference
- For evaluating full precision inference use run_vilt.py for ViLT models and run_meter.py for METER model.
- For evaluating post-training dynamic quantization use run_vilt_ptq.py for ViLT models and run_meter_ptq.py for METER model.


### Evaluating VQAv2
```python
python run_vilt.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> per_gpu_batchsize=<BS_FITS_YOUR_GPU> task_finetune_vqa_randaug test_only=True precision=32 load_path="<YOUR_WEIGHT_ROOT>/model_weights_vqa.ckpt"

example:
python run_vilt.py with data_root=/data-4/users/mileriso/datasets/VQAv2/arrows num_gpus=1 num_nodes=1 per_gpu_batchsize=32 task_finetune_vqa_randaug test_only=True precision=32 load_path="/data-4/users/mileriso/models/vilt_vqa.ckpt"
```

This script will generate 'result/vqa_submit_vilt_vqa_DATE.json'. Then you need to run the assessment script as follows:
```python
python assess_vqa.py model_name prediction_file_name

example:
python assess_vqa.py vilt vqa_submit_vilt_vqa_20250521_1007
```

The prediction json file can be found under ```result/``` The output will shows the correct number of guesses as well as the accuracy.

### Evaluating NLVR2
```python
python run_vilt.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> per_gpu_batchsize=<BS_FITS_YOUR_GPU> task_finetune_nlvr2_randaug test_only=True precision=32 load_path="<YOUR_WEIGHT_ROOT>/model_weights_nlvr2.ckpt"

example:
python run_meter.py with data_root=/data-4/users/mileriso/datasets/NLVR2/arrows num_gpus=1 num_nodes=1  task_finetune_nlvr2_clip_bert per_gpu_batchsize=8 load_path="/data-4/users/mileriso/models/meter_nlvr2.ckpt" clip16 text_roberta image_size=288 test_only=True
```
The output will look like:
```python
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'nlvr2/dev/accuracy': tensor(0.7486, device='cuda:0'),
 'nlvr2/dev/accuracy_epoch': tensor(0.7565, device='cuda:0'),
 'nlvr2/dev/loss': tensor(0.8581, device='cuda:0'),
 'nlvr2/dev/loss_epoch': tensor(0.8609, device='cuda:0'),
 'nlvr2/test/accuracy': tensor(0.7735, device='cuda:0'),
 'nlvr2/test/accuracy_epoch': tensor(0.7652, device='cuda:0'),
 'nlvr2/test/loss': tensor(0.7796, device='cuda:0'),
 'nlvr2/test/loss_epoch': tensor(0.8381, device='cuda:0'),
 'val/the_metric': tensor(0.7652, device='cuda:0')}
--------------------------------------------------------------------------------
INFO - ViLT - Completed after 0:01:31
```

The important metric is the ```'nlvr2/test/accuracy'``` for this study.


## Evaluating CLS-KD (Retraining)

First step is replace the ```data_root``` and ```load_path``` fields with your dataset and model directories under ```configs.py```.  Here each dictionary defines the CLI arguments for running the experiments. The name of the dictionary indicates for which model, task, and dataset this configuration is. For example ```meter_config_nlvr2_id``` means the METER model using in-distribution dataset on NLVR2 task.


### Training Configuration
```run_vilt_config.py``` and ```run_meter_config.py``` files contain the definitions of the training hyperparametes for ViLT and METER models respectively. These values are pre-configured but can be changed. These hyperparameters are:

**Core Training Parameters**

- EPOCHS: Defines the number of complete passes through the training dataset
- MAX_STEPS: Maximum number of training iterations regardless of epoch completion
- LEARNING_RATE: Step size at which the model's parameters are updated during training. 0.0001 is the default.
- DATASET: Specifies which dataset to use for training options are nlvr_ood and nlvr_id for OOD and in-domain datasets respectively.
- PERCENTAGE: Fraction of the dataset to use can have values between 0 and 1 where 1 = 100% of the dataset
- GPU: List of GPU device IDs to use for training ([0] means using only the first GPU)

**Knowledge Distillation Hyper-Parameters**

These values are the hyperparameters of our method and default values are already set. Unless you want to tune these should not change them.

- ALPHA_KD: Controls the distillation strengt. Default is 0.5 and 0 for QAT-only training.
- KD_LAYER: Specifies which transformer layer to apply knowledge distillation on. These are the bottleneck layers identified within the study.
- modules_to_train: Dictionary defining which specific layers to fine-tune during training. The rest of the layers will be frozen durin retraining
    - layer_names: List of model components to be trained
    - kd_layer: Layer for knowledge distillation

**Experiment Tracking**
- LOG_DIR: Directory where training logs are saved (tesnorboard)
- EXP_NAME: Unique experiment name with timestamp for tracking different runs

These parameters allow you to control various aspects of model training, from basic hyperparameters to specific knowledge distillation settings.

After the configuration, to run retraining and evaluate the method use:
```python
python run_vilt_kd.py
python run_meter_kd.py
```

This will give an output similar to the NLVR2 inference.