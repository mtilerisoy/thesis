def init_config(num_gpus=1, num_nodes=1, per_gpu_batchsize=64, precision=32):

    return num_gpus, num_nodes, per_gpu_batchsize, precision

exp_name = "ood_nlvr2"
seed = 0
datasets = ["ood_nlvr2"]
loss_names = {"nlvr2": 1}
batch_size = 4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
accelerator = "gpu"
test_only=True
num_gpus, num_nodes, per_gpu_batchsize, precision = init_config()

# Image setting
train_transform_keys = ["pixelbert"]
val_transform_keys = ["pixelbert"]
image_size = 384
max_image_len = -1
patch_size = 32
draw_false_image = 1
image_only = False

# Text Setting
vqav2_label_size = 3129
max_text_len = 40
tokenizer = "bert-base-uncased"
vocab_size = 30522
whole_word_masking = False
mlm_prob = 0.15
draw_false_text = 0

# Transformer Setting
vit = "vit_base_patch32_384"
hidden_size = 768
num_heads = 12
num_layers = 12
mlp_ratio = 4
drop_rate = 0.1

# # Optimizer Setting
# optim_type = "adamw"
# learning_rate = 1e-4
# weight_decay = 0.01
# decay_power = 1
# max_epoch = 100
# max_steps = 25000
# warmup_steps = 2500
# end_lr = 0
# lr_mult = 1  # multiply lr for downstream heads

# # Downstream Setting
# get_recall_metric = False

# # PL Trainer Setting
# resume_from = None
# fast_dev_run = False
# val_check_interval = 1.0
# test_only = False

# below params varies with the environment
data_root = "/data-4/users/mileriso/datasets/OOD/arrows"
log_dir = "result"
per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
num_gpus = 1
num_nodes = 1
load_path="/data-4/users/mileriso/models/vilt_nlvr2.ckpt"
num_workers = 8
precision = 32

# Task Settings
train_transform_keys = ["pixelbert_randaug"]
batch_size = 128
max_epoch = 10
max_steps = 1
warmup_steps = 0.1
draw_false_image = 0
learning_rate = 1e-4