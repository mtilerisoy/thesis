from datetime import datetime
current_time = datetime.now().strftime("%Y%m%d_%H%M")

EPOCHS = 3
MAX_STEPS = -1
LEARNING_RATE = 0.0001 #1e-5

DATASET = "nlvr2_original"
PERCENTAGE = 1
GPU = [1]

ALPHA_KD = 0
KD_LAYER = 0
TEMPERATURE = 1.0

modules_to_train = {'layer_names': [#"scale_factor",
                                        f"transformer.blocks.{KD_LAYER}.mlp.fc1",
                                        f"transformer.blocks.{KD_LAYER}.mlp.fc2"],
                        'kd_layer': KD_LAYER}


LOG_DIR = "experiments/logs"
EXP_NAME = f"xxx{current_time}_QAT_only_log_CLS_lr{LEARNING_RATE}"