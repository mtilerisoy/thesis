from datetime import datetime
current_time = datetime.now().strftime("%Y%m%d_%H%M")

EPOCHS = 3
MAX_STEPS = -1
LEARNING_RATE = 0.01 #1e-5

DATASET = "nlvr2_original"
PERCENTAGE = 0.5
GPU = [0,1]

ALPHA_KD = 0.5
KD_LAYER = 0
TEMPERATURE = 1.0

modules_to_train = {'layer_names': ["scale_factor",
                                        f"transformer.blocks.{KD_LAYER}.mlp.fc1",
                                        f"transformer.blocks.{KD_LAYER}.mlp.fc2"],
                        'kd_layer': KD_LAYER}


LOG_DIR = "experiments/logs"
EXP_NAME = f"{current_time}_SCALE_kd_{EPOCHS}_{MAX_STEPS}_{LEARNING_RATE}_{ALPHA_KD}_{KD_LAYER}_{DATASET}_{PERCENTAGE}"