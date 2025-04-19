from datetime import datetime
current_time = datetime.now().strftime("%Y%m%d_%H%M")

EPOCHS = 5
MAX_STEPS = 50
LEARNING_RATE = 0.0001

DATASET = "vqa_original"
PERCENTAGE = 1
GPU = [0]

ALPHA_KD = 0 # 0 for QAT only and 1 for QAT + KD
KD_LAYER = 0
TEMPERATURE = 1.0

modules_to_train = {'layer_names': [f"transformer.blocks.{KD_LAYER}.mlp.fc1",
                                    f"transformer.blocks.{KD_LAYER}.mlp.fc2",
                                    f"pooler.dense"],
                        'kd_layer': KD_LAYER}


LOG_DIR = "experiments/logs"
EXP_NAME = f"xxx{current_time}VQA_train_all_compress_QAT{LEARNING_RATE}"