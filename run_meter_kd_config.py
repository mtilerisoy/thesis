from datetime import datetime
current_time = datetime.now().strftime("%Y%m%d_%H%M")
print(current_time)

EPOCHS = 3
MAX_STEPS = -1
LEARNING_RATE = 0.0001 #1e-5

DATASET = "nlvr2_original"
PERCENTAGE = 1
GPU = [0]

ALPHA_KD = 1
KD_LAYER = 2 # this is METER model
TEMPERATURE = 1.0

modules_to_train = {'layer_names': [#"scale_factor",
                                    f"text_transformer.encoder.layer.{KD_LAYER}.intermediate.dense",
                                    f"text_transformer.encoder.layer.{KD_LAYER}.output.dense"],
                        'kd_layer': KD_LAYER}


LOG_DIR = "experiments/METER/logs"
EXP_NAME = f"xxx_{current_time}_NO_Scale_Distill_Pooler_lr{LEARNING_RATE}_correct_loss"