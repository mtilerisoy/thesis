from datetime import datetime
current_time = datetime.now().strftime("%Y%m%d_%H%M")
print(current_time)

EPOCHS = 1
MAX_STEPS = 90
LEARNING_RATE = 0.0001 #1e-5 #0.0001

DATASET = "nlvr2_original"
PERCENTAGE = 1
GPU = [1]

ALPHA_KD = 0.5
KD_LAYER = 2 # this is METER model
TEMPERATURE = 1.0

modules_to_train = {'layer_names': [f"text_transformer.encoder.layer.{KD_LAYER}.intermediate.dense",
                                    f"text_transformer.encoder.layer.{KD_LAYER}.output.dense",
                                    f"cross_modal_text_pooler.dense"],
                        'kd_layer': KD_LAYER}


LOG_DIR = "experiments/METER/logs"
EXP_NAME = f"xxx_{current_time}_train_all_compress_all_Pooler_short{LEARNING_RATE}"