import argparse
from sparse_autoencoders.AutoInterpretation import *

parser = argparse.ArgumentParser()

parser.add_argument(
    "num_samples",
    type=int,
    help="Number of Interpretation Samples to obtain"
)

parser.add_argument(
    "--dataset_path",
    default="./tokenized_dataset",
    type=str,
    help="ID of Layer from which the Activations are obtained"
)

parser.add_argument(
    "--autoencoder_path",
    default="./autoencoder.pt",
    type=str,
    help="ID of Layer from which the Activations are obtained"
)

parser.add_argument(
    "--save_path",
    default="./interpretation_samples.pt",
    type=str,
    help="ID of Layer from which the Activations are obtained"
)

parser.add_argument(
    "--target_model_name",
    default="codellama/CodeLlama-7b-Instruct-hf",
    type=str,
    help="ID of Layer from which the Activations are obtained"
)

parser.add_argument(
    "--target_model_device",
    default="cuda:0",
    type=str,
    help="ID of Layer from which the Activations are obtained"
)

parser.add_argument(
    "--autoencoder_device",
    default="cuda:1",
    type=str,
    help="ID of Layer from which the Activations are obtained"
)

"""
Parse Arguments
"""
args = parser.parse_args()

NUM_SAMPLES = args.num_samples

DATASET_PATH = args.dataset_path
AUTOENCODER_PATH = args.autoencoder_path
SAVE_PATH = args.save_path

TARGET_MODEL_NAME = args.target_model_name

TARGET_MODEL_DEVICE = args.target_model_device
AUTOENCODER_DEVICE = args.autoencoder_device

"""
interpretation_config = InterpretationConfig(
    "/nfs/data/students/ebenz_bsc2024/tokenized_dataset",
    "codellama/CodeLlama-7b-Instruct-hf",
    "codellama/CodeLlama-34b-Instruct-hf",
    "/nfs/data/students/ebenz_bsc2024/autoenc_2/autoenc_lr2e-4_0.5_32_nr/50000.pt"
)
"""

"""
Obtain Interpretation Samples
"""

interpretation_config = InterpretationConfig(
    DATASET_PATH,
    TARGET_MODEL_NAME,
    "",
    AUTOENCODER_PATH
)

interpreter = AutoInterpreter(interpretation_config)

interpreter.load_dataset()
interpreter.load_target_model(TARGET_MODEL_DEVICE)
interpreter.load_autoencoder(AUTOENCODER_DEVICE)

interpreter.obtain_interpretation_samples(NUM_SAMPLES)
interpreter.save_interpretation_samples(SAVE_PATH)
