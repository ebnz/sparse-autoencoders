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
    help="Path of Tokenized Dataset to use for Obtaining Interpretation Samples"
)

parser.add_argument(
    "--autoencoder_path",
    default="./autoencoder.pt",
    type=str,
    help="Path of Autoencoder Model to analyze"
)

parser.add_argument(
    "--save_path",
    default="./interpretation_samples.pt",
    type=str,
    help="Path to save the Interpretation Samples to"
)

parser.add_argument(
    "--target_model_name",
    default="codellama/CodeLlama-7b-Instruct-hf",
    type=str,
    help="Name of Target-Model. Currently, only CodeLlama-Models are supported"
)

parser.add_argument(
    "--target_model_device",
    default="cuda:0",
    type=str,
    help="Device of Target Model"
)

parser.add_argument(
    "--autoencoder_device",
    default="cuda:1",
    type=str,
    help="Device of Autoencoder"
)

parser.add_argument(
    "--log_freq_upper",
    default=-0.1,
    type=float,
    help="Maximal Log-Feature-Frequency at which a Feature should be interpreted"
)

parser.add_argument(
    "--log_freq_lower",
    default=-2,
    type=float,
    help="Minimal Log-Feature-Frequency at which a Feature should be interpreted"
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

LOG_FREQ_UPPER = args.log_freq_upper
LOG_FREQ_LOWER = args.log_freq_lower

"""
Checks
"""
if LOG_FREQ_UPPER <= LOG_FREQ_LOWER:
    raise argparse.ArgumentError("log_freq_upper <= log_freq_lower. "
                                 "Please make sure that log_freq_upper > log_freq_lower")

if LOG_FREQ_UPPER >= 0 or LOG_FREQ_LOWER >= 0:
    raise argparse.ArgumentError("log_freq_upper and log_freq_lower must be negative!")

"""
Obtain Interpretation Samples
"""

interpretation_config = CodeLlamaInterpretationConfig(
    DATASET_PATH,
    TARGET_MODEL_NAME,
    "",
    AUTOENCODER_PATH
)

interpreter = AutoInterpreter(interpretation_config)

interpreter.load_dataset()
interpreter.load_target_model(TARGET_MODEL_DEVICE)
interpreter.load_autoencoder(AUTOENCODER_DEVICE)

interpreter.obtain_interpretation_samples(NUM_SAMPLES, log_freq_upper=LOG_FREQ_UPPER, log_freq_lower=LOG_FREQ_LOWER)
interpreter.save_interpretation_samples(SAVE_PATH)
