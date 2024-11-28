import argparse
import os
from tqdm import tqdm

from sparse_autoencoders.AutoInterpretation import *

# ToDo: Implement Uploading data to ElasticIndex

parser = argparse.ArgumentParser()

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
    "--interpretation_samples_path",
    default="./interpretation_samples.pt",
    type=str,
    help="ID of Layer from which the Activations are obtained"
)

parser.add_argument(
    "--interpretation_model_name",
    default="codellama/CodeLlama-34b-Instruct-hf",
    type=str,
    help="ID of Layer from which the Activations are obtained"
)

parser.add_argument(
    "--num_gpus",
    default=4,
    type=int,
    help="Number of GPUs to use"
)

parser.add_argument(
    "--num_interpretation_samples",
    default=3,
    type=int,
    help="Number of Interpretation Samples to use"
)

parser.add_argument(
    "--num_simulation_samples",
    default=3,
    type=int,
    help="Number of Simulation Samples to use"
)

parser.add_argument(
    "--local_rank",
    default=0,
    type=int,
    help="Local Rank"
)

"""
Parse Arguments
"""
args = parser.parse_args()

DATASET_PATH = args.dataset_path
AUTOENCODER_PATH = args.autoencoder_path
INTERPRETATION_SAMPLES_PATH = args.interpretation_samples_path

INTERPRETATION_MODEL_NAME = args.interpretation_model_name

NUM_GPUS = args.num_gpus

NUM_INTERPRETATION_SAMPLES = args.num_interpretation_samples
NUM_SIMULATION_SAMPLES = args.num_simulation_samples

"""
Interpretation
"""

interpretation_config = InterpretationConfig(
    DATASET_PATH,
    "",
    INTERPRETATION_MODEL_NAME,
    ""
)

local_rank = int(os.getenv("LOCAL_RANK", "0"))

interpreter = AutoInterpreter(interpretation_config)
interpreter.load_dataset()
interpreter.load_interpretation_model_deepspeed(NUM_GPUS)
interpreter.load_interpretation_samples(INTERPRETATION_SAMPLES_PATH)


if local_rank == 0:
    file = open("/nfs/home/ebenz_bsc2024/sims.txt", "w")

progress_bar = tqdm(desc="Interpretation", total=20)

for idx, feature_index in enumerate(interpreter.interpretable_neuron_indices):
    user_prompt_interpretation = interpreter.generate_interpretation_prompt(feature_index, NUM_INTERPRETATION_SAMPLES)
    interpretation = interpreter.get_explanation(user_prompt_interpretation)

    user_prompt_simulation = interpreter.generate_simulation_prompt(feature_index, NUM_SIMULATION_SAMPLES, interpretation)

    scores_simulated = interpreter.get_simulation(user_prompt_simulation)
    scores_gt = interpreter.generate_ground_truth_scores(feature_index, NUM_SIMULATION_SAMPLES)

    if local_rank == 0:
        file.write("\nSimulated\n")
        file.write(str(scores_simulated))
        file.write("\nGround Truth\n")
        file.write(str(scores_gt))
        file.write("\n---------------------------------------------------------\n")
        progress_bar.update(1)

    if idx >= 20:
        if local_rank == 0:
            file.close()
        break
