import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlamaForCausalLM, CodeLlamaTokenizer
from utils.TokenizedDataset import *

from utils.AutoInterpretation import *

"""
DEPRECATED
"""

raise DeprecationWarning("Deprecated")

parser = argparse.ArgumentParser()

parser.add_argument(
    "layer_id",
    type=int,
    help="ID of Layer from which the Activations are obtained"
)

parser.add_argument(
    "layer_type",
    type=str,
    help="Type of Layer from which the Activations are collected. Select 'attn' or 'mlp'."
)

parser.add_argument(
    "--num_batches",
    default=0,
    type=int,
    help="Amount of Batches to process"
)

parser.add_argument(
    "--dataset_path",
    default="./tokenized_dataset",
    type=str,
    help="Path to the Pretokenized Dataset"
)

parser.add_argument(
    "--save_path",
    default="./model_activations",
    type=str,
    help="Path to save the obtained Model Activations to"
)

parser.add_argument(
    "--model_name",
    default="codellama/CodeLlama-7b-Instruct-hf",
    type=str,
    help="HuggingFace-Name for the Model for obtaining Activations. Currently only works for CodeLlama-Models"
)

parser.add_argument(
    "--batch_size",
    default=16,
    type=int,
    help=""
)

parser.add_argument(
    "--num_tokens",
    default=64,
    type=int,
    help=""
)

parser.add_argument(
    "--device",
    default="cuda:0",
    type=str,
    help="Device to load the Model to"
)

"""
Parse Arguments
"""
args = parser.parse_args()

LAYER_ID = args.layer_id
LAYER_TYPE = args.layer_type

NUM_BATCHES = args.num_batches

DATASET_PATH = args.dataset_path
TARGET_MODEL_NAME = args.model_name

DEVICE = args.device

BATCH_SIZE = args.batch_size
NUM_TOKENS = args.num_tokens

#Number of Activation-Vectors that are saved in one file. Currently kept constant
VECS_PER_DS_FILE = 256 * BATCH_SIZE * NUM_TOKENS

SAVE_PATH = args.save_path

"""
Checks for Arguments
"""
#layer_id and layer_type are checked in AutoInterpretation-Class

if not os.path.isdir(DATASET_PATH):
    raise Exception("Provided dataset_path does not exist")

if not os.path.isdir(SAVE_PATH):
    os.mkdir(SAVE_PATH)

#Technically also a check, since raises an Exception, if model_name is invalid
interpretation_config = InterpretationConfig(
    DATASET_PATH,
    TARGET_MODEL_NAME,
    TARGET_MODEL_NAME,
    ""
)

"""
Custom Hook-Methods for Obtaining/Saving Activations
"""
def hook_mlp(module, input, output):
    global raw_activation_vecs

    # Append each MLP-/Attention-Activations to mlp_activations
    mlp_activations.append(output.detach().cpu())

    # If mlp_activations contains a specific amount (VECS_PER_DS_FILE) of Activation Vectors, write the Vectors to the disk
    if len(mlp_activations) * BATCH_SIZE * NUM_TOKENS >= VECS_PER_DS_FILE:
        tensor = torch.vstack(mlp_activations)
        reshaped = torch.reshape(tensor[::, :NUM_TOKENS], (len(mlp_activations) * BATCH_SIZE * NUM_TOKENS, 4096))
        torch.save(reshaped, os.path.join(SAVE_PATH, f"fragment_{len(os.listdir(SAVE_PATH))}.pt"))
        mlp_activations = []

def hook_attn(module, input, output):
    global raw_activation_vecs

    # Append each MLP-/Attention-Activations to mlp_activations
    mlp_activations.append(output[0].detach().cpu())

    # If mlp_activations contains a specific amount (VECS_PER_DS_FILE) of Activation Vectors, write the Vectors to the disk
    if len(mlp_activations) * BATCH_SIZE * NUM_TOKENS >= VECS_PER_DS_FILE:
        tensor = torch.vstack(mlp_activations)
        reshaped = torch.reshape(tensor[::, :NUM_TOKENS], (len(mlp_activations) * BATCH_SIZE * NUM_TOKENS, 4096))
        torch.save(reshaped, os.path.join(SAVE_PATH, f"fragment_{len(os.listdir(SAVE_PATH))}.pt"))
        mlp_activations = []


interpreter = AutoInterpreter(interpretation_config)
interpreter.load_dataset(dataset_batch_size=BATCH_SIZE, num_tokens=NUM_TOKENS)
interpreter.load_target_model(DEVICE)

if LAYER_TYPE == "mlp":
    interpreter.target_model.setup_hook(hook_mlp, LAYER_ID, LAYER_TYPE)
elif LAYER_TYPE == "attn":
    interpreter.target_model.setup_hook(hook_attn, LAYER_ID, LAYER_TYPE)
else:
    raise Exception("Unrecognized Type of layer_type")


"""
Go forth and Transform!
"""

#Temporary location for the Activation Vectors until VECS_PER_DS_FILE Activation Vectors are sampled and the Vectors get stored on the disk
raw_activation_vecs = []

with tqdm(total=NUM_BATCHES) as pbar:
    for idx, input_ids in enumerate(interpreter.dl):
        cropped_input_ids = input_ids[::, :NUM_TOKENS]
        input_ids_cuda = cropped_input_ids.to(DEVICE)

        interpreter.target_model.run_target_model_until_layer(input_ids_cuda, LAYER_ID)

        pbar.update(1)

        if idx == NUM_BATCHES and NUM_BATCHES != 0:
            break

