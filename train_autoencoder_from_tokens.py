import torch
from torch.utils.data import DataLoader

from sparse_autoencoders.TransformerModels import CodeLlamaModel
from sparse_autoencoders import AutoEncoder
from sparse_autoencoders.Datasets import TokenizedDatasetPreload

import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument(
    "layer_id",
    type=int,
    help="ID of Layer from which the Activations are obtained"
)

parser.add_argument(
    "layer_type",
    type=str,
    help="Type of Layer from which the Activations are collected. Select 'attn_sublayer', 'mlp_sublayer' or "
         "'mlp_activations'."
)

parser.add_argument(
    "--num_batches",
    default=0,
    type=int,
    help="Amount of Batches (AutoEncoder-sized) used to train the AutoEncoder on"
)

parser.add_argument(
    "--dataset_path",
    default="./tokenized_dataset",
    type=str,
    help="Path to the Pretokenized Dataset"
)

parser.add_argument(
    "--save_path",
    default="./trained_autoencoders",
    type=str,
    help="Path to save the trained AutoEncoders to"
)

parser.add_argument(
    "--model_name",
    default="codellama/CodeLlama-7b-Instruct-hf",
    type=str,
    help="HuggingFace-Name for the Model for obtaining Activations. Currently only works for CodeLlama-Models"
)

parser.add_argument(
    "--batch_size_llm",
    default=16,
    type=int,
    help="Batch Size used to obtain Model Activations from the LLM"
)

parser.add_argument(
    "--batch_size_autoencoder",
    default=256,
    type=int,
    help="Batch Size used to train the AutoEncoder"
)

parser.add_argument(
    "--num_tokens",
    default=0,
    type=int,
    help="Amount of Tokens used to train AutoEncoder"
)

parser.add_argument(
    "--device_llm",
    default="cuda:0",
    type=str,
    help="Device to load the LLM to"
)

parser.add_argument(
    "--device_autoencoder",
    default="cuda:1",
    type=str,
    help="Device to load the AutoEncoder to"
)

parser.add_argument(
    "--learning_rate",
    type=float,
    default=1e-4,
    help="Learning-Rate for AutoEncoder-Training. "
)

parser.add_argument(
    "--l1_coefficient",
    type=float,
    default=1,
    help="L1-Coefficient or Sparsity-Coefficient for AutoEncoder-Training. "
)

parser.add_argument(
    "--act_vec_size",
    type=int,
    default=11008,
    help="Size of the Activation-Vector inputted to the AutoEncoder. "
)
parser.add_argument(
    "--dict_vec_size",
    type=int,
    default=11008*4,
    help="Size of the Dictionary-Vector produced by the AutoEncoder. "
)

parser.add_argument(
    "--batches_between_ckpt",
    type=int,
    default=12500,
    help="Number of Batches to train the AutoEncoder, before the Model is saved as a Checkpoint-File and a "
         "Feature-Frequencies-Image is generated."
)

parser.add_argument(
    "--num_batches_preload",
    type=int,
    default=25_000,
    help="Buffer-Size of Activation-Vectors for Training. If Buffer is empty, will be refilled. "
         "Larger Buffer results in higher Randomness while Training. "
)

parser.add_argument(
    "--neuron_resampling_method",
    type=str,
    default="",
    help="Strategy for Neuron-Resampling. Currently available: 'replacement', 'anthropic'"
)

parser.add_argument(
    "--neuron_resampling_interval",
    type=int,
    default=25_000,
    help="Amount of Batches (AutoEncoder-sized) to train, until Neuron-Resampling"
)

parser.add_argument(
    "--normalize_dataset",
    action="store_true",
    help="If activated, all Activation Vectors in the Dataset will be normalized to L2-Norm of sqrt(n) "
         "(with n being input dimension of AutoEncoder)"
)

parser.add_argument(
    "--mlp_activations_hookpoint",
    type=str,
    default="model.layers.{}.mlp.act_fn",
    help="Hookpoint description for MLP-Activations. e.g. model.layers.{}.mlp.act_fn ({} for Layer Index)"
)

parser.add_argument(
    "--mlp_sublayer_hookpoint",
    type=str,
    default="model.layers.{}.mlp",
    help="Hookpoint description for MLP-Sublayer. e.g. model.layers.{}.mlp ({} for Layer Index)"
)

parser.add_argument(
    "--attn_sublayer_hookpoint",
    type=str,
    default="model.layers.{}.self_attn",
    help="Hookpoint description for Attention-Sublayer. e.g. model.layers.{}.self_attn ({} for Layer Index)"
)

"""
Parse Arguments
"""
args = parser.parse_args()

LAYER_INDEX = args.layer_id
LAYER_TYPE = args.layer_type

NUM_BATCHES = args.num_batches

DATASET_PATH = args.dataset_path
SAVE_PATH = args.save_path
IMAGE_PATH = os.path.join(SAVE_PATH, "freq_images")

TARGET_MODEL_NAME = args.model_name

DEVICE_LLM = args.device_llm
DEVICE_AE = args.device_autoencoder

LEARNING_RATE = args.learning_rate
L1_COEFFICIENT = args.l1_coefficient

ACT_VEC_SIZE = args.act_vec_size
DICT_VEC_SIZE = args.dict_vec_size

BATCH_SIZE_LLM = args.batch_size_llm
BATCH_SIZE_AE = args.batch_size_autoencoder
NUM_TOKENS = args.num_tokens

BATCHES_BETWEEN_CHECKPOINTS = args.batches_between_ckpt

NUM_BATCHES_PRELOAD = args.num_batches_preload

NEURON_RESAMPLING_METHOD = args.neuron_resampling_method
NEURON_RESAMPLING_INTERVAL = args.neuron_resampling_interval

NORMALIZE_DATASET = args.normalize_dataset

MLP_ACTIVATIONS_HOOKPOINT = args.mlp_activations_hookpoint
MLP_SUBLAYER_HOOKPOINT = args.mlp_sublayer_hookpoint
ATTN_SUBLAYER_HOOKPOINT = args.attn_sublayer_hookpoint

"""
Checks for Arguments
"""
# layer_id and layer_type are checked in TransformerModels-Class

if not os.path.isdir(DATASET_PATH):
    raise Exception("Provided dataset_path does not exist")

if not os.path.isdir(SAVE_PATH):
    os.mkdir(SAVE_PATH)

if not os.path.isdir(IMAGE_PATH):
    os.mkdir(IMAGE_PATH)

if BATCH_SIZE_AE % BATCH_SIZE_LLM != 0:
    raise Exception("batch_size_ae must be multiple of batch_size_llm!")

"""
Load Dataset and Model
"""
# Load Dataset
dataset = TokenizedDatasetPreload(DATASET_PATH, dtype=torch.int)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_LLM, shuffle=True)

# If NUM_TOKENS == 0, use all available Tokens
# next(iter(dataloader)) is of shape [BATCH_SIZE_LLM, NUM_TOKENS]
if NUM_TOKENS == 0:
    NUM_TOKENS = next(iter(dataloader)).shape[-1]

print(f"using {NUM_TOKENS} tokens")

# Load Target Model
target_model = CodeLlamaModel(TARGET_MODEL_NAME, device=DEVICE_LLM)


"""
Custom Hook-Methods for Obtaining/Saving Activations
"""
def hook_mlp(module, input, output):
    global raw_activation_vecs

    # Append each MLP-/Attention-Activations to raw_activation_vecs
    raw_activation_vecs.append(output.detach().cpu())


def hook_attn(module, input, output):
    global raw_activation_vecs

    # Append each MLP-/Attention-Activations to raw_activation_vecs
    raw_activation_vecs.append(output[0].detach().cpu())


def hook_mlp_acts(module, input, output):
    global raw_activation_vecs

    # Append each MLP-/Attention-Activations to raw_activation_vecs
    raw_activation_vecs.append(output.detach().cpu())


if LAYER_TYPE == "mlp_sublayer":
    target_model.setup_hook(hook_mlp, MLP_SUBLAYER_HOOKPOINT.format(LAYER_INDEX))
elif LAYER_TYPE == "attn_sublayer":
    target_model.setup_hook(hook_mlp, ATTN_SUBLAYER_HOOKPOINT.format(LAYER_INDEX))
elif LAYER_TYPE == "mlp_activations":
    target_model.setup_hook(hook_mlp, MLP_ACTIVATIONS_HOOKPOINT.format(LAYER_INDEX))
else:
    raise AttributeError("Unrecognized Type of layer_type")


"""
Go forth and Transform!
"""

# Temporary location for the Activation Vectors until they're used for training the AutoEncoder
raw_activation_vecs = []

model = AutoEncoder.AutoEncoderAnthropic(ACT_VEC_SIZE, DICT_VEC_SIZE).to(DEVICE_AE)
if NEURON_RESAMPLING_METHOD != "":
    model.enable_neuron_resampling(NEURON_RESAMPLING_METHOD, NEURON_RESAMPLING_INTERVAL)
if BATCHES_BETWEEN_CHECKPOINTS != 0:
    model.enable_checkpointing(LEARNING_RATE, L1_COEFFICIENT, BATCH_SIZE_AE, LAYER_TYPE, LAYER_INDEX, SAVE_PATH,
                               IMAGE_PATH, BATCHES_BETWEEN_CHECKPOINTS)

model.train(True)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

trained_batches = 0
training_finished = False

"""
A 'short' Comment on num_batches_to_train = int((BATCH_SIZE_LLM / BATCH_SIZE_AE) * NUM_TOKENS * len(interpreter.dl))

len(interpreter.dl) gives amount of Batches with size of BATCH_SIZE_LLM;

Multiplication with NUM_TOKENS:
From each Text-/Code-Sample, we take one Activation Vector for each Token.
Since there are 64 (by default) Tokens per Sample, we take 64 Activation Vectors per Sample;

Multiplication with (BATCH_SIZE_LLM / BATCH_SIZE_AE):
Calculate Amount of Batches with size of BATCH_SIZE_AE from Amount of Batches with size of BATCH_SIZE_LLM
"""
if NUM_BATCHES != 0:
    num_batches_to_train = NUM_BATCHES
else:
    num_batches_to_train = int((BATCH_SIZE_LLM / BATCH_SIZE_AE) * NUM_TOKENS * len(dataloader))

training_pbar = tqdm(desc="Training", total=num_batches_to_train)
buffer_pbar = tqdm(desc="Filling up Buffer", total=NUM_BATCHES_PRELOAD)

for input_ids in dataloader:
    # Generate Activation Vectors if NUM_BATCHES_PRELOAD is not met
    if NUM_TOKENS * len(raw_activation_vecs) < NUM_BATCHES_PRELOAD * (BATCH_SIZE_AE / BATCH_SIZE_LLM):
        cropped_input_ids = input_ids[::, :NUM_TOKENS]
        input_ids_cuda = cropped_input_ids.to(DEVICE_LLM)

        target_model.run_model_until_layer(input_ids_cuda, LAYER_INDEX)
        buffer_pbar.update(int(NUM_TOKENS * BATCH_SIZE_LLM / BATCH_SIZE_AE))

        continue

    buffer_pbar.reset()

    # Prepare Batches, if enough Activation Vectors Sampled
    unbatched = torch.vstack(raw_activation_vecs)
    unbatched = torch.reshape(unbatched, [unbatched.shape[0] * unbatched.shape[1], unbatched.shape[2]])

    if NORMALIZE_DATASET:
        # Scale all Dataset Rows to L2-Norm of sqrt(n) (with n being input dimension of AutoEncoder)
        avg_norm = torch.norm(unbatched, p=2)
        unbatched = torch.sqrt(torch.Tensor([model.n])) * unbatched / avg_norm

    print(unbatched.shape)
    # Shuffle
    random_indexing = torch.randperm(unbatched.shape[0])
    unbatched = unbatched[random_indexing]

    # Make Batches
    batches = list(torch.split(unbatched, BATCH_SIZE_AE))

    # Reset Raw Activation Vectors
    raw_activation_vecs = []

    for batch in batches:
        # Train one step
        trained_batches += 1
        X = batch.to(DEVICE_AE)
        X_hat, f = model(X)

        optimizer.zero_grad()

        loss = model.loss(X, X_hat, f, l1=L1_COEFFICIENT)
        loss.backward()

        optimizer.step()

        training_pbar.update(1)

        # Early-Stopping after Amount of Batches (in BATCH_SIZE_AE)
        if trained_batches > NUM_BATCHES and NUM_BATCHES != 0:
            training_finished = True
            break

    if training_finished:
        break
