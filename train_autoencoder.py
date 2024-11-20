import argparse

import torch

from utils.SAEMetrics import save_freqs_img_for_training
from utils.TokenizedDataset import *
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from utils.AutoEncoder import *
import pickle
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument(
    "layer_id",
    type=int,
    help="ID of Layer from which the Activations are obtained. "
)

parser.add_argument(
    "layer_type",
    type=str,
    help="Type of Layer from which the Activations are collected. Select 'attn' or 'mlp'. "
)

parser.add_argument(
    "--dataset_path",
    default="./model_activations",
    type=str,
    help="Path to save the trained Model-Checkpoints to. "
)

parser.add_argument(
    "--save_path",
    default="./saved_autoencoders",
    type=str,
    help="Path to save the obtained Model Activations to"
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=256,
    help="Batch Size for AutoEncoder-Training. "
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
    default=1e-3,
    help="L1-Coefficient or Sparsity-Coefficient for AutoEncoder-Training. "
)

parser.add_argument(
    "--act_vec_size",
    type=int,
    default=4096,
    help="Size of the Activation-Vector inputted to the AutoEncoder. "
)
parser.add_argument(
    "--dict_vec_size",
    type=int,
    default=16384,
    help="Size of the Dictionary-Vector produced by the AutoEncoder. "
)

parser.add_argument(
    "--device",
    type=str,
    default="cuda:0",
    help="Device to load the Model to"
)

parser.add_argument(
    "--batches_between_ckpt",
    type=int,
    default=2500,
    help="Number of Batches to train the AutoEncoder, before the Model is saved as a Checkpoint-File and a "
         "Feature-Frequencies-Image is generated."
)

parser.add_argument(
    "--batches_for_ckpt_img",
    type=int,
    default=1000,
    help="Number of Batches used to generate the Feature-Frequencies-Image. Too few Batches might not be able to "
         "represent Low-Frequency-Features properly."
)


"""
User-set Variables
"""
#ToDo: Make shell-script
args = parser.parse_args()

#AutoEncoder Hyperparameters
ACT_VEC_SIZE = args.act_vec_size
DICT_VEC_SIZE = args.dict_vec_size
SAVE_PATH = args.save_path
BATCH_SIZE_TRAINING = args.batch_size
LEARNING_RATE = args.learning_rate
L1_COEFFICIENT = args.l1_coefficient

LAYER_TYPE = args.layer_type
LAYER_INDEX = args.layer_id

BATCHES_BETWEEN_CHECKPOINTS = args.batches_between_ckpt
BATCHES_FOR_CHECKPOINT_IMG = args.batches_for_ckpt_img

DEVICE = args.device

#dataset_path = "/nfs/data/students/ebenz_bsc2024/transformer_activations/tensors_attn_l1_250_large"
dataset_path = args.dataset_path

ds = TokenizedDatasetPreload(dataset_path, partial_preload=None)
dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE_TRAINING, shuffle=True, pin_memory=True, pin_memory_device=DEVICE)

if not os.path.isdir(SAVE_PATH):
    os.mkdir(SAVE_PATH)

if not os.path.isdir(f"{SAVE_PATH}/freq_images"):
    os.mkdir(f"{SAVE_PATH}/freq_images")

"""
Training
"""
net = AutoEncoderAnthropic(ACT_VEC_SIZE, DICT_VEC_SIZE).to(DEVICE)
loss_func = AELossNN
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

losses = []
index = 0

for batch in tqdm(dl):
    index += 1
    X = batch.to(DEVICE)
    X_hat, f = net(X)

    optimizer.zero_grad()

    loss = loss_func(X, X_hat, f, lam=L1_COEFFICIENT)
    loss.backward()

    optimizer.step()
    losses.append(loss.tolist())
    if index % BATCHES_BETWEEN_CHECKPOINTS == 0:
        save_freqs_img_for_training(net, dl, DEVICE, no_batches=BATCHES_FOR_CHECKPOINT_IMG, filename=f"{SAVE_PATH}/freq_images/{index}.png")
        with open(os.path.join(SAVE_PATH, f"{index}.pt"), "wb") as f:
            pickle.dump({
                "STATE_DICT": net.state_dict(),
                "ACT_VEC_SIZE": ACT_VEC_SIZE,
                "DICT_VEC_SIZE": DICT_VEC_SIZE,
                "LEARNING_RATE": LEARNING_RATE,
                "L1_COEFFICIENT": L1_COEFFICIENT,
                "BATCH_SIZE_TRAINING": BATCH_SIZE_TRAINING,
                "LAYER_TYPE": LAYER_TYPE,
                "LAYER_INDEX": LAYER_INDEX,
                "INTERPRETATIONS": None,
                "MINS": None,
                "MAXS": None
            }, f)