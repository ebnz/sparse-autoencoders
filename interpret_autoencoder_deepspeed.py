import argparse
from tqdm import tqdm
from elasticsearch import Elasticsearch

from sparse_autoencoders.AutoInterpretation import *
from sparse_autoencoders.utils import calculate_correlation_from_kv_dict

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset_path",
    default="./tokenized_dataset",
    type=str,
    help="Path of Tokenized Dataset to use for Obtaining Interpretation Samples"
)

parser.add_argument(
    "--interpretation_samples_path",
    default="./interpretation_samples.pt",
    type=str,
    help="Path of the saved Interpretation-Samples"
)

parser.add_argument(
    "--interpretation_model_name",
    default="meta-llama/Llama-2-13b-chat-hf",
    type=str,
    help="Name of Interpretation-Model. Currently, only CodeLlama-Models are supported"
)

parser.add_argument(
    "--num_gpus",
    default=4,
    type=int,
    help="Number of GPUs to use"
)

parser.add_argument(
    "--autoencoder_path",
    default="./autoencoder.pt",
    type=str,
    help="Path to Autoencoder to interpret"
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

parser.add_argument(
    "--ssl_cert",
    default="",
    type=str,
    help="Path to SSL-Cert of ElasticSearch"
)

parser.add_argument(
    "server_address",
    type=str,
    help="Address to ElasticSearch-Server (e.g. https://DOMAIN:PORT)"
)

parser.add_argument(
    "api_key",
    type=str,
    help="API-Key to ElasticSearch-Server"
)

"""
Parse Arguments
"""
args = parser.parse_args()

DATASET_PATH = args.dataset_path
INTERPRETATION_SAMPLES_PATH = args.interpretation_samples_path

INTERPRETATION_MODEL_NAME = args.interpretation_model_name

NUM_GPUS = args.num_gpus

AUTOENCODER_PATH = args.autoencoder_path

NUM_INTERPRETATION_SAMPLES = args.num_interpretation_samples
NUM_SIMULATION_SAMPLES = args.num_simulation_samples

SSL_CERT_PATH = args.ssl_cert

LOCAL_RANK = args.local_rank

SERVER_ADDRESS = args.server_address
API_KEY = args.api_key

"""
Interpretation
"""

interpretation_config = CodeLlamaInterpretationConfig(
    DATASET_PATH,
    "",
    INTERPRETATION_MODEL_NAME,
    ""
)

interpreter = AutoInterpreter(interpretation_config)
interpreter.load_dataset()
interpreter.load_interpretation_model_deepspeed(NUM_GPUS)
interpreter.load_interpretation_samples(INTERPRETATION_SAMPLES_PATH)

with open(AUTOENCODER_PATH, "rb") as f:
    sae_config = pickle.load(f)

# ElasticSearch Index-Name
INDEX_NAME = (f'llama_{sae_config["MODEL_TYPE"]}_{sae_config["LAYER_TYPE"]}_{sae_config["LAYER_INDEX"]}_'
              f'{sae_config["ACT_VEC_SIZE"]}_{sae_config["DICT_VEC_SIZE"]}_{sae_config["LEARNING_RATE"]}_'
              f'{sae_config["L1_COEFFICIENT"]}').lower()

if LOCAL_RANK == 0:
    # Open ElasticSearch-Connection
    if SSL_CERT_PATH == "":
        print("WARN: Skipping the SSL-Cert-Verification. Provide a Path to a SSL-Certificate in non-protected Networks")
        client = Elasticsearch(SERVER_ADDRESS, api_key=API_KEY, verify_certs=False)
    else:
        client = Elasticsearch(SERVER_ADDRESS, api_key=API_KEY, verify_certs=True, ca_certs=SSL_CERT_PATH)

NUM_INTERPRETABLE_FEATURES = len(interpreter.interpretable_neuron_indices)
progress_bar = tqdm(desc="Interpretation", total=NUM_INTERPRETABLE_FEATURES)

for idx, item in enumerate(interpreter.interpretable_neuron_indices):
    interp_neuron_index = idx
    feature_index = item.item()

    # Interpretation
    user_prompt_interpretation = interpreter.generate_interpretation_prompt(interp_neuron_index,
                                                                            NUM_INTERPRETATION_SAMPLES)
    interpretation = interpreter.get_explanation(user_prompt_interpretation)

    # Simulation
    scores_gt, scores_simulated = {}, {}
    for i in range(NUM_SIMULATION_SAMPLES):
        user_prompt_simulation = interpreter.generate_simulation_prompt(interp_neuron_index,
                                                                        NUM_INTERPRETATION_SAMPLES + i, interpretation)

        # Merge dicts
        scores_simulated = scores_simulated | interpreter.get_simulation(user_prompt_simulation)
        scores_gt = scores_gt | interpreter.generate_ground_truth_scores(interp_neuron_index,
                                                                         NUM_INTERPRETATION_SAMPLES + i)

    correlation_score = calculate_correlation_from_kv_dict(scores_gt, scores_simulated)

    tokens = []

    for token in scores_gt:
        if scores_gt[token] > 0:
            tokens.append(token)

    if LOCAL_RANK == 0:
        document = {
            "layer": sae_config["LAYER_INDEX"],
            "dim": item.item(),
            "tokens": tokens,
            "score": correlation_score,
            "interpretation": interpretation
        }
        client.create(document=document, index=INDEX_NAME, id=feature_index)
        progress_bar.update(1)
