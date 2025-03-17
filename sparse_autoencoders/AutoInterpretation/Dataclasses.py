import torch
from dataclasses import dataclass

import sparse_autoencoders

from .PromptGenerator import PromptGeneratorBase
from .TokenScoreRegexFilter import TokenScoreRegexFilter, TokenScoreRegexFilterAverage

# from sparse_autoencoders.AutoInterpretation.TokenScoreRegexFilter import TokenScoreRegexFilter
# from sparse_autoencoders.AutoInterpretation.PromptGenerator import PromptGeneratorBase

"""
Dataclasses for Interpretation
"""
@dataclass
class InterpretationConfig:
    dataset_path: str

    target_model_name: str

    interpretation_model_name: str

    autoencoder_path: str

    prompt_builder: PromptGeneratorBase

    mlp_sublayer_module_name: str
    attn_sublayer_module_name: str
    mlp_activations_module_name: str

    @staticmethod
    def get_mlp_hook(autoencoder, autoencoder_device, interpretable_neuron_indices, dict_vecs=None):
        def mlp_hook(module, input, output):
            activations = output.detach().cpu()
            X_hat, f = autoencoder(activations.to(autoencoder_device, torch.float32))
            if dict_vecs is not None:
                # Only select the Features with log-Frequency between boundaries set in obtain_interpretation_samples
                dict_vec = f.detach().cpu()[::, ::, interpretable_neuron_indices]
                dict_vecs.append(dict_vec)

        return mlp_hook

    @staticmethod
    def get_attn_hook(autoencoder, autoencoder_device, interpretable_neuron_indices, dict_vecs=None):
        def attn_hook(module, input, output):
            activations = output[0].detach().cpu()
            X_hat, f = autoencoder(activations.to(autoencoder_device, torch.float32))
            if dict_vecs is not None:
                # Only select the Features with log-Frequency between boundaries set in obtain_interpretation_samples
                dict_vec = f.detach().cpu()[::, ::, interpretable_neuron_indices]
                dict_vecs.append(dict_vec)

        return attn_hook

    @staticmethod
    def get_mlp_acts_hook(autoencoder, autoencoder_device, interpretable_neuron_indices, dict_vecs=None):
        def mlp_acts_hook(model, input, output):
            activations = output.detach().cpu()
            X_hat, f = autoencoder(activations.to(autoencoder_device, torch.float32))
            if dict_vecs is not None:
                # Only select the Features with log-Frequency between boundaries set in obtain_interpretation_samples
                dict_vec = f.detach().cpu()[::, ::, interpretable_neuron_indices]
                dict_vecs.append(dict_vec)

        return mlp_acts_hook

    # Hierarchy:
    # * "for": 8 (high activation value, as it is a keyword that the neuron is looking for)
    # * "Fix": 10
    # * `Fix`: 10 (...)
    # * `Fix`: 10
    # * def: 10 (since the word "def" is present in the text)
    # * def: 10
    # * def: 8-10 (...)
    # * def: 8-10
    # * def<tab>10 (...)
    # * def<tab>10

    simulation_filters = [
    # * "for": 8 (high activation value, as it is a keyword that the neuron is looking for)
    TokenScoreRegexFilter('".+": [0-9]?[0-9]? (.+)', '\A".+":', '[0-9]+ \(')
    .set_token_cropping(1, 2)
    .set_score_cropping(0, 2),

    # * "Fix": 10
    TokenScoreRegexFilter('".+": [0-9]?[0-9]?', '".+":', '[0-9]+\Z')
    .set_token_cropping(1, 2),

    # * `Fix`: 10 (...)
    TokenScoreRegexFilter('`.+`: [0-9]?[0-9]? (.+)', '\A`.+`:', '[0-9]+ \(')
    .set_token_cropping(1, 2)
    .set_score_cropping(0, 2),

    # * `Fix`: 10
    TokenScoreRegexFilter('`.+`: [0-9]?[0-9]?', '\A`.+`:', '[0-9]+\Z')
    .set_token_cropping(1, 2),

    # * def: 10 (since the word "def" is present in the text)
    TokenScoreRegexFilter('.+: [0-9]?[0-9]? (.+)', '\A.+:', '[0-9]+ \(')
    .set_token_cropping(0, 1)
    .set_score_cropping(0, 2),

    # * def: 10
    TokenScoreRegexFilter('.+: [0-9]?[0-9]?', '\A.+:', '[0-9]+\Z')
    .set_token_cropping(0, 1),

    # * def: 8-10 (...)
    TokenScoreRegexFilterAverage('.+: [0-9]+-[0-9]+', '\A.+:', '[0-9]+-[0-9]+ \(')
    .set_token_cropping(1, 1),

    # * def: 8-10
    TokenScoreRegexFilterAverage('.+: [0-9]+-[0-9]+', '\A.+:', '[0-9]+-[0-9]+\Z')
    .set_token_cropping(1, 1),

    # * def<tab>10 (...)
    TokenScoreRegexFilter('.+[ \t]+[0-9]?[0-9]? \(', '\A\D+ ', '[0-9]+ \(')
    .set_token_cropping(0, 1)
    .set_score_cropping(0, 2),

    # * def<tab>10
    TokenScoreRegexFilter('.+[ \t]+[0-9]?[0-9]?', '\A.+ ', '[0-9]+\Z')
    .set_token_cropping(0, 1)
]

