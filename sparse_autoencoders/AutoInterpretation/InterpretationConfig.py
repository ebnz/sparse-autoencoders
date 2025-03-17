import torch
from dataclasses import dataclass

from .PromptGenerator import PromptGeneratorBase, CodeLlamaPromptGenerator
from .TokenScoreRegexFilter import TokenScoreRegexFilter, TokenScoreRegexFilterAverage

"""
Dataclasses for Interpretation
"""
@dataclass
class InterpretationConfigBase:
    """
    Configuration for AutoInterpreter
    :param dataset_path: The Prompt for the LLM
    :param target_model_name: HuggingFace-Name of the Target-LLM
    :param interpretation_model_name: HuggingFace-Name of the Interpretation-LLM
    :param autoencoder_path: Path to AutoEncoder
    """
    dataset_path: str
    target_model_name: str
    interpretation_model_name: str
    autoencoder_path: str

    prompt_generator = PromptGeneratorBase()
    include_complete_texts = True
    include_filtered_tokens = True

    """
    Define Hookpoints in Transformer-Model. e.g. model.layers.{}.mlp ({} for Layer Index)
    """
    mlp_sublayer_module_name = ""
    attn_sublayer_module_name = ""
    mlp_activations_module_name = ""

    """
    Functions get_mlp_hook, get_attn_hook, get_mlp_acts_hook return a Hook-Function, which takes all Activations, 
    extracts only the Features in interpretable_neuron_indices and appends them to dict_vecs
    """
    @staticmethod
    def get_mlp_hook(autoencoder, autoencoder_device, interpretable_neuron_indices, dict_vecs=None):
        raise NotImplementedError("This class is an Interface")

    @staticmethod
    def get_attn_hook(autoencoder, autoencoder_device, interpretable_neuron_indices, dict_vecs=None):
        raise NotImplementedError("This class is an Interface")

    @staticmethod
    def get_mlp_acts_hook(autoencoder, autoencoder_device, interpretable_neuron_indices, dict_vecs=None):
        raise NotImplementedError("This class is an Interface")

    """
    Define Instances of TokenScoreRegexFilter or TokenScoreRegexFilterAverage here to parse Simulation Answer from LLM
    e.g.
    
    TokenScoreRegexFilter('".+": [0-9]?[0-9]? (.+)', '\A".+":', '[0-9]+ \(')
        .set_token_cropping(1, 2)
        .set_score_cropping(0, 2)
    """
    simulation_filters = []


"""
CodeLlama Interpretation Config
"""
@dataclass
class CodeLlamaInterpretationConfig(InterpretationConfigBase):
    prompt_generator = CodeLlamaPromptGenerator()
    include_complete_texts = False
    include_filtered_tokens = True

    mlp_sublayer_module_name = "model.layers.{}.mlp"
    attn_sublayer_module_name = "model.layers.{}.self_attn"
    mlp_activations_module_name = "model.layers.{}.mlp.act_fn"

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

