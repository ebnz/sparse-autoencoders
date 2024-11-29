from dataclasses import dataclass

from sparse_autoencoders.AutoInterpretation.TokenScoreRegexFilter import *

"""
Dataclasses for Interpretation
"""
@dataclass
class InterpretationConfig:
    dataset_path: str   #"/nfs/data/students/ebenz_bsc2024/multip_stack_tokenized"

    target_model_name: str

    interpretation_model_name: str

    autoencoder_path: str

    interpretation_system_prompt = ("We're studying neurons in a neural network. Each neuron looks for some particular "
                                    "thing in a short document. Look at the parts of the document the neuron "
                                    "activates for and summarize in a single sentence what the neuron is looking for. "
                                    "Don't list examples of words. \n The activation format is token<tab>activation. "
                                    "Activation values range from 0 to 10. A neuron finding what it's looking for is "
                                    "represented by a non-zero activation value. The higher the activation value, "
                                    "the stronger the match.")

    simulation_system_prompt = ("We're studying neurons in a neural network. Each neuron looks for some particular "
                                "thing in a short document. Look at an explanation of what the neuron does, "
                                "and try to predict its activations on each particular token. \n The activation "
                                "format is token<tab>activation, and activations range from 0 to 10. A neuron finding what it's looking for is represented by a non-zero activation value. The higher the activation value, the stronger the match. Most activations will be 0.")

    token_replacement_chars = {
        "▁": ""
    }

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