from dataclasses import dataclass

from sparse_autoencoders.AutoInterpretation.TokenScoreRegexFilter import *
from sparse_autoencoders.AutoInterpretation.PromptGenerator import PromptGeneratorBase

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

