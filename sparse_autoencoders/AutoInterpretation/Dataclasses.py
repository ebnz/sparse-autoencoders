from dataclasses import dataclass

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
