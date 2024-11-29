import torch

class PromptGeneratorBase:
    def __init__(self):
        raise NotImplementedError("Class PromptGeneratorBase is an Interface")

    def get_interpretation_prompt(self):
        raise NotImplementedError("Class PromptGeneratorBase is an Interface")

    def get_simulation_prompt(self):
        raise NotImplementedError("Class PromptGeneratorBase is an Interface")

class CodeLlamaPromptGenerator(PromptGeneratorBase):
    def __init__(self):
        super().__init__()

    def get_interpretation_prompt(self, complete_texts, tokens, activations):
        # Cast to Python-List if needed
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()

        if isinstance(activations, torch.Tensor):
            activations = activations.tolist()

        # Map activations to int
        activations = map(int, activations)

        # List all complete Text-Fragments
        user_prompt = "Neuron: \n"
        user_prompt += "The complete documents: \n\n"

        for text in complete_texts:
            user_prompt += f"{text}\n\n"

        # List all Tokens with Activations
        user_prompt += "Activations: \n"
        user_prompt += "<start>\n"

        for token, activation in zip(tokens, activations):
            user_prompt += f"* {token} \x09 {activation} \n"

        user_prompt += "<end>\n"

        # List all Tokens with zeros filtered out
        user_prompt += "Same activations, but with all zeros filtered out: \n"
        user_prompt += "<start>\n"

        for token, activation in zip(tokens, activations):
            if activation != 0:
                user_prompt += f"* {token} \x09 {activation} \n"

        user_prompt += "<end>\n"
        user_prompt += "\n \n"

        return user_prompt

    def get_simulation_prompt(self):
        pass