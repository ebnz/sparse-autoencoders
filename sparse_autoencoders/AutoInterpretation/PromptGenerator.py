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

    def get_simulation_prompt(self, tokens, explanation):
        # Cast to Python-List if needed
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()

        # Example for the LLM
        user_prompt = f'''
Neuron 1: 
Explanation of neuron 1 behavior: the main thing this neuron does is find phrases related to community. 
Activations: 
<start>
* the \x09 0
* sense \x09 0
* of \x09 7
* together \x09 7
* ness \x09 7
* in \x09 0
* our \x09 0
* town \x09 1
* is \x09 0
* strong \x09 0
* . \x09 0
<end>

Neuron 2: 
Explanation of neuron 2 behavior: '''

        # Append Explanation of the Neuron
        user_prompt += f"{explanation}\n"

        # Append Activations
        user_prompt += '''
Activations: 
<start>
'''

        for token in tokens:
            user_prompt += f"* {token} \x09 <unknown> \n"

        user_prompt += ("<end>\n\nInfer the unknown activations of Neuron 2 as a list of numerical values "
                        "ranging from 0-10. One value per token.")

        return user_prompt
