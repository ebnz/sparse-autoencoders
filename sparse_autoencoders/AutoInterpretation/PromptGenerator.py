import torch
from sparse_autoencoders.utils import apply_dict_replacement


class PromptGeneratorBase:
    def __init__(self):
        """
        An interface, that handles Generation of Interpretation/Simulation Prompts for AutoInterpretation.
        May be defined for each Language Model individually.
        """
        self.interpretation_system_prompt = ""
        self.simulation_system_prompt = ""

        # Replacements to be done in tokens (e.g. "_ing" -> "ing")
        self.token_replacements = {}

        # Replacements to be done on LLM-Outputs (e.g. "</s>" -> "")
        self.output_replacements = {}

    def get_interpretation_prompt(self, complete_texts, tokens, activations,
                                  include_complete_texts=True, include_filtered_tokens=True):
        """
        Generates the Interpretation Prompt to infer an explanation of the Token Activations.
        :rtype: str
        :type complete_texts: list[str]
        :type tokens: list[str]
        :type activations: list[int]
        :param complete_texts: Complete documents for the Interpretation Prompt
        :param tokens: Tokens for the Interpretation Prompt
        :param activations: Activations to the Tokens for the Interpretation Prompt
        :param include_complete_texts: Whether to include the complete texts into the Interpretation Prompt
        :param include_filtered_tokens: Whether to include a section of all non-zero activating Tokens into the Prompt
        """
        raise NotImplementedError("Class PromptGeneratorBase is an Interface")

    def get_interpretation_system_prompt(self):
        """
        Returns the System Prompt for Interpretation.
        :rtype: str
        :return: System Prompt
        """
        return self.interpretation_system_prompt

    def get_simulation_prompt(self, tokens, explanation):
        """
        Generates the Simulation Prompt for Simulation of a Neuron.
        :rtype: str
        :type tokens: list[str]
        :type explanation: str
        :param tokens: Tokens for the Simulation Prompt
        :param explanation: Explanation, generated in the Interpretation-Process
        """
        raise NotImplementedError("Class PromptGeneratorBase is an Interface")

    def get_simulation_system_prompt(self):
        """
        Returns the System Prompt for a Simulation.
        :rtype: str
        :return: System Prompt
        """
        return self.simulation_system_prompt

    def extract_llm_output(self, raw_output):
        """
        Extracts the newly generated part from a LLM-Output. Separates the newly generated output from a LLM from the
        Union of Prompt and newly generated output.
        :rtype: str
        :type raw_output: str
        :param raw_output: Newly generated part of the LLM-Output
        :return: Newly generated LLM Output
        """
        raise NotImplementedError("Class PromptGeneratorBase is an Interface")

    def replace_special_tokens(self, raw_output):
        """
        Removes the Special Tokens from a String.
        :rtype: str
        :type raw_output: str
        :param raw_output: Input, on which the special Tokens are removed
        :return: Parameter raw_output, where special tokens are removed
        """
        raise NotImplementedError("Class PromptGeneratorBase is an Interface")


class CodeLlamaPromptGenerator(PromptGeneratorBase):
    def __init__(self):
        """
        PromptGenerator-Class, modified for CodeLlama LLMs.
        """
        super().__init__()

        # Prompts from https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html
        self.interpretation_system_prompt = (
            "We're studying neurons in a neural network. Each neuron looks for some particular "
            "thing in a short document. Look at the parts of the document the neuron "
            "activates for and summarize in a single sentence what the neuron is looking for. "
            "Don't list examples of words. \n The activation format is token<tab>activation. "
            "Activation values range from 0 to 10. A neuron finding what it's looking for is "
            "represented by a non-zero activation value. The higher the activation value, "
            "the stronger the match.")

        self.simulation_system_prompt = (
            "We're studying neurons in a neural network. "
            "Each neuron looks for some particular thing in a short document. "
            "Look at an explanation of what the neuron does, "
            "and try to predict its activations on each particular token. \n The activation "
            "format is token<tab>activation, and activations range from 0 to 10. "
            "A neuron finding what it's looking for is represented by a non-zero activation "
            "value. The higher the activation value, the stronger the match. "
            "Most activations will be 0.")

        self.token_replacements = {
            "▁": ""
        }

        self.output_replacements = {
            "</s>": "",
            "<s>": "",
            "[/INST]": "",
            "[INST]": ""
        }

    def get_interpretation_prompt(self, complete_texts, tokens, activations,
                                  include_complete_texts=True, include_filtered_tokens=True):
        # Cast to Python-List if needed
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()

        if isinstance(activations, torch.Tensor):
            activations = activations.tolist()

        # Map activations to int
        activations = list(map(int, activations))

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
            if activation > 0:
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

    def extract_llm_output(self, raw_output):
        generated_text = raw_output.split("[/INST]")[-1].strip()
        return self.replace_special_tokens(generated_text)

    def replace_special_tokens(self, raw_output):
        return apply_dict_replacement(raw_output, self.output_replacements)
