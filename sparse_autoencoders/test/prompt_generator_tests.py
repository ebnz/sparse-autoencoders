import unittest

from sparse_autoencoders.AutoInterpretation.PromptGenerator import CodeLlamaPromptGenerator


class MyPromptGenerator(CodeLlamaPromptGenerator):
    def __init__(self):
        super().__init__()

        self.interpretation_system_prompt = "my_int_sysprompt"

        self.simulation_system_prompt = "my_sim_sysprompt"

        self.token_replacements = {
            "to_replace": ""
        }

        self.output_replacements = {
            "[/INST]": "",
            "[INST]": ""
        }


class PromptGeneratorTest(unittest.TestCase):
    prompt_generator = MyPromptGenerator()

    complete_texts = ["text_1", "text_2"]
    tokens = ["a", "b", "c"]
    activations = [6, 9, 0]

    def test_get_int_prompt_normal(self):
        int_prompt = ("Neuron: \nThe complete documents: \n\ntext_1\n\ntext_2\n\nActivations: \n"
                      "<start>\n* a \x09 6 \n* b \x09 9 \n* c \x09 0 \n<end>\nSame activations, "
                      "but with all zeros filtered out: \n<start>\n* a \x09 6 \n* b \x09 9 \n<end>\n\n \n")

        self.assertEqual(
            int_prompt,
            self.prompt_generator.get_interpretation_prompt(
                self.complete_texts,
                self.tokens,
                self.activations,
                include_complete_texts=True,
                include_filtered_tokens=True
            )
        )

    def test_get_int_prompt_no_inputs(self):
        int_prompt = ("Neuron: \nThe complete documents: \n\nActivations: \n<start>\n<end>\n"
                      "Same activations, but with all zeros filtered out: \n<start>\n<end>\n\n \n")

        self.assertEqual(
            int_prompt,
            self.prompt_generator.get_interpretation_prompt(
                [],
                [],
                [],
                include_complete_texts=True,
                include_filtered_tokens=True
            )
        )

    def test_get_int_prompt_no_filtered_tokens(self):
        int_prompt = ("Neuron: \nThe complete documents: \n\ntext_1\n\ntext_2\n\nActivations: \n"
                      "<start>\n* a \x09 6 \n* b \x09 9 \n* c \x09 0 \n<end>\n\n \n")

        self.assertEqual(
            int_prompt,
            self.prompt_generator.get_interpretation_prompt(
                self.complete_texts,
                self.tokens,
                self.activations,
                include_complete_texts=True,
                include_filtered_tokens=False
            )
        )

    def test_get_int_prompt_no_comp_texts(self):
        int_prompt = ("Neuron: \nActivations: \n<start>\n* a \x09 6 \n* b \x09 9 \n* c \x09 0 \n<end>\nSame "
                      "activations, but with all zeros filtered out: \n<start>\n* a \x09 6 \n* b \x09 9 \n<end>\n\n \n")

        self.assertEqual(
            int_prompt,
            self.prompt_generator.get_interpretation_prompt(
                self.complete_texts,
                self.tokens,
                self.activations,
                include_complete_texts=False,
                include_filtered_tokens=True
            )
        )

    def text_get_sim_prompt_normal(self):
        my_sim_prompt = '''
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
Explanation of neuron 2 behavior: 

my_explanation\n

Activations: 
<start>
* a \x09 <unknown> \n
* b \x09 <unknown> \n
* c \x09 <unknown> \n
<end>\n\nInfer the unknown activations of Neuron 2 as a list of numerical values ranging from 0-10. One value per token.'''

        self.assertEqual(my_sim_prompt, self.prompt_generator.get_simulation_prompt(self.tokens, "my_explanation"))

    def test_special_tokens(self):
        self.assertEqual(
            "abcd",
            self.prompt_generator.replace_special_tokens(
                "[INST][/INST]abcd"
            )
        )

    def test_extract_llm_output(self):
        self.assertEqual(
            "blablabla_answer",
            self.prompt_generator.extract_llm_output(
                "[INST]INSTRUCTION[/INST]blablabla_answer[INST]"
            )
        )


if __name__ == '__main__':
    unittest.main()
