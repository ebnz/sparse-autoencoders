import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sparse_autoencoders import AutoEncoder, TransformerModels, Datasets, utils

from sparse_autoencoders.AutoInterpretation.TokenScoreRegexFilter import RegexException

import pickle
from itertools import product
import re


class AutoInterpreter:
    def __init__(self, interpretation_config):
        """
        Class for AutoInterpretation of Sparse Autoencoders trained on Transformer Activations.
        :param interpretation_config: Configuration-Object defining Interpretation process
        """

        super().__init__()

        #Member Variables
        self.fragments = []
        self.mean_feature_activations = None
        self.rescaled = None

        self.interpretation_config = interpretation_config
        self.BATCH_SIZE = None
        self.NUM_TOKENS = None

        self.target_model = None

        self.interpretation_model = None

        self.autoencoder = None

        self.ds = None
        self.dl = None

        self.target_model_hook_handles = []

        self.simulation_filters = interpretation_config.simulation_filters

    def load_dataset(self, dataset_batch_size=8, num_tokens=64, partial_preload=None):
        print("Loading Dataset")

        self.BATCH_SIZE = dataset_batch_size
        self.NUM_TOKENS = num_tokens

        self.ds = Datasets.TokenizedDatasetPreload(self.interpretation_config.dataset_path, dtype=torch.int, partial_preload=partial_preload)
        self.dl = DataLoader(self.ds, batch_size=self.BATCH_SIZE, shuffle=False)    # ToDo: Consider shuffling

    def load_target_model(self, device):
        # ToDo: Maybe add float16 constraint
        print("Loading Target Model")
        self.target_model = TransformerModels.CodeLlamaModel(self.interpretation_config.target_model_name, device=device)

    @utils.ModelNeededDecorators.PARAMETER_NEEDED("target_model")
    def setup_hook_obtain_frequency_information(self):
        layer_id = self.autoencoder_config["LAYER_INDEX"]
        layer_type = self.autoencoder_config["LAYER_TYPE"]

        # Set Target-Model-Hook
        def mlp_hook(module, input, output):
            activations = output.detach().cpu()
            self.autoencoder(activations.to(self.autoencoder_device, torch.float32))

        def attn_hook(module, input, output):
            activations = output[0].detach().cpu()
            self.autoencoder(activations.to(self.autoencoder_device, torch.float32))

        def mlp_hook_inputs(model, input, output):
            activations = input[0].detach().cpu()
            self.autoencoder(activations.to(self.autoencoder_device, torch.float32))

        if layer_type == "attn_sublayer":
            self.target_model.setup_hook(attn_hook, layer_id, layer_type)
        elif layer_type == "mlp_sublayer":
            self.target_model.setup_hook(mlp_hook, layer_id, layer_type)
        elif layer_type == "mlp_activations":
            self.target_model.setup_hook(mlp_hook_inputs, layer_id, layer_type)

    @utils.ModelNeededDecorators.PARAMETER_NEEDED("target_model")
    def setup_hook_obtain_interpretation_samples(self):
        layer_id = self.autoencoder_config["LAYER_INDEX"]
        layer_type = self.autoencoder_config["LAYER_TYPE"]

        # Set Target-Model-Hook
        def mlp_hook(module, input, output):
            activations = output.detach().cpu()
            X_hat, f = self.autoencoder(activations.to(self.autoencoder_device, torch.float32))
            # Only select the Features with log-Frequency between boundaries set in function obtain_interpretation_samples
            dict_vec = f.detach().cpu()[::, ::, self.interpretable_neuron_indices]
            self.dict_vecs.append(dict_vec)

        def attn_hook(module, input, output):
            activations = output[0].detach().cpu()
            X_hat, f = self.autoencoder(activations.to(self.autoencoder_device, torch.float32))
            # Only select the Features with log-Frequency between boundaries set in function obtain_interpretation_samples
            dict_vec = f.detach().cpu()[::, ::, self.interpretable_neuron_indices]
            self.dict_vecs.append(dict_vec)

        def mlp_hook_inputs(model, input, output):
            activations = input[0].detach().cpu()
            X_hat, f = self.autoencoder(activations.to(self.autoencoder_device, torch.float32))
            # Only select the Features with log-Frequency between boundaries set in function obtain_interpretation_samples
            dict_vec = f.detach().cpu()[::, ::, self.interpretable_neuron_indices]
            self.dict_vecs.append(dict_vec)

        if layer_type == "attn_sublayer":
            self.target_model.setup_hook(attn_hook, layer_id, layer_type)
        elif layer_type == "mlp_sublayer":
            self.target_model.setup_hook(mlp_hook, layer_id, layer_type)
        elif layer_type == "mlp_activations":
            self.target_model.setup_hook(mlp_hook_inputs, layer_id, layer_type)

    def load_interpretation_model(self, device):
        # ToDo: Maybe add float16 constraint
        print("Loading Interpretation Model")
        self.interpretation_model = TransformerModels.CodeLlamaModel(self.interpretation_config.interpretation_model_name, device=device)

    def load_interpretation_model_deepspeed(self, num_gpus=4):
        print("Loading Interpretation Model")
        self.interpretation_model = TransformerModels.CodeLlamaModelDeepspeed(self.interpretation_config.interpretation_model_name, num_gpus)

    def load_autoencoder(self, device):
        print("Loading AutoEncoder")
        self.autoencoder_device = device

        with open(self.interpretation_config.autoencoder_path, "rb") as f:
            self.autoencoder_config = pickle.load(f)
        self.autoencoder = AutoEncoder.load_model_from_config(self.autoencoder_config)
        self.autoencoder.to(self.autoencoder_device)

        self.dict_vecs = []

    @utils.ModelNeededDecorators.PARAMETER_NEEDED("target_model")
    @utils.ModelNeededDecorators.PARAMETER_NEEDED("ds")
    def obtain_interpretation_samples(self, num_batches, log_freq_lower=-4, log_freq_upper=-3):
        """
        Run num_batches Batches of data through the LLM to do AutoInterpretation on the later.
        Only Features with log-Frequency in between log_freq_lower and log_freq_upper are interpreted.
        :param num_batches: Amount of Batches to run through the LLM
        :param log_freq_lower: Lower boundary for log-Feature-Frequency
        :param log_freq_upper: Upper boundary for log-Feature-Frequency
        :return: Nothing
        """

        self.setup_hook_obtain_frequency_information()
        self.autoencoder.enable_checkpointing(self.autoencoder_config["LEARNING_RATE"],
                                              self.autoencoder_config["L1_COEFFICIENT"],
                                              self.autoencoder_config["BATCH_SIZE_TRAINING"],
                                              self.autoencoder_config["LAYER_TYPE"],
                                              self.autoencoder_config["LAYER_INDEX"],
                                              "", "", 999_999_999)

        # Number of Transformer-Runs for Frequency-Information Obtaining
        num_freq_info_runs = int(100 * (10 ** (-1 * log_freq_lower)) / (self.BATCH_SIZE * self.NUM_TOKENS))

        print("Obtaining Feature Frequency Information")
        with tqdm(total=num_freq_info_runs) as pbar:
            for idx, input_ids in enumerate(self.dl):
                cropped_input_ids = input_ids[::, :self.NUM_TOKENS]     #Crop sequences of input ids to length NUM_TOKENS
                input_ids_cuda = cropped_input_ids.to(self.target_model.device)     #Send to proper device

                #Run Model
                self.target_model.run_model_until_layer(input_ids_cuda, self.autoencoder_config["LAYER_INDEX"])
                #Update Progress-Bar and stop progress, if amount of batches reached
                pbar.update(1)
                if idx >= num_freq_info_runs - 1:
                    break

        # Calculate Neurons in Histogram-Range
        self.activation_counts_log10, _ = self.autoencoder.calculate_activation_counts(
            log10=True,
            num_samples=num_batches * self.BATCH_SIZE * self.NUM_TOKENS
        )
        interpretable_neuron_mask_raw = torch.bitwise_and(self.activation_counts_log10 > log_freq_lower,
                                                          self.activation_counts_log10 < log_freq_upper)
        self.interpretable_neuron_mask = torch.bitwise_and(interpretable_neuron_mask_raw,
                                                           torch.isfinite(self.activation_counts_log10))
        self.interpretable_neuron_indices = torch.where(self.interpretable_neuron_mask)[0]

        print(self.autoencoder.count_f_checkpoint)
        print(self.activation_counts_log10)
        print(torch.sum(self.interpretable_neuron_indices))


        print("Obtaining Interpretation Samples")
        self.setup_hook_obtain_interpretation_samples()
        with tqdm(total=num_batches) as pbar:
            for idx, input_ids in enumerate(self.dl):
                cropped_input_ids = input_ids[::, :self.NUM_TOKENS]     #Crop sequences of input ids to length NUM_TOKENS
                input_ids_cuda = cropped_input_ids.to(self.target_model.device)     #Send to proper device

                #Run Model
                self.target_model.run_model_until_layer(input_ids_cuda, self.autoencoder_config["LAYER_INDEX"])

                #Append each processed Batch-Index to self.fragments
                for i in range(self.BATCH_SIZE):
                    # Only append index. Real Fragment-Text is saved in Dataset (don't reload bc. of shuffle)
                    self.fragments.append([idx, i])
                    #self.fragments.append(cropped_input_ids[i, ::].unsqueeze(0))

                #Update Progress-Bar and stop progress, if amount of batches reached
                pbar.update(1)
                if idx == num_batches - 1:
                    break

        #self.dict_vecs is set in the Forward Hook of the Model. This Hook is set up in the Constructor of this class
        #Concatenate all Dict-Vec-Batches to a single large list
        acts = torch.cat(self.dict_vecs, dim=0).detach().cpu().to(torch.float16)

        #Rescale Dict-Feature-Activations 'acts' to range 0 - 10
        self.mins = torch.min(torch.min(acts, dim=1).values, dim=0).values
        minus_min = acts - self.mins
        self.maxs = torch.max(torch.max(minus_min, dim=1).values, dim=0).values
        div_by_max = torch.div(minus_min, self.maxs)
        self.rescaled = 10 * torch.where(div_by_max.isnan(), 0, div_by_max)

        #Calculate the mean activation of a feature for each text fragment
        #ToDo: self.rescaled has shape [num_text_fragments, num_tokens, size_of_dict_vec]
        #If 10 Batches with Batch Size 8, 64 Tokens per Text Fragment and 5000 Features have log-Frequency in boundaries,
        #then the shape of self.rescaled is [10 * 8, 64, 5000]
        self.mean_feature_activations = torch.mean(self.rescaled, dim=1)


    @utils.ModelNeededDecorators.PARAMETER_NEEDED("rescaled")
    def save_interpretation_samples(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                "fragments": self.fragments,
                "rescaled": self.rescaled,
                "mean_feature_activations": self.mean_feature_activations,
                "interpretable_neuron_indices": self.interpretable_neuron_indices
            }, f)

    def load_interpretation_samples(self, path):
        print("Loading Interpretation-Samples")

        with open(path, "rb") as f:
            obj = pickle.load(f)
            self.fragments = obj["fragments"]
            self.rescaled = obj["rescaled"]
            self.mean_feature_activations = obj["mean_feature_activations"]
            #self.activation_counts_log10 = obj["act_counts_log10"]

            #self.interpretable_neuron_mask = torch.bitwise_and(self.activation_counts_log10 > -4,
            #                                                   self.activation_counts_log10 < -3)
            self.interpretable_neuron_indices = obj["interpretable_neuron_indices"]

    @utils.ModelNeededDecorators.PARAMETER_NEEDED("rescaled")
    def get_fragment(self, feature_index, i_top_fragments, return_activations=False):
        """
        Returns a Text Fragment on which the given feature_index activates strongly.
        :param feature_index: Feature index on which the returned Text Fragment activates strongly
        :param i_top_fragments: Rank of activation strength of the returned Text Fragment. 0 -> Most strong activation
        :param return_activations: Whether to also return the rescaled token activations to the Text Fragment
        :return: The Text Fragment on which the given feature_index activates strongly
        """

        #Compute the fragments on which the given Neuron activates the most
        feature_activations = self.mean_feature_activations[::, feature_index]  #Feature Activations on all Text Fragments
        top_fragment_ids = feature_activations.topk(i_top_fragments + 1, largest=True).indices  #Compute top-k most activating Text Fragments
        tf_id = top_fragment_ids[i_top_fragments]   #Text-Fragment-ID, representing the position of the Text Fragment in self.fragments

        #Return procedure
        fragment_idx = self.fragments[tf_id]
        fragment = self.ds[self.BATCH_SIZE * fragment_idx[0] + fragment_idx[1]]

        # ToDo: Remove
        #print(f"Used fragment_idx: {fragment_idx}")
        #text = self.target_model.tokenizer.batch_decode([fragment])[0]
        #print(text)

        if return_activations:
            rescaled_per_token_feature_acts = self.rescaled[tf_id, :, feature_index]
            return fragment, rescaled_per_token_feature_acts

        return fragment

    def generate_interpretation_prompt(self, feature_index, num_top_fragments):
        """
        Generates an Interpretation Prompt to infer an Explanation of a Feature.
        :param feature_index: The index of the feature to infer an Interpretation for
        :param num_top_fragments: Number of text fragments used for obtaining the Interpretation
        :return: User Prompt
        """

        user_prompt = "Neuron: \n"
        user_prompt += "The complete documents: \n\n"

        for i_top_fragments in range(num_top_fragments):
            fragment, rescaled_per_token_feature_acts = self.get_fragment(
                feature_index,
                i_top_fragments,
                return_activations=True
            )

            text = self.interpretation_model.tokenizer.batch_decode([fragment])[0]
            user_prompt += f"{text}\n\n"

        user_prompt += "Activations: \n"
        user_prompt += "<start>\n"

        for i_top_fragments in range(num_top_fragments):
            fragment, rescaled_per_token_feature_acts = self.get_fragment(
                feature_index,
                i_top_fragments,
                return_activations=True
            )

            for i in range(min([len(fragment), self.NUM_TOKENS])):
                token = self.interpretation_model.tokenizer.convert_ids_to_tokens([fragment[i]])[0]
                activation = int(rescaled_per_token_feature_acts[i])
                user_prompt += f"* {token} \x09 {activation} \n"

        user_prompt += "<end>\n"
        user_prompt += "Same activations, but with all zeros filtered out: \n"
        user_prompt += "<start>\n"

        for i_top_fragments in range(num_top_fragments):
            fragment, rescaled_per_token_feature_acts = self.get_fragment(
                feature_index,
                i_top_fragments,
                return_activations=True
                )
            for i in range(min([len(fragment), self.NUM_TOKENS])):
                if int(rescaled_per_token_feature_acts[i]) != 0:
                    token = self.interpretation_model.tokenizer.convert_ids_to_tokens([fragment[i]])[0]
                    activation = int(rescaled_per_token_feature_acts[i])
                    user_prompt += f"* {token} \x09 {activation} \n"

        user_prompt += "<end>\n"
        user_prompt += "\n \n"

        return user_prompt

    @utils.ModelNeededDecorators.PARAMETER_NEEDED("interpretation_model")
    def get_explanation(self, user_prompt, raw=False):
        """
        Generate the Explanation for a specific Feature.
        :param user_prompt: User prompt to infer the Interpretation of a Neuron
        :param raw: Whether to return the Explanation including the input or only the Explanation
        :return: Explanation of the Neuron's behavior
        """

        # Generate Prompt from user_prompt and system_prompt
        prompt = f"[INST]<<SYS>>{self.interpretation_config.interpretation_system_prompt}<</SYS>>\n{user_prompt}[/INST]"

        # Generate Explanation, raw_explanation consists of complete answer of LLM including initial prompt
        raw_explanation = self.interpretation_model.generate_on_prompt(prompt, max_new_tokens=100)

        explanation = (raw_explanation.split("[/INST]")[-1]
                       .replace('</s>', '')
                       .replace('<s>', '')
                       .replace('[/INST]', '')
                       .replace('[INST]', '')
                       .strip())

        return raw_explanation if raw else explanation

    def generate_simulation_prompt(self, feature_index, i_top_fragments, explanation):
        """
        Generates a Simulation Prompt for the Interpretation Model.
        :param feature_index: The Feature which should be simulated
        :param i_top_fragments: The Text Fragment which should be simulated. 0 -> most activating fragment
        :param explanation: The Explanation of the Neuron generated by the Interpretation Model
        :return: Generated User Prompt
        """

        fragment = self.get_fragment(feature_index, i_top_fragments)

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
Explanation of neuron 2 behavior: {explanation}
Activations: 
<start>
'''

        for i in range(len(fragment)):
            token = self.interpretation_model.tokenizer.convert_ids_to_tokens([fragment[i]])[0]
            user_prompt += f"* {token} \x09 <unknown> \n"

        user_prompt += ("<end>\n\nInfer the unknown activations of Neuron 2 as a list of numerical values "
                        "ranging from 0-10. One value per token.")

        return user_prompt

    @utils.ModelNeededDecorators.PARAMETER_NEEDED("interpretation_model")
    def get_simulation(self, user_prompt, return_raw_simulation=False):
        """
        Runs a simulation prompt on the Interpretation Model and retrieves the inferred activations.
        :param user_prompt: User Prompt which infers the Neuron activations
        :param return_raw_simulation: Whether to return the raw Simulation-Output from the Interpretation-LLM
        :return: List of tuples containing the inferred neuron activations
        """

        # Generate Prompt from user_prompt and system_prompt
        prompt = f"[INST]<<SYS>>{self.interpretation_config.simulation_system_prompt}<</SYS>>\n{user_prompt}[/INST]"

        # Generate Explanation, raw_explanation consists of complete answer of LLM including initial prompt
        raw_simulation = self.interpretation_model.generate_on_prompt(prompt, max_new_tokens=1000)

        simulation = (raw_simulation.split("[/INST]")[-1]
                      .replace('</s>', '')
                      .replace('<s>', '')
                      .replace('[/INST]', '')
                      .replace('[INST]', ''))

        kv_dict = {}

        lines = list(simulation.split("\n"))

        replacement_dict = {
            "\n": "",
            "\t": "",
            "  ": " ",
        }

        for raw_line in lines:
            # Apply Replacements in <replacement_dict> and remove leading Asterisk in each line
            line = apply_dict_replacement(raw_line, replacement_dict)
            line = remove_leading_asterisk(line)

            for regex_filter in self.simulation_filters:
                # Check each provided TokenScoreRegexFilter for match
                does_match = regex_filter.match(line)
                if not does_match:
                    # Skip Filter if no match
                    continue

                try:
                    # If Match, extract Token and Score and append to kv_dict
                    token = regex_filter.get_token(line)
                    score = regex_filter.get_score(line)
                except RegexException as e:
                    print(e)
                    continue

                # Replace unwanted Characters in the Token
                token = apply_dict_replacement(token, self.interpretation_config.token_replacement_chars)

                kv_dict[token] = score

                # If Filter Matches, skip all following Filters and proceed with next Line
                break

        # Rescale all Token Scores to 0-10 (Sometimes the Interpretation-LLM outputs only binary/0-1)
        maximum = max(kv_dict.values())
        minimum = min(kv_dict.values())

        kv_dict_rescaled = {key: (10 / (maximum - minimum)) * (kv_dict[key] - minimum) for key in kv_dict}

        if return_raw_simulation:
            return kv_dict_rescaled, raw_simulation
        return kv_dict_rescaled

    @utils.ModelNeededDecorators.PARAMETER_NEEDED("rescaled")
    def generate_ground_truth_scores(self, feature_index, i_top_fragments):
        """
        Generates the Ground Truth Activation-Sample for a given Text Sample ID.
        :param feature_index: The Feature Index for which the Activation-Sample should be sample
        :param i_top_fragments: The Text-Fragment-ID for which the Activation should be sampled
        :return: Ground Truth Activation-Sample
        """

        fragment, rescaled_per_token_feature_acts = self.get_fragment(
            feature_index,
            i_top_fragments,
            return_activations=True
        )

        kv_dict = {}

        for i in range(len(fragment)):
            key = f"{self.interpretation_model.tokenizer.convert_ids_to_tokens([fragment[i]])[0]}"
            value = int(rescaled_per_token_feature_acts[i])

            # Replace unwanted Characters in the Token
            key = apply_dict_replacement(key, self.interpretation_config.token_replacement_chars)

            kv_dict[key] = value

        return kv_dict


def calculate_correlation_from_kv_dict(kv_dict_gt, kv_dict_simulated):
    """
    Calculates the correlation between two Activation-Samples.
    :param kv_dict_gt: Ground Truth Activation Sample
    :param kv_dict_simulated: Simulated Activation Sample
    :return: Correlation Score of Activation Samples
    """

    datapoints = []
    for key_gt, key_simulated in product(kv_dict_gt, kv_dict_simulated):
        if key_gt == key_simulated:
            datapoints.append([kv_dict_gt[key_gt], kv_dict_simulated[key_simulated]])

    datapoints_tensor = torch.Tensor(datapoints).T
    corr_mat = torch.corrcoef(datapoints_tensor)

    return float(corr_mat[0, 1])


"""
Simulation Helper-Functions
"""
def apply_dict_replacement(inp_str, replacement_dict):
    out_str = inp_str

    for key in replacement_dict.keys():
        out_str = out_str.replace(key, replacement_dict[key])

    return out_str


def remove_leading_asterisk(line):
    if line.startswith("* "):
        return line[2:]
    return line
