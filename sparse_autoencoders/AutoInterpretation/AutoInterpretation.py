import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sparse_autoencoders

from sparse_autoencoders import utils
from .InterpretationConfig import InterpretationConfigBase
from sparse_autoencoders.AutoInterpretation.TokenScoreRegexFilter import RegexException

import pickle


class AutoInterpreter:
    def __init__(self, interpretation_config: InterpretationConfigBase):
        """
        Class for AutoInterpretation of Sparse Autoencoders trained on Transformer Activations.
        :param interpretation_config: Configuration-Object defining Interpretation process
        """

        super().__init__()

        # Member Variables
        self.interpretation_config = interpretation_config
        self.simulation_filters = interpretation_config.simulation_filters
        self.dict_vecs = []

        # Dataset
        self.ds = None
        self.dl = None
        self.BATCH_SIZE = None
        self.NUM_TOKENS = None

        # LLM-Models
        self.target_model = None
        self.interpretation_model = None

        self.target_model_hook_handles = []

        # Feature Activations
        self.fragments = []
        self.mean_feature_activations = None
        self.rescaled_feature_activations = None

        # Feature Frequency Information
        self.activation_counts_log10 = None
        self.interpretable_neuron_indices = None
        self.interpretable_neuron_mask = None

        # Autoencoder
        self.autoencoder_config = None
        self.autoencoder_device = None
        self.autoencoder = None

    """
    Model-Loading Functions
    """
    def load_dataset(self, dataset_batch_size=8, num_tokens=64):
        """
        Loads the Dataset specified in the InterpretationConfig
        :param dataset_batch_size: Batchsize, the Dataset is fed to the LLM
        :param num_tokens: Number of Tokens used from each Dataset entry
        """
        print("Loading Dataset")

        self.BATCH_SIZE = dataset_batch_size
        self.NUM_TOKENS = num_tokens

        self.ds = sparse_autoencoders.Datasets.TokenizedDatasetPreload(
            self.interpretation_config.dataset_path,
            dtype=torch.int
        )

        self.dl = DataLoader(self.ds, batch_size=self.BATCH_SIZE, shuffle=False)

    def load_target_model(self, device):
        """
        Loads the Target-LLM
        :param device: Device, on which the Model is offloaded to
        """
        print("Loading Target Model")
        self.target_model = sparse_autoencoders.CodeLlamaModel(
            self.interpretation_config.target_model_name,
            device=device
        )

    def load_interpretation_model(self, device):
        """
        Loads the Interpretation-LLM.
        :param device: Device, on which the Model is offloaded to
        """
        print("Loading Interpretation Model")
        self.interpretation_model = sparse_autoencoders.CodeLlamaModel(
            self.interpretation_config.interpretation_model_name,
            device=device
        )

    def load_interpretation_model_deepspeed(self, num_gpus=4):
        """
        Load the Interpretation-LLM via DeepSpeed on multiple GPUs.
        :param num_gpus: Number of GPUs to allocate
        """
        print("Loading Interpretation Model")
        self.interpretation_model = sparse_autoencoders.CodeLlamaModelDeepspeed(
            self.interpretation_config.interpretation_model_name,
            num_gpus
        )

    def load_autoencoder(self, device):
        """
        Load Autoencoder.
        :param device: Device, on which the Autoencoder is offloaded to
        """
        print("Loading AutoEncoder")
        self.autoencoder_device = device

        with open(self.interpretation_config.autoencoder_path, "rb") as f:
            self.autoencoder_config = pickle.load(f)
        self.autoencoder = sparse_autoencoders.AutoEncoder.load_model_from_config(self.autoencoder_config)
        self.autoencoder.to(self.autoencoder_device)

        self.interpretable_neuron_indices = torch.arange(0, self.autoencoder_config["DICT_VEC_SIZE"], 1)
        self.dict_vecs = []

    """
    Preparation Functions
    """
    @utils.ModelNeededDecorators.PARAMETER_NEEDED("target_model")
    @utils.ModelNeededDecorators.PARAMETER_NEEDED("autoencoder")
    def setup_sae_hook(self, dict_vecs: list | None = None):
        """
        Sets the Hook to obtain the Activation Vectors from the LLM.
        Activation Vectors automatically processed to Dictionary Vectors by the Autoencoder and saved in dict_vecs.
        :param dict_vecs: Destination to save the
        """
        layer_id = self.autoencoder_config["LAYER_INDEX"]
        layer_type = self.autoencoder_config["LAYER_TYPE"]

        if layer_type == "attn_sublayer":
            hook = self.interpretation_config.get_mlp_hook(self.autoencoder, self.autoencoder_device,
                                                           self.interpretable_neuron_indices, dict_vecs=dict_vecs)
            self.target_model.setup_hook(hook, self.interpretation_config.attn_sublayer_module_name.format(layer_id))
        elif layer_type == "mlp_sublayer":
            hook = self.interpretation_config.get_attn_hook(self.autoencoder, self.autoencoder_device,
                                                            self.interpretable_neuron_indices, dict_vecs=dict_vecs)
            self.target_model.setup_hook(hook, self.interpretation_config.mlp_sublayer_module_name.format(layer_id))
        elif layer_type == "mlp_activations":
            hook = self.interpretation_config.get_mlp_acts_hook(self.autoencoder, self.autoencoder_device,
                                                                self.interpretable_neuron_indices, dict_vecs=dict_vecs)
            self.target_model.setup_hook(hook, self.interpretation_config.mlp_activations_module_name.format(layer_id))
        else:
            raise ValueError(f"layer_type <{layer_type}> not recognized")

    @utils.ModelNeededDecorators.PARAMETER_NEEDED("target_model")
    @utils.ModelNeededDecorators.PARAMETER_NEEDED("ds")
    def obtain_interpretation_samples(self, num_batches, log_freq_lower=-2, log_freq_upper=-0.1):
        """
        1. Run data through the Target-LLM to obtain Feature-Frequency Information.
        2. Run num_batches Batches of data through the Target-LLM to do AutoInterpretation on.
        Only obtain Features with log-Frequency in between log_freq_lower and log_freq_upper
        :param num_batches: Amount of Batches to run through the Target-LLM
        :param log_freq_lower: Lower boundary for log-Feature-Frequency
        :param log_freq_upper: Upper boundary for log-Feature-Frequency
        """

        # Setup Hooks for Activation-Vector Sampling, Enable Checkpoints for Feature-Frequency Collection
        self.setup_sae_hook()
        self.autoencoder.enable_checkpointing(self.autoencoder_config["LEARNING_RATE"],
                                              self.autoencoder_config["L1_COEFFICIENT"],
                                              self.autoencoder_config["BATCH_SIZE_TRAINING"],
                                              self.autoencoder_config["LAYER_TYPE"],
                                              self.autoencoder_config["LAYER_INDEX"],
                                              "", "", 999_999_999)

        # Number of Transformer-Runs for Frequency-Information Obtaining
        # Inverse Probability for least probable neuron to activate
        inv_prob_least_probable_neuron = 10 ** (-1 * log_freq_lower)
        # Use 100 * Inverse Probability for least probable neuron to activate as amount of LLM-Runs
        # This ensures, that each Neuron activated at least a specific times for Feature Freq. Obtaining
        num_freq_info_runs = int(100 * inv_prob_least_probable_neuron / (self.BATCH_SIZE * self.NUM_TOKENS))

        print("Obtaining Feature Frequency Information")
        with tqdm(total=num_freq_info_runs) as pbar:
            for idx, input_ids in enumerate(self.dl):
                # Crop sequences of input ids to length NUM_TOKENS, Send to proper device
                cropped_input_ids = input_ids[::, :self.NUM_TOKENS]
                input_ids_cuda = cropped_input_ids.to(self.target_model.device)

                # Run Model, Update Progress-Bar and stop progress, if amount of batches reached
                self.target_model.run_model_until_layer(input_ids_cuda, self.autoencoder_config["LAYER_INDEX"])
                pbar.update(1)
                if idx >= num_freq_info_runs - 1:
                    break

        # Calculate Activation Frequency of each Neuron
        self.activation_counts_log10, _ = self.autoencoder.calculate_activation_counts(
            log10=True,
            num_samples=num_freq_info_runs * self.BATCH_SIZE * self.NUM_TOKENS
        )

        # Calculate Neurons in specific Activation-Frequency-Range
        interpretable_neuron_mask_raw = torch.bitwise_and(self.activation_counts_log10 > log_freq_lower,
                                                          self.activation_counts_log10 < log_freq_upper)
        # Filter out inf's/-inf's
        self.interpretable_neuron_mask = torch.bitwise_and(interpretable_neuron_mask_raw,
                                                           torch.isfinite(self.activation_counts_log10))
        # Obtain Feature-Indices
        self.interpretable_neuron_indices = torch.where(self.interpretable_neuron_mask)[0]

        # Now finally obtain Feature-Activations, only for Features in frequency-range
        print("Obtaining Interpretation Samples")
        self.setup_sae_hook(dict_vecs=self.dict_vecs)
        with tqdm(total=num_batches) as pbar:
            for idx, input_ids in enumerate(self.dl):
                # Crop sequences of input ids to length NUM_TOKENS, Send to proper device
                cropped_input_ids = input_ids[::, :self.NUM_TOKENS]
                input_ids_cuda = cropped_input_ids.to(self.target_model.device)

                # Run Model, Update Progress-Bar and stop progress, if amount of batches reached
                self.target_model.run_model_until_layer(input_ids_cuda, self.autoencoder_config["LAYER_INDEX"])

                # Only append index. Real Fragment-Text is saved in Dataset
                for i in range(self.BATCH_SIZE):
                    # [batch index, index inside batch]
                    self.fragments.append([idx, i])

                pbar.update(1)
                if idx == num_batches - 1:
                    break

        # self.dict_vecs is set in the Forward Hook of the Model (Setup in function setup_sae_hook)
        # Concatenate all Dict-Vec-Batches to a single large list
        raw_activations = torch.cat(self.dict_vecs, dim=0).detach().cpu().to(torch.float16)

        # Rescale Dict-Feature-Activations 'raw_activations' to range 0 - 10
        minima = torch.min(torch.min(raw_activations, dim=1).values, dim=0).values
        minus_min = raw_activations - minima
        maxima = torch.max(torch.max(minus_min, dim=1).values, dim=0).values
        div_by_max = torch.div(minus_min, maxima)
        self.rescaled_feature_activations = 10 * torch.where(div_by_max.isnan(), 0, div_by_max)

        # Calculate the mean activation of a Feature for each text fragment
        self.mean_feature_activations = torch.mean(self.rescaled_feature_activations, dim=1)

    @utils.ModelNeededDecorators.PARAMETER_NEEDED("rescaled_feature_activations")
    def save_interpretation_samples(self, path):
        """
        Saves the Interpretation-Samples (Feature Activations over a text corpus) to disk.
        Additionally contains Hyperparameters of Autoencoder used to obtain the Samples.
        :param path: Path to save Interpretation-Samples to
        """
        # Frequency-Information, Strongly Activating Text Fragments and their Activations
        interpretation_samples = {
                "fragments": self.fragments,
                "rescaled": self.rescaled_feature_activations,
                "mean_feature_activations": self.mean_feature_activations,
                "interpretable_neuron_indices": self.interpretable_neuron_indices,
                "SAE_INFO": {}
            }

        # Autoencoder-Information except state-dict
        for key in self.autoencoder_config.keys():
            if key == "STATE_DICT":
                continue
            interpretation_samples["SAE_INFO"][key] = self.autoencoder_config[key]

        # Write to disk
        with open(path, "wb") as f:
            pickle.dump(interpretation_samples, f)

    def load_interpretation_samples(self, path):
        """
        Load Interpretation-Samples from disk.
        :param path: Path, the Interpretation-Samples are loaded from
        """
        print("Loading Interpretation-Samples")

        with open(path, "rb") as f:
            obj = pickle.load(f)

        self.fragments = obj["fragments"]
        self.rescaled_feature_activations = obj["rescaled"]
        self.mean_feature_activations = obj["mean_feature_activations"]
        self.interpretable_neuron_indices = obj["interpretable_neuron_indices"]

    @utils.ModelNeededDecorators.PARAMETER_NEEDED("rescaled_feature_activations")
    def get_fragment(self, interp_neuron_index, rank, return_activations=False):
        """
        Returns one Text Fragment on which the given interp_neuron_index activates strongly.
        :param interp_neuron_index: Interpretable Neuron Index on which the returned Text Fragment activates strongly
        :param rank: Rank of activation strength of the returned Text Fragment. 0 -> strongest activation
        :param return_activations: Whether to also return the rescaled token activations for the Text Fragment
        :return: The Text Fragment on which the given feature_index activates strongly. If return_activation == True,
        additionally Feature Activations
        """

        # Compute the fragments on which the given Neuron activates the most
        # Feature Activations on all Text Fragments on all Features in interpretable_neuron_indices
        feature_activations = self.mean_feature_activations[::, interp_neuron_index]
        # Compute top-k most activating Text Fragments
        top_fragment_ids = feature_activations.topk(rank + 1, largest=True).indices
        # Text-Fragment-ID, representing the position of the Text Fragment in self.fragments
        tf_id = top_fragment_ids[rank]

        # Return procedure
        fragment_idx = self.fragments[tf_id]
        # fragment_idx is of [batch index, index inside batch]
        fragment = self.ds[self.BATCH_SIZE * fragment_idx[0] + fragment_idx[1]]

        if return_activations:
            rescaled_per_token_feature_acts = self.rescaled_feature_activations[tf_id, :, interp_neuron_index]
            return fragment, rescaled_per_token_feature_acts

        return fragment

    def generate_interpretation_prompt(self, interp_neuron_index, num_top_fragments):
        """
        Generates an Interpretation Prompt to infer an Explanation of a Feature.
        :param interp_neuron_index: The Interpretable Neuron Index of the Feature to infer an Interpretation for
        :param num_top_fragments: Number of text fragments used for obtaining the Interpretation
        :return: User Prompt for Interpretation
        """

        # Obtain complete texts
        complete_texts = []
        for i in range(num_top_fragments):
            fragment = self.get_fragment(
                interp_neuron_index,
                i,
                return_activations=False
            )

            try:
                text = self.interpretation_model.tokenizer.batch_decode([fragment])[0]
            except Exception:
                # IndexError -> due to passing
                continue
            complete_texts.append(text)

        # Obtain Tokens and Activations of Fragments
        tokens, activations = [], []
        for i in range(num_top_fragments):
            fragment, rescaled_per_token_feature_acts = self.get_fragment(
                interp_neuron_index,
                i,
                return_activations=True
            )
            try:
                tokens += self.interpretation_model.tokenizer.convert_ids_to_tokens(fragment[:self.NUM_TOKENS])
            except Exception:
                # IndexError -> due to passing
                continue
            activations += rescaled_per_token_feature_acts[:self.NUM_TOKENS]

        # Generate User-Prompt using PromptGenerator of InterpretationConfig
        user_prompt = self.interpretation_config.prompt_generator.get_interpretation_prompt(
            complete_texts,
            tokens,
            activations,
            include_complete_texts=self.interpretation_config.include_complete_texts,
            include_filtered_tokens=self.interpretation_config.include_filtered_tokens
        )

        return user_prompt

    @utils.ModelNeededDecorators.PARAMETER_NEEDED("interpretation_model")
    def get_explanation(self, user_prompt, raw=False):
        """
        Generate the Explanation for a specific Feature.
        :param user_prompt: User prompt to infer the Interpretation of a Neuron
        :param raw: Whether to return the Explanation including the input or only the Explanation
        :return: If raw == False, Explanation of the Neuron's behavior. If raw == True, the raw output of the LLM.
        """

        # Generate Explanation, raw_explanation consists of complete answer of LLM including initial prompt
        system_prompt = self.interpretation_config.prompt_generator.get_interpretation_system_prompt()
        raw_explanation = self.interpretation_model.generate_instructive(system_prompt, user_prompt, max_new_tokens=100)

        explanation = self.interpretation_config.prompt_generator.extract_llm_output(raw_explanation)

        return raw_explanation if raw else explanation

    def generate_simulation_prompt(self, interp_neuron_index, rank, explanation):
        """
        Generates a Simulation Prompt for the Interpretation Model.
        :param interp_neuron_index: The Interpretable Neuron Index of the Feature which should be simulated
        :param rank: The Text Fragment which should be simulated. 0 -> most activating fragment
        :param explanation: The Explanation of the Neuron, generated by the Interpretation Model
        :return: Generated User Prompt
        """

        fragment = self.get_fragment(interp_neuron_index, rank)
        try:
            tokens = self.interpretation_model.tokenizer.convert_ids_to_tokens(fragment[:self.NUM_TOKENS])
        except Exception:
            # IndexError -> due to passing
            tokens = []

        user_prompt = self.interpretation_config.prompt_generator.get_simulation_prompt(tokens, explanation)

        return user_prompt

    @utils.ModelNeededDecorators.PARAMETER_NEEDED("interpretation_model")
    def get_simulation(self, user_prompt, raw=False):
        """
        Runs a simulation prompt on the Interpretation Model and parses the inferred activations.
        :param user_prompt: User Prompt which infers the Neuron activations
        :param raw: Whether to return the raw Simulation-Output from the Interpretation-LLM
        :return: List of tuples containing the inferred neuron activations. If raw == True, additionally the raw
        Simulation Output.
        """

        # Generate Explanation, raw_explanation consists of complete answer of LLM including initial prompt
        system_prompt = self.interpretation_config.prompt_generator.get_simulation_system_prompt()
        raw_simulation = self.interpretation_model.generate_instructive(system_prompt, user_prompt, max_new_tokens=1000)

        simulation = self.interpretation_config.prompt_generator.extract_llm_output(raw_simulation)

        kv_dict = {}

        lines = list(simulation.split("\n"))

        replacement_dict = {
            "\n": "",
            "\t": "",
            "  ": " ",
        }

        for raw_line in lines:
            # Apply Replacements in <replacement_dict> and remove leading Asterisk in each line
            line = utils.apply_dict_replacement(raw_line, replacement_dict)
            line = utils.remove_leading_asterisk(line)

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
                token = utils.apply_dict_replacement(token, self.interpretation_config.prompt_generator.token_replacements)

                kv_dict[token] = int(score)

                # If Filter Matches, skip all following Filters and proceed with next Line
                break

        if len(kv_dict.values()) == 0:
            return kv_dict

        # Rescale all Token Scores to 0-10 (Sometimes the Interpretation-LLM outputs only binary/0-1)
        maximum = max(kv_dict.values())
        minimum = min(kv_dict.values())

        # Transform Simulated Scores from Range [minimum, maximum] to [0, maximum - minimum] and rescale to [0, 10]
        # If maximum == minimum, all scores will be zero
        if maximum == minimum:
            scaling_factor = 1
        else:
            scaling_factor = 10 / (maximum - minimum)
        kv_dict_rescaled = {key: scaling_factor * (kv_dict[key] - minimum) for key in kv_dict}

        if raw:
            return kv_dict_rescaled, raw_simulation
        return kv_dict_rescaled

    @utils.ModelNeededDecorators.PARAMETER_NEEDED("rescaled_feature_activations")
    def generate_ground_truth_scores(self, interp_neuron_index, rank):
        """
        Generates the Ground Truth Activation-Sample for a given Text Sample ID.
        :param interp_neuron_index: The Interpretable Neuron Index for which the Activation-Sample should be sampled
        :param rank: The rank of the Text-Fragment. 0 -> Most strongly activation, 1 -> Second most strongly, ...
        :return: Ground Truth Activation-Sample
        """

        fragment, rescaled_per_token_feature_acts = self.get_fragment(
            interp_neuron_index,
            rank,
            return_activations=True
        )

        kv_dict = {}

        for i in range(len(fragment)):
            try:
                key = f"{self.interpretation_model.tokenizer.convert_ids_to_tokens([fragment[i]])[0]}"
            except Exception:
                # IndexError -> due to passing
                continue
            value = int(rescaled_per_token_feature_acts[i])

            # Replace unwanted Characters in the Token
            key = utils.apply_dict_replacement(key, self.interpretation_config.prompt_generator.token_replacements)

            kv_dict[key] = value

        return kv_dict
