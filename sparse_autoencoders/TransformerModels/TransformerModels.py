class TransformerModelWrapper:
    def __init__(self, model, tokenizer=None, device="cpu"):
        """
        A Wrapper-Class for a Transformer-like LLM
        :type model: transformers.Model
        :type tokenizer: transformer.Tokenizer
        :type device: str
        :param model: HuggingFace Transformer Model or Name
        :param tokenizer: HuggingFace Tokenizer or Name
        :param device: Device for the Model
        """
        if isinstance(model, str):
            self.load_model_from_name(model)
        else:
            self.model = model

        if tokenizer is None and not isinstance(model, str):
            raise AttributeError("Must specify tokenizer when passing loaded Model as model and not str")
        elif isinstance(tokenizer, str):
            self.load_tokenizer_from_name(tokenizer)
        elif tokenizer is None and isinstance(model, str):
            self.load_tokenizer_from_name(model)
        else:
            self.tokenizer = tokenizer

        self.device = device
        self.to(device)

        self.model_hook_handles = []

    def load_model_from_name(self, model_name):
        """
        Loads a HuggingFace Transformer by Name.
        :type model_name: str
        :param model_name: Name of Transformer
        """
        raise NotImplementedError("This Class is an interface")

    def load_tokenizer_from_name(self, tokenizer_name):
        """
        Loads a HuggingFace Tokenizer by Name.
        :type tokenizer_name: str
        :param tokenizer_name: Name of Tokenizer
        """
        raise NotImplementedError("This Class is an interface")

    def to(self, device):
        """
        Offloads the Transformer Model to the specified Device.
        :type device: str
        :param device: Device to offload to
        """
        self.device = device
        self.model.to(self.device)

    def generate_on_prompt(self, prompt, top_p=0.9, temperature=1.0, max_new_tokens=500, add_special_tokens=True):
        """
        Generates an answer to a specific prompt with a chosen LLM-Model and Tokenizer.
        :type prompt: str
        :type top_p: float
        :type temperature: float
        :type max_new_tokens: int
        :type add_special_tokens: bool
        :param prompt: The Prompt for the LLM
        :param top_p: top_p Parameter for Beam Searching the LLM
        :param temperature: Randomness Factor
        :param max_new_tokens: Maximum Amount of new Tokens that are generated
        :param add_special_tokens: Whether to add special tokens such as <s>, </s>, [INST], [/INST], ...
        :return: Decoded output of LLM to the given prompt
        """

        # Tokenize input string and send to device on which the model is located (e.g. cuda)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens)
        input_ids_cuda = inputs["input_ids"].to(self.device)
        attention_mask_cuda = inputs["attention_mask"].to(self.device)

        # Generate output with LLM model
        output = self.model.generate(
            input_ids_cuda,
            attention_mask=attention_mask_cuda,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Return decoded tokens
        return self.tokenizer.decode(output[0])

    def generate_instructive(self, system_prompt, user_prompt, top_p=0.9, temperature=0.1, max_new_tokens=500, add_special_tokens=True):
        """
        Generates an answer to a specific prompt with a chosen LLM-Model and Tokenizer.
        :type system_prompt: str
        :type user_prompt: str
        :type top_p: float
        :type temperature: float
        :type max_new_tokens: int
        :type add_special_tokens: bool
        :param system_prompt: The System-Prompt for the LLM
        :param user_prompt: The User-Prompt for the LLM
        :param top_p: top_p Parameter for Beam Searching the LLM
        :param temperature: Randomness Factor
        :param max_new_tokens: Maximum Amount of new Tokens that are generated
        :param add_special_tokens: Whether to add special tokens such as <s>, </s>, [INST], [/INST], ...
        :return: Decoded output of LLM to the given prompt
        """
        raise NotImplementedError("This Class is an interface")

    def run_model_until_layer(self, input_ids, stop_layer):
        """
        Run the Model and stop after the layer_output-th Layer.
        :type input_ids: torch.Tensor
        :type stop_layer: int
        :param input_ids: Input IDs on which the Target Model should be run
        :param stop_layer: Layer at which the Model-Run is stopped
        :return: Output Activations of the layer_output-th Layer
        """
        raise NotImplementedError("This Class is an interface")

    def setup_hook(self, hook, module_name, permanent=False):
        """
        Place a Hook in a Transformer-Model.
        :type hook: function
        :type module_name: str
        :type permanent: bool
        :param hook: Hook to install
        :param module_name: Name of the Module, the Hook is placed on
        :param permanent: Whether a call of clear_hooks should remove the Hook
        """
        modules_dict = dict(self.model.named_modules())

        # Retrieve Module from Name and register Hook
        try:
            module = modules_dict[module_name]
        except KeyError:
            raise ValueError(f"Module: <{module_name}> does not exist in Model <{self.model}> "
                             f"and is not registered in layer_aliases of TransformerModelWrapper")

        handle = module.register_forward_hook(hook)

        # Permanent Hooks can't be removed
        if not permanent:
            self.model_hook_handles.append(handle)

    def clear_hooks(self):
        """
        Clear all non-permanent Hooks placed in this Model.
        """
        for hook in self.model_hook_handles:
            hook.remove()
        self.model_hook_handles = []

    # Delegate missing Methods/Attributes to self.model
    def __getattr__(self, name):
        return getattr(self.model, name)
