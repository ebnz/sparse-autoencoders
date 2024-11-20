class TransformerModelWrapper:
    def __init__(self, model, tokenizer=None, device="cpu"):
        if isinstance(model, str):
            self.load_model_from_name(model)
        else:
            self.model = model

        if tokenizer is None and not isinstance(model, str):
            raise AttributeError("Must specify tokenizer when passing loaded Model as model and not str")
        elif isinstance(tokenizer, str):
            self.load_tokenizer_from_name(tokenizer)
        else:
            self.tokenizer = tokenizer

        self.device = device

        self.model_hook_handles = []

    def load_model_from_name(self, model_name):
        raise NotImplementedError("This Class is an interface")

    def load_tokenizer_from_name(self, tokenizer_name):
        raise NotImplementedError("This Class is an interface")

    def to(self, device):
        self.device = device
        self.model.to(self.device)

    def generate_on_prompt(self, prompt, top_p=0.9, temperature=0.1, max_new_tokens=500, add_special_tokens=True):
        """
        Generates an answer to a specific prompt with a chosen LLM-Model and Tokenizer.
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

        # Generate output with LLM model
        output = self.model.generate(
            input_ids_cuda,
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
        raise NotImplementedError("This Class is an interface")

    def run_model_until_layer(self, input_ids, stop_layer):
        raise NotImplementedError("This Class is an interface")

    def setup_hook(self, hook, layer_id, layer_type):
        raise NotImplementedError("This class is an interface")

    def clear_hooks(self):
        for hook in self.model_hook_handles:
            hook.remove()
        self.model_hook_handles = []

    # Delegate missing Methods/Attributes to self.model
    def __getattr__(self, name):
        return getattr(self.model, name)
