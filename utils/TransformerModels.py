import torch
import deepspeed

from transformers import LlamaForCausalLM, CodeLlamaTokenizer

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


class CodeLlamaModel(TransformerModelWrapper):
    def __init__(self, model, tokenizer=None, device="cpu"):
        super().__init__(model, tokenizer=tokenizer, device=device)

        self.device = device
        self.to(self.device)

    def load_model_from_name(self, model_name):
        self.model = LlamaForCausalLM.from_pretrained(self.model_name)

    def load_tokenizer_from_name(self, tokenizer_name):
        self.tokenizer = CodeLlamaTokenizer.from_pretrained(self.model_name)

    def generate_instructive(self, system_prompt, user_prompt, top_p=0.9, temperature=0.1, max_new_tokens=500, add_special_tokens=True):
        prompt = f"[INST]<<SYS>>{system_prompt}<</SYS>>\n{user_prompt}[/INST]"

        return self.generate_on_prompt(prompt, top_p=top_p, temperature=temperature, max_new_tokens=max_new_tokens,
                                       add_special_tokens=add_special_tokens)

    def run_model_until_layer(self, input_ids, stop_layer):
        """
        Run the Model and stop after the layer_output-th Layer.
        :param input_ids: Input IDs on which the Target Model should be run
        :param stop_layer: Layer at which the Model-Run is stopped
        :return: Output Activations of the layer_output-th Layer
        """
        # Embed Tokens and generate position_ids for the Layers. The API does not accept position_ids=None at the moment
        layer_output = self.model.model.embed_tokens(input_ids.to(self.device))
        position_ids = torch.tensor([[i for i in range(input_ids.shape[1])]])
        position_ids = position_ids.repeat(input_ids.shape[0], 1).to(self.device)

        # Run Layers of the Model manually
        for i_layer in range(stop_layer + 1):
            layer_output = self.model.model.layers[i_layer](layer_output, position_ids=position_ids)[0]

        return layer_output

    def setup_hook(self, hook, layer_id, layer_type):
        if layer_id < 0 or layer_id >= len(self.target_model.model.layers):
            raise Exception("layer_id not found")

        if layer_type == "attn_sublayer":
            handle = self.target_model.model.layers[layer_id].self_attn.register_forward_hook(hook)
        elif layer_type == "mlp_sublayer":
            handle = self.target_model.model.layers[layer_id].mlp.register_forward_hook(hook)
        elif layer_type == "mlp_activations":
            handle = self.target_model.model.layers[layer_id].mlp.down_proj.register_forward_hook(hook)
        else:
            raise Exception("Unrecognized Type of layer_type")
        self.model_hook_handles.append(handle)


class CodeLlamaModelDeepspeed(CodeLlamaModel):
    def __init__(self, model, num_gpus, tokenizer=None):
        super().__init__(model, tokenizer=tokenizer, device="cpu")

        # Send Model to Deepspeed for Multi-GPU
        ds_engine_large = deepspeed.init_inference(self.model,
                                                   tensor_parallel={"tp_size": num_gpus},
                                                   dtype=torch.float16,
                                                   checkpoint=None,
                                                   replace_with_kernel_inject=False)
        self.model = ds_engine_large.module
