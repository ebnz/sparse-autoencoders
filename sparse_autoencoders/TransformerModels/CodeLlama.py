import torch
import deepspeed

from transformers import LlamaForCausalLM, CodeLlamaTokenizer

from .TransformerModels import TransformerModelWrapper

"""
Implementations of TransformerModelWrapper for CodeLlama
"""

class CodeLlamaModel(TransformerModelWrapper):
    def __init__(self, model, tokenizer=None, device="cpu"):
        super().__init__(model, tokenizer=tokenizer, device=device)

    def load_model_from_name(self, model_name):
        self.model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

    def load_tokenizer_from_name(self, tokenizer_name):
        self.tokenizer = CodeLlamaTokenizer.from_pretrained(tokenizer_name)

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

    def setup_hook(self, hook, layer_id, layer_type, permanent=False):
        if layer_id < 0 or layer_id >= len(self.model.model.layers) or layer_id is None:
            raise Exception("layer_id not found")

        if layer_type == "attn_sublayer":
            handle = self.model.model.layers[layer_id].self_attn.register_forward_hook(hook)
        elif layer_type == "mlp_sublayer":
            handle = self.model.model.layers[layer_id].mlp.register_forward_hook(hook)
        elif layer_type == "mlp_activations":
            handle = self.model.model.layers[layer_id].mlp.down_proj.register_forward_hook(hook)
        else:
            try:
                # Recursively get Attributes
                # ToDo: ugly as hell, tidy up
                attribute = self.model.model.layers[layer_id] if layer_id is not None else self.model.model
                for attribute_name in layer_type.split("."):
                    attribute = attribute.__getattribute__(attribute_name)
                handle = attribute.register_forward_hook(hook)
            except AttributeError:
                raise AttributeError("Unrecognized Type of layer_type")
        if not permanent:
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

        # Manually set device for DeepSpeed
        self.device = "cuda"
