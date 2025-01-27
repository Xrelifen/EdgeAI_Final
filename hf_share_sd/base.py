# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2023
#####################################################
import torch
from torch import nn
from torch import float16
from typing import Callable, Union
from functools import partial
from tqdm import tqdm
from transformers.models.llama import LlamaModel

from hqq.models.hf.base import AutoHQQHFModel
from hqq.models.base import get_all_children_from_model, forward_device_hooked, find_parent, name_to_linear_tag
from hqq.models.base import _QUANT_LAYERS
from hqq.core.quantize import *
from hqq.core.utils import cleanup

_DECODER_LAYERS = ["layers"]

def get_modules_by_substring(model, substring: str, ignore: list = []) -> list:
    """
    Return all modules from `model` whose last name contains `substring`,
    ignoring any final submodule names listed in `ignore`.
    """
    matched = []
    for name, module in model.named_modules():
        # Only collect leaf modules (like in your get_all_children_from_model)
        if (name.split(".")[-1] not in ignore):
            if substring in name.split(".")[-1]:
                matched.append((name, module))
    return matched

def get_linear_from_model(model, ignore: list = []) -> list:
    matched = []
    for name, module in model.named_modules():
        if (type(module) in _QUANT_LAYERS) and (name.split(".")[-1] not in ignore):
            matched.append((name, module))
    return matched

class AutoHQQHFShareSDModel(AutoHQQHFModel):
    @classmethod
    def patch_linearlayers(
        cls,
        model,
        patch_fct: Callable,
        patch_params: Union[dict, None],
        verbose: bool = True,
    ):
        ignore_tags = cls.get_ignore_layers(model)

        tmp_mapping = {}
        for name, module in model.named_modules():
            if (type(module) in _QUANT_LAYERS) and (name not in ignore_tags):
                tmp_mapping[name] = module

        modified_pairs = []
        for name in tqdm(tmp_mapping, disable=not verbose):
            linear_tag = name_to_linear_tag(name)
            patch_param = (
                patch_params[linear_tag] if (linear_tag in patch_params) else None
            )

            parent = find_parent(model, name)
            new_name = "quant_" + name.split(".")[-1]
            setattr(
                parent,
                new_name,
                patch_fct(tmp_mapping[name], patch_param),
            )

            if patch_param is not None:
                modified_pairs.append((tmp_mapping[name], getattr(parent, new_name))) # orginal, patched

        cleanup()
        return modified_pairs

    @classmethod
    def patch_decoderlayers(
        cls,
        model,
        patch_fct: Callable,
        patch_params: Union[dict, None],
        verbose: bool = True,
    ) -> None:
        tmp_mapping = {}
        for name, module in model.named_modules():
            if (name.split(".")[-1] in _DECODER_LAYERS):
                if isinstance(module, (nn.ModuleList, list)):
                    for i, sub_module in enumerate(module):
                        tmp_mapping[name + "." + str(i)] = sub_module
                else:
                    raise ValueError("Decoder layer should be a nn.ModuleList or list")

        for name in tqdm(tmp_mapping, disable=not verbose):
            linear_tag = name_to_linear_tag(name)
            patch_param = (
                patch_params[linear_tag] if (linear_tag in patch_params) else None
            )
            setattr(
                find_parent(model, name),
                name.split(".")[-1], # [MODIFIED] We give a new name for the quantized layer
                patch_fct(tmp_mapping[name], patch_param),
            )

        cleanup()

    # Main patching function
    @classmethod
    def patch_model(
        cls,
        model,
        patch_nonlinear_fct: Callable,
        patch_decoder_fct: Callable,
        patch_params: dict,
        verbose: bool = True,
    ) -> None:
        model.eval()
        cls.freeze_model(model)
        cls.autoname_modules(model)
        cls.patch_nonlinearlayers(model, patch_nonlinear_fct, verbose=verbose)
        cls.patch_decoderlayers(model, patch_decoder_fct, patch_params, verbose=verbose)
        # cls.patch_linearlayers(model, patch_linear_fct, patch_params, verbose=verbose)
        cleanup()


    # Main function to quantize a model. Basically goes through the linear layers specfied in the patching function and replaces them with HQQLinear
    @classmethod
    def quantize_model(
        cls,
        model,
        quant_config: dict,
        compute_dtype: torch.dtype = float16,
        device: Union[str, list, dict] = "cuda",
    ):
        # Check if the model was already quantized
        if getattr(model, "hqq_quantized", False):
            print("Model was already quantized")
            return

        # Set linear tags automatically
        cls.setup_model(model)

        # Use the same quantization config for all linear layers. Use None to skip quantizing a specfic layer.
        if True in [(key in model.linear_tags) for key in quant_config.keys()]:
            # If the user doesn't specify a key from get_linear_tags, the layer is not quantized via (key, None)
            patch_params = {key: None for key in model.linear_tags}
            patch_params.update(quant_config)
        else:
            # Same quant_config for all layers
            patch_params = {k: quant_config for k in model.linear_tags}
        # print("patch_params: \n", patch_params)
        # print("quant_config: \n", quant_config)
        # exit()


        # Get list of all nodes in order
        all_nodes = get_all_children_from_model(model, [])  # ordered nodes
        try:
            # Extract block names: This is following Hugging Face models.
            num_blocks = (
                len(model.model.layers)
                if hasattr(model, "model")
                else len(model.layers)
            )
            all_blocks = ["model.layers." + str(i) for i in range(num_blocks)]
        except Exception:
            all_blocks = None
            print(
                "Default model structure not supported. Make sure you feed device as dictionary as {name_block: device}"
            )

        if isinstance(
            device, dict
        ):  # input as {module block name (str): device (str or torch.device)}
            device_map = device
            num_devices = len(set([device_map[k] for k in device_map]))
            all_blocks = list(device_map.keys())

        node_to_block = {}
        for node in all_nodes:
            res = [block for block in all_blocks if (block in node)]
            node_to_block[node] = res[-1] if (len(res) > 0) else node

        # Set device-map
        if isinstance(device, str):  # single device as str
            device_map = {k: device for k in all_blocks + all_nodes}
            num_devices = 1

        if isinstance(device, list):  # list of devices
            num_devices = len(device)
            device_map = {}
            for node in all_nodes:
                if ".layers" in node:
                    break
                device_map[node] = device[0]

            for node in all_nodes[::-1]:
                if ".layers" in node:
                    break
                device_map[node] = device[-1]

            step, k = len(all_blocks) // num_devices, 0
            for i in range(0, len(all_blocks), step):
                for j in range(i, i + step):
                    device_map[all_blocks[min(j, len(all_blocks) - 1)]] = device[
                        min(k, num_devices - 1)
                    ]
                k += 1

        # Map nodes to block devices
        for node in all_nodes:
            device_map[node] = device_map[node_to_block[node]]

        # We create a new HQQLinear for each nn.Linear in specified layers
        def _patch_linear(linear_layer, quant_config):
            if type(linear_layer) is HQQLinear:
                return linear_layer

            current_device = device_map[linear_layer.name]

            if quant_config is not None:
                out_module = HQQLinear(
                    linear_layer,
                    quant_config,
                    del_orig=False, # [MODIFIED] We keep the original layer
                    compute_dtype=compute_dtype,
                    device=current_device,
                )
            else:
                out_module = linear_layer.to(device=current_device, dtype=compute_dtype)

            out_module.device = current_device
            return out_module

        def _patch_decoder_layer(layer, quant_config):
            # Return immediately if this layer is already patched
            if getattr(layer, "_patched", False):
                return layer
            layer._patched = True

            #TODO: write another function to obtain modified_pairs & all_linears, instead of obtaining from patch_linearlayers.
            #TODO: The function will be used for rebuilding the new_forward on load model. (Currently save/load model is not supported)
            # Patch linear layers and store (original, quantized) pairs
            modified_pairs = cls.patch_linearlayers(
                layer, _patch_linear, patch_params, verbose=False
            )
            all_linears = get_linear_from_model(layer)
            original_forward = layer.forward

            def new_forward(self, *args, draft_mode=False, **kwargs):
                # If draft mode not requested, run the original forward
                if not draft_mode:
                    return original_forward(self, *args, **kwargs)
                
                # Enable draft mode on all linear modules
                for _, linear in all_linears:
                    linear._draft_mode = True

                # Swap in the quantized forward methods
                backups = {}
                for org_linear, quant_linear in modified_pairs:
                    backups[org_linear] = org_linear.forward
                    org_linear.forward = quant_linear.forward

                # Run forward with quantized linears
                outputs = original_forward(self, *args, **kwargs)

                # Restore original forward methods
                for org_linear, _ in modified_pairs:
                    org_linear.forward = backups[org_linear]

                return outputs

            layer.forward = new_forward
            return layer
        
        def _patch_decoder_layer(layer, quant_config):
            # Return immediately if this layer is already patched
            if getattr(layer, "_patched", False):
                return layer
            layer._patched = True

            #TODO: write another function to obtain modified_pairs & all_linears, instead of obtaining from patch_linearlayers.
            #TODO: The function will be used for rebuilding the new_forward on load model. (Currently save/load model is not supported)
            # Patch linear layers and store (original, quantized) pairs
            modified_pairs = cls.patch_linearlayers(
                layer, _patch_linear, patch_params, verbose=False
            )
            all_linears = get_linear_from_model(layer)

            def set_draft_mode(self, draft_mode):
                if draft_mode:
                    for _, linear in all_linears:
                        linear._draft_mode = True

                    # Swap in the quantized forward methods
                    self._backups = getattr(self, "_backups", {})
                    for org_linear, quant_linear in modified_pairs:
                        self._backups[org_linear] = org_linear.forward
                        org_linear.forward = quant_linear.forward
                else:
                    for _, linear in all_linears:
                        linear._draft_mode = False

                    # Restore original forward methods
                    if hasattr(self, "_backups"):
                        for org_linear, _ in modified_pairs:
                            org_linear.forward = self._backups[org_linear]

            layer.set_draft_mode = set_draft_mode
            return layer

        def _patch_other(layer):
            current_device = device_map[layer.name]
            layer.device = current_device
            return layer.to(device=current_device, dtype=compute_dtype)

        cls.patch_model(model, _patch_other, _patch_decoder_layer, patch_params)

        # for name, decoder_layers in get_modules_by_substring(model, "layers"):
        #     for i, block in enumerate(decoder_layers):

        decoder_layers = get_modules_by_substring(model, "layers")[0][1]

        # set draft mode for the model
        def set_draft_mode(draft_mode):
            for layer in decoder_layers:
                layer.set_draft_mode(layer, draft_mode)
        model.set_draft_mode = set_draft_mode

        # Insert device switcher
        if num_devices > 1:
            core_model = model if hasattr(model, "layers") else model.model

            # Make sure the input (first node) has the input in the right device during generation
            input_node_child_name = all_nodes[0].split(".")[-1]
            input_node = getattr(core_model, input_node_child_name)
            input_node.device = device_map[all_nodes[0]]
            input_node.forward_orig = input_node.forward
            input_node.forward = partial(forward_device_hooked, input_node)
            setattr(core_model, input_node_child_name, input_node)

            # Make sure all inputs to the blocks are in the right device
            for i in range(len(core_model.layers)):
                core_model.layers[i].device = device_map[core_model.layers[i].name]
                core_model.layers[i].forward_orig = core_model.layers[i].forward
                core_model.layers[i].forward = partial(
                    forward_device_hooked, core_model.layers[i]
                )

        # Set base class
        model.base_class = cls

        model.hqq_quantized = True

        return model
