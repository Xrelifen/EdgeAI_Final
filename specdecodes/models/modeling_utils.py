from functools import wraps
import os
import logging
from typing import Dict, List, Mapping, Optional, Union

import torch
import torch.nn as nn

from accelerate.utils import (
    PrefixedDataset,
    find_device,
    named_module_tensors,
    send_to_device,
    check_device_map,
    extract_submodules_state_dict,
    find_tied_parameters,
    offload_state_dict,
    OffloadedWeightsLoader,
    retie_parameters,
    is_bnb_available,
    is_mlu_available,
    is_musa_available,
    is_npu_available,
    is_xpu_available,
    check_cuda_p2p_ib_support,
)
from accelerate.hooks import ModelHook, SequentialHook
from accelerate.utils.memory import clear_device_cache
from accelerate.utils.modeling import get_non_persistent_buffers, check_device_same
from accelerate.utils.other import recursive_getattr
from accelerate.hooks import (
    AlignDevicesHook,
    add_hook_to_module,
)

import nvtx

_accelerate_added_attributes = ["to", "cuda", "npu", "xpu", "mlu", "musa"]

def set_module_tensor_to_device(
    module: nn.Module,
    tensor_name: str,
    device: Union[int, str, torch.device],
    prefetch: bool = False,
    value: Optional[torch.Tensor] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    fp16_statistics: Optional[torch.HalfTensor] = None,
    tied_params_map: Optional[Dict[int, Dict[torch.device, torch.Tensor]]] = None,
):
    """
    A helper function to set a given tensor (parameter of buffer) of a module on a specific device (note that doing
    `param.to(device)` creates a new tensor not linked to the parameter, which is why we need this function).

    Args:
        module (`torch.nn.Module`):
            The module in which the tensor we want to move lives.
        tensor_name (`str`):
            The full name of the parameter/buffer.
        device (`int`, `str` or `torch.device`):
            The device on which to set the tensor.
        value (`torch.Tensor`, *optional*):
            The value of the tensor (useful when going from the meta device to any other device).
        dtype (`torch.dtype`, *optional*):
            If passed along the value of the parameter will be cast to this `dtype`. Otherwise, `value` will be cast to
            the dtype of the existing parameter in the model.
        fp16_statistics (`torch.HalfTensor`, *optional*):
            The list of fp16 statistics to set on the module, used for 8 bit model serialization.
        tied_params_map (Dict[int, Dict[torch.device, torch.Tensor]], *optional*, defaults to `None`):
            A map of current data pointers to dictionaries of devices to already dispatched tied weights. For a given
            execution device, this parameter is useful to reuse the first available pointer of a shared weight on the
            device for all others, instead of duplicating memory.
    """
    # Recurse if needed
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
        tensor_name = splits[-1]

    if tensor_name not in module._parameters and tensor_name not in module._buffers:
        raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")
    is_buffer = tensor_name in module._buffers
    old_value = getattr(module, tensor_name)

    # Treat the case where old_value (or a custom `value`, typically offloaded to RAM/disk) belongs to a tied group, and one of the weight
    # in the tied group has already been dispatched to the device, by avoiding reallocating memory on the device and just copying the pointer.
    if (
        value is not None
        and tied_params_map is not None
        and value.data_ptr() in tied_params_map
        and device in tied_params_map[value.data_ptr()]
    ):
        module._parameters[tensor_name] = tied_params_map[value.data_ptr()][device]
        return
    elif (
        tied_params_map is not None
        and old_value.data_ptr() in tied_params_map
        and device in tied_params_map[old_value.data_ptr()]
    ):
        module._parameters[tensor_name] = tied_params_map[old_value.data_ptr()][device]
        return

    if old_value.device == torch.device("meta") and device not in ["meta", torch.device("meta")] and value is None:
        raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {device}.")

    param = module._parameters[tensor_name] if tensor_name in module._parameters else None
    param_cls = type(param)

    if value is not None:
        # We can expect mismatches when using bnb 4bit since Params4bit will reshape and pack the weights.
        # In other cases, we want to make sure we're not loading checkpoints that do not match the config.
        if old_value.shape != value.shape and param_cls.__name__ != "Params4bit":
            raise ValueError(
                f'Trying to set a tensor of shape {value.shape} in "{tensor_name}" (which has shape {old_value.shape}), this looks incorrect.'
            )

        if dtype is None:
            # For compatibility with PyTorch load_state_dict which converts state dict dtype to existing dtype in model
            value = value.to(old_value.dtype)
        elif not str(value.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
            value = value.to(dtype)

    device_quantization = None
    with torch.no_grad():
        # leave it on cpu first before moving them to cuda
        # # fix the case where the device is meta, we don't want to put it on cpu because there is no data =0
        if (
            param is not None
            and param.device.type != "cuda"
            and torch.device(device).type == "cuda"
            and param_cls.__name__ in ["Int8Params", "FP4Params", "Params4bit"]
        ):
            device_quantization = device
            device = "cpu"
        # `torch.Tensor.to(<int num>)` is not supported by `torch_npu` (see this [issue](https://github.com/Ascend/pytorch/issues/16)).
        if isinstance(device, int):
            if is_npu_available():
                device = f"npu:{device}"
            elif is_mlu_available():
                device = f"mlu:{device}"
            elif is_musa_available():
                device = f"musa:{device}"
            elif is_xpu_available():
                device = f"xpu:{device}"
        if "xpu" in str(device) and not is_xpu_available():
            raise ValueError(f'{device} is not available, you should use device="cpu" instead')
        if value is None:
            new_value = old_value.to(device, non_blocking=prefetch)
            if dtype is not None and device in ["meta", torch.device("meta")]:
                if not str(old_value.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
                    new_value = new_value.to(dtype)

                if not is_buffer:
                    module._parameters[tensor_name] = param_cls(new_value, requires_grad=old_value.requires_grad)
        elif isinstance(value, torch.Tensor):
            new_value = value.to(device, non_blocking=prefetch)
        else:
            new_value = torch.tensor(value, device=device)
        if device_quantization is not None:
            device = device_quantization
        if is_buffer:
            module._buffers[tensor_name] = new_value
        elif value is not None or not check_device_same(torch.device(device), module._parameters[tensor_name].device):
            param_cls = type(module._parameters[tensor_name])
            kwargs = module._parameters[tensor_name].__dict__
            if param_cls.__name__ in ["Int8Params", "FP4Params", "Params4bit"]:
                if param_cls.__name__ == "Int8Params" and new_value.dtype == torch.float32:
                    # downcast to fp16 if any - needed for 8bit serialization
                    new_value = new_value.to(torch.float16)
                # quantize module that are going to stay on the cpu so that we offload quantized weights
                if device == "cpu" and param_cls.__name__ == "Int8Params":
                    new_value = param_cls(new_value, requires_grad=old_value.requires_grad, **kwargs).to(0).to("cpu")
                    new_value.CB = new_value.CB.to("cpu")
                    new_value.SCB = new_value.SCB.to("cpu")
                else:
                    new_value = param_cls(new_value, requires_grad=old_value.requires_grad, **kwargs).to(device, non_blocking=prefetch)
            elif param_cls.__name__ in ["QTensor", "QBitsTensor"]:
                new_value = torch.nn.Parameter(new_value, requires_grad=old_value.requires_grad).to(device, non_blocking=prefetch)
            else:
                new_value = param_cls(new_value, requires_grad=old_value.requires_grad).to(device, non_blocking=prefetch)

            module._parameters[tensor_name] = new_value
            if fp16_statistics is not None:
                module._parameters[tensor_name].SCB = fp16_statistics.to(device, non_blocking=prefetch)
                del fp16_statistics
            # as we put the weight to meta, it doesn't have SCB attr anymore. make sure that it is not a meta weight
            if (
                module.__class__.__name__ == "Linear8bitLt"
                and getattr(module.weight, "SCB", None) is None
                and str(module.weight.device) != "meta"
            ):
                # quantize only if necessary
                device_index = torch.device(device).index if torch.device(device).type == "cuda" else None
                if not getattr(module.weight, "SCB", None) and device_index is not None:
                    if module.bias is not None and module.bias.device.type != "meta":
                        # if a bias exists, we need to wait until the bias is set on the correct device
                        module = module.cuda(device_index)
                    elif module.bias is None:
                        # if no bias exists, we can quantize right away
                        module = module.cuda(device_index)
            elif (
                module.__class__.__name__ == "Linear4bit"
                and getattr(module.weight, "quant_state", None) is None
                and str(module.weight.device) != "meta"
            ):
                # quantize only if necessary
                device_index = torch.device(device).index if torch.device(device).type == "cuda" else None
                if not getattr(module.weight, "quant_state", None) and device_index is not None:
                    module.weight = module.weight.cuda(device_index)
    # clean pre and post foward hook
    if device != "cpu":
        clear_device_cache()

    # When handling tied weights, we update tied_params_map to keep track of the tied weights that have already been allocated on the device in
    # order to avoid duplicating memory, see above.
    if (
        tied_params_map is not None
        and old_value.data_ptr() in tied_params_map
        and device not in tied_params_map[old_value.data_ptr()]
    ):
        tied_params_map[old_value.data_ptr()][device] = new_value
    elif (
        value is not None
        and tied_params_map is not None
        and value.data_ptr() in tied_params_map
        and device not in tied_params_map[value.data_ptr()]
    ):
        tied_params_map[value.data_ptr()][device] = new_value


class PrefetchHook(ModelHook):
    """
    Prefetching
    """

    def __init__(
        self,
        module_name: str = "",
        prefetch_name: str = "",
        prefetch_module: nn.Module = None,
        execution_device: Optional[Union[int, str, torch.device]] = None,
        offload: bool = False,
        io_same_device: bool = False,
        weights_map: Optional[Mapping] = None,
        offload_buffers: bool = False,
        place_submodules: bool = False,
        skip_keys: Optional[Union[str, List[str]]] = None,
        tied_params_map: Optional[Dict[int, Dict[torch.device, torch.Tensor]]] = None,
    ):
        self.execution_device = execution_device
        self.offload = offload
        self.io_same_device = io_same_device
        self.weights_map = weights_map
        self.offload_buffers = offload_buffers
        self.place_submodules = place_submodules
        self.skip_keys = skip_keys

        # Will contain the input device when `io_same_device=True`.
        self.input_device = None
        self.param_original_devices = {}
        self.buffer_original_devices = {}
        self.tied_params_names = set()

        # The hook pre_forward/post_forward need to have knowledge of this dictionary, as with offloading we want to avoid duplicating memory
        # for tied weights already loaded on the target execution device.
        self.tied_params_map = tied_params_map

        # Additional map
        self.prefetch_name = prefetch_name
        self.prefetch_module = prefetch_module
        self.module_name = module_name

        self.prefetch_stream = torch.cuda.Stream()

    def __repr__(self):
        return (
            f"PrefetchHook(execution_device={self.execution_device}, offload={self.offload}, "
            f"io_same_device={self.io_same_device}, offload_buffers={self.offload_buffers}, "
            f"place_submodules={self.place_submodules}, skip_keys={repr(self.skip_keys)})"
            f"module_name={self.module_name}, prefetch_name={self.prefetch_name}"
        )

    def init_hook(self, module):
        # In case the AlignDevicesHook is on meta device, ignore tied weights as data_ptr() is then always zero.
        if self.execution_device == "meta" or self.execution_device == torch.device("meta"):
            self.tied_params_map = None

        if not self.offload and self.execution_device is not None:
            # print(f'recurse: {self.place_submodules}')
            for name, _ in named_module_tensors(module, recurse=self.place_submodules):
                set_module_tensor_to_device(module, name, self.execution_device, tied_params_map=self.tied_params_map)
        elif self.offload:
            self.original_devices = {
                name: param.device for name, param in named_module_tensors(module, recurse=self.place_submodules)
            }
            if self.weights_map is None:
                self.weights_map = {
                    name: param.to("cpu")
                    for name, param in named_module_tensors(
                        module, include_buffers=self.offload_buffers, recurse=self.place_submodules
                    )
                }
            for name, _ in named_module_tensors(
                module, include_buffers=self.offload_buffers, recurse=self.place_submodules, remove_non_persistent=True
            ):
                # When using disk offloading, we can not rely on `weights_map[name].data_ptr()` as the reference pointer,
                # as we have no guarantee that safetensors' `file.get_tensor()` will always give the same pointer.
                # As we have no reliable way to track the shared data pointer of tied weights in this case, we use tied_params_names: List[str]
                # to add on the fly pointers to `tied_params_map` in the pre_forward call.
                if (
                    self.tied_params_map is not None
                    and recursive_getattr(module, name).data_ptr() in self.tied_params_map
                ):
                    self.tied_params_names.add(name)

                set_module_tensor_to_device(module, name, "meta")

            if not self.offload_buffers and self.execution_device is not None:
                for name, _ in module.named_buffers(recurse=self.place_submodules):
                    set_module_tensor_to_device(
                        module, name, self.execution_device, tied_params_map=self.tied_params_map
                    )
            elif self.offload_buffers and self.execution_device is not None:
                for name in get_non_persistent_buffers(module, recurse=self.place_submodules):
                    set_module_tensor_to_device(
                        module, name, self.execution_device, tied_params_map=self.tied_params_map
                    )
        return module

    @nvtx.annotate()
    def pre_forward(self, module, *args, **kwargs):
        # logging.info(f'module_name: {self.module_name}')

        if self.io_same_device:
            self.input_device = find_device([args, kwargs])

        self.tied_pointers_to_remove = set()

        # logging.info(f'Prefetch module: {self.prefetch_name}')
        with torch.cuda.stream(self.prefetch_stream):
            for weight_name, _ in named_module_tensors(
                self.prefetch_module,
                include_buffers=self.offload_buffers,
                recurse=self.place_submodules,
                remove_non_persistent=True
            ):
                # logging.info(f'weight_name: {weight_name}')
                fp16_statistics = None
                value = self.weights_map[weight_name]
                if "weight" in weight_name and weight_name.replace("weight", "SCB") in self.weights_map.keys():
                    if value.dtype == torch.int8:
                        fp16_statistics = self.weights_map[weight_name.replace("weight", "SCB")]

                # In case we are using offloading with tied weights, we need to keep track of the offloaded weights
                # that are loaded on device at this point, as we will need to remove them as well from the dictionary
                # self.tied_params_map in order to allow to free memory.
                if weight_name in self.tied_params_names and value.data_ptr() not in self.tied_params_map:
                    self.tied_params_map[value.data_ptr()] = {}

                if (
                    value is not None
                    and self.tied_params_map is not None
                    and value.data_ptr() in self.tied_params_map
                    and self.execution_device not in self.tied_params_map[value.data_ptr()]
                ):
                    self.tied_pointers_to_remove.add((value.data_ptr(), self.execution_device))

                set_module_tensor_to_device(
                    self.prefetch_module,
                    weight_name,
                    self.execution_device,
                    prefetch=True,
                    value=value,
                    fp16_statistics=fp16_statistics,
                    tied_params_map=self.tied_params_map,
                )

        # logging.info(f'After prefetch mem usage: {torch.cuda.max_memory_allocated(self.input_device)/1024**2} MiB')

        return send_to_device(args, self.execution_device), send_to_device(
            kwargs, self.execution_device, skip_keys=self.skip_keys,
        )
    
    @nvtx.annotate()
    def post_forward(self, module, output):
        if self.offload:
            for name, _ in named_module_tensors(
                module,
                include_buffers=self.offload_buffers,
                recurse=self.place_submodules,
                remove_non_persistent=True,
            ):
                set_module_tensor_to_device(module, name, "meta")
                if type(module).__name__ == "Linear8bitLt":
                    module.state.SCB = None
                    module.state.CxB = None

            # We may have loaded tied weights into self.tied_params_map (avoiding to load them several times in e.g. submodules): remove them from
            # this dictionary to allow the garbage collector to do its job.
            for value_pointer, device in self.tied_pointers_to_remove:
                del self.tied_params_map[value_pointer][device]
            self.tied_pointers_to_remove = set()

        if self.io_same_device and self.input_device is not None:
            output = send_to_device(output, self.input_device, skip_keys=self.skip_keys)

        self.prefetch_stream.synchronize()

        return output

    def detach_hook(self, module):
        if self.offload:
            for name, device in self.original_devices.items():
                if device != torch.device("meta"):
                    set_module_tensor_to_device(module, name, device, value=self.weights_map.get(name, None))
        return module


def attach_prefetch_hook(
    module: nn.Module,
    module_name: str = "",
    prefetch_name_map: Dict[str, str] = None,
    module_map: Dict[str, nn.Module] = None,
    execution_device: Optional[Union[torch.device, Dict[str, torch.device]]] = None,
    offload: Union[bool, Dict[str, bool]] = False,
    weights_map: Mapping = None,
    offload_buffers: bool = False,
    skip_keys: Optional[Union[str, List[str]]] = None,
    preload_module_classes: Optional[List[str]] = None,
    tied_params_map: Optional[Dict[int, Dict[torch.device, torch.Tensor]]] = None,
):
    """
    Attaches `PrefetchHook` to all blocks of a given model as needed.

    Args:
        module (`torch.nn.Module`):
            The module where we want to attach the hooks.
        execution_device (`torch.device` or `Dict[str, torch.device]`, *optional*):
            The device on which inputs and model weights should be placed before the forward pass. It can be one device
            for the whole module, or a dictionary mapping module name to device.
        offload (`bool`, *optional*, defaults to `False`):
            Whether or not the weights should be offloaded after the forward pass. It can be one boolean for the whole
            module, or a dictionary mapping module name to boolean.
        weights_map (`Mapping[str, torch.Tensor]`, *optional*):
            When the model weights are offloaded, a (potentially lazy) map from param names to the tensor values.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to include the associated module's buffers when offloading.
        module_name (`str`, *optional*, defaults to `""`):
            The name of the module.
        skip_keys (`str` or `List[str]`, *optional*):
            A list of keys to ignore when moving inputs or outputs between devices.
        preload_module_classes (`List[str]`, *optional*):
            A list of classes whose instances should load all their weights (even in the submodules) at the beginning
            of the forward. This should only be used for classes that have submodules which are registered but not
            called directly during the forward, for instance if a `dense` linear layer is registered, but at forward,
            `dense.weight` and `dense.bias` are used in some operations instead of calling `dense` directly.
        tied_params_map (Optional[Dict[int, Dict[torch.device, torch.Tensor]]], *optional*, defaults to `None`):
            A map of data pointers to dictionaries of devices to already dispatched tied weights. For a given execution
            device, this parameter is useful to reuse the first available pointer of a shared weight for all others,
            instead of duplicating memory.
    """
    # If one device and one offload, we've got one hook.
    # if not isinstance(execution_device, Mapping) and not isinstance(offload, dict):
    #     if not offload:
    #         hook = PrefetchHook(
    #             execution_device=execution_device,
    #             io_same_device=True,
    #             skip_keys=skip_keys,
    #             place_submodules=True,
    #             tied_params_map=tied_params_map,
    #         )
    #         add_hook_to_module(module, hook)
    #     else:
    #         attach_prefetch_hook(
    #             module,
    #             execution_device=execution_device,
    #             offload=True,
    #             weights_map=weights_map,
    #             offload_buffers=offload_buffers,
    #             module_name=module_name,
    #             skip_keys=skip_keys,
    #             tied_params_map=tied_params_map,
    #         )
    #     return

    if not isinstance(execution_device, Mapping):
        execution_device = {key: execution_device for key in offload.keys()}
    if not isinstance(offload, Mapping):
        offload = {key: offload for key in execution_device.keys()}

    if (   
        module_name in execution_device 
        and module_name in offload 
        and not offload[module_name] 
    ):
        if module_name not in prefetch_name_map.keys():
            hook = AlignDevicesHook(
                execution_device=execution_device[module_name],
                offload_buffers=offload_buffers,
                io_same_device=(module_name == ""),
                place_submodules=True,
                skip_keys=skip_keys,
                tied_params_map=tied_params_map
            )
        else:
            # logging.info(f'On GPU but need prefetch: {module_name}')
            prefetch_name = prefetch_name_map[module_name]
            prefix = f"{prefetch_name}." if len(module_name) > 0 else ""
            prefixed_weights_map = PrefixedDataset(weights_map, prefix)
            hook = PrefetchHook(
                module_name=module_name,
                prefetch_name=prefetch_name,
                prefetch_module=module_map[prefetch_name],
                execution_device=execution_device[module_name],
                offload_buffers=offload_buffers,
                io_same_device=(module_name == ""),
                weights_map=prefixed_weights_map,
                place_submodules=True,
                skip_keys=skip_keys,
                tied_params_map=tied_params_map,
            )
        add_hook_to_module(module, hook)
    elif (
        module_name in execution_device
        and module_name in offload
        and offload[module_name]
    ):
        if module_name in prefetch_name_map.keys():
            # logging.info(f'On CPU need prefetch: {module_name}')
            prefetch_name = prefetch_name_map[module_name]
            prefix = f"{prefetch_name}." if len(module_name) > 0 else ""
            prefixed_weights_map = PrefixedDataset(weights_map, prefix)
            hook = PrefetchHook(
                module_name=module_name,
                prefetch_name=prefetch_name,
                prefetch_module=module_map[prefetch_name],
                execution_device=execution_device[module_name],
                offload_buffers=offload_buffers,
                offload=offload,
                io_same_device=(module_name == ""),
                weights_map=prefixed_weights_map,
                place_submodules=True,
                skip_keys=skip_keys,
                tied_params_map=tied_params_map,
            )
        else:
            # logging.info(f'On CPU No need to prefetch: {module_name}')
            prefix = f"{module_name}." if len(module_name) > 0 else ""
            prefixed_weights_map = PrefixedDataset(weights_map, prefix)
            hook = AlignDevicesHook(
                execution_device=execution_device[module_name],
                offload_buffers=offload_buffers,
                offload=True,
                io_same_device=(module_name == ""),
                weights_map=prefixed_weights_map,
                place_submodules=True,
                skip_keys=skip_keys,
                tied_params_map=tied_params_map
            )
        add_hook_to_module(module, hook)
    # elif module_name == "":
    #     hook = PrefetchHook(
    #         execution_device=execution_device.get(""),
    #         io_same_device=True,
    #         skip_keys=skip_keys,
    #         tied_params_map=tied_params_map,
    #     )
    #     add_hook_to_module(module, hook)

    for child_name, child in module.named_children():
        child_name = f"{module_name}.{child_name}" if len(module_name) > 0 else child_name
        attach_prefetch_hook(
            child,
            module_name=child_name,
            prefetch_name_map=prefetch_name_map,
            module_map=module_map,
            execution_device=execution_device,
            offload=offload,
            weights_map=weights_map,
            offload_buffers=offload_buffers,
            preload_module_classes=preload_module_classes,
            skip_keys=skip_keys,
            tied_params_map=tied_params_map,
        )


def dispatch_model_with_prefetch(
    model: nn.Module,
    device_map: Dict[str, Union[str, int, torch.device]],
    prefetch_name_map: Dict[str, str] = None,
    module_map: Dict[str, nn.Module] = None,
    main_device: Optional[torch.device] = None,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
    offload_dir: Optional[Union[str, os.PathLike]] = None,
    offload_index: Optional[Dict[str, str]] = None,
    offload_buffers: bool = False,
    skip_keys: Optional[Union[str, List[str]]] = None,
    preload_module_classes: Optional[List[str]] = None,
    force_hooks: bool = False,
):
    # Error early if the device map is incomplete.
    check_device_map(model, device_map)

    # We need to force hook for quantized model that can't be moved with to()
    if getattr(model, "quantization_method", "bitsandbytes") == "bitsandbytes":
        # since bnb 0.43.2, we can move 4-bit model
        if getattr(model, "is_loaded_in_8bit", False) or (
            getattr(model, "is_loaded_in_4bit", False) and not is_bnb_available(min_version="0.43.2")
        ):
            force_hooks = True

    # We attach hooks if the device_map has at least 2 different devices or if
    # force_hooks is set to `True`. Otherwise, the model in already loaded
    # in the unique device and the user can decide where to dispatch the model.
    # If the model is quantized, we always force-dispatch the model
    if (len(set(device_map.values())) > 1) or force_hooks:
        if main_device is None:
            if set(device_map.values()) == {"cpu"} or set(device_map.values()) == {"cpu", "disk"}:
                main_device = "cpu"
            else:
                main_device = [d for d in device_map.values() if d not in ["cpu", "disk"]][0]

        if main_device != "cpu":
            cpu_modules = [name for name, device in device_map.items() if device == "cpu"]
            if state_dict is None and len(cpu_modules) > 0:
                state_dict = extract_submodules_state_dict(model.state_dict(), cpu_modules)

        disk_modules = [name for name, device in device_map.items() if device == "disk"]
        if offload_dir is None and offload_index is None and len(disk_modules) > 0:
            raise ValueError(
                "We need an `offload_dir` to dispatch this model according to this `device_map`, the following submodules "
                f"need to be offloaded: {', '.join(disk_modules)}."
            )
        if (
            len(disk_modules) > 0
            and offload_index is None
            and (not os.path.isdir(offload_dir) or not os.path.isfile(os.path.join(offload_dir, "index.json")))
        ):
            disk_state_dict = extract_submodules_state_dict(model.state_dict(), disk_modules)
            offload_state_dict(offload_dir, disk_state_dict)

        execution_device = {
            name: main_device if device in ["cpu", "disk"] else device for name, device in device_map.items()
        }
        execution_device[""] = main_device
        offloaded_devices = ["disk"] if main_device == "cpu" or main_device == "mps" else ["cpu", "disk"]
        offload = {name: device in offloaded_devices for name, device in device_map.items()}
        save_folder = offload_dir if len(disk_modules) > 0 else None
        if state_dict is not None or save_folder is not None or offload_index is not None:
            device = main_device if offload_index is not None else None
            weights_map = OffloadedWeightsLoader(
                state_dict=state_dict, save_folder=save_folder, index=offload_index, device=device
            )
        else:
            weights_map = None

        # When dispatching the model's parameters to the devices specified in device_map, we want to avoid allocating memory several times for the
        # tied parameters. The dictionary tied_params_map keeps track of the already allocated data for a given tied parameter (represented by its
        # original pointer) on each devices.
        # print(f'weights_map: {weights_map}')
        tied_params = find_tied_parameters(model)

        tied_params_map = {}
        for group in tied_params:
            for param_name in group:
                # data_ptr() is enough here, as `find_tied_parameters` finds tied params simply by comparing `param1 is param2`, so we don't need
                # to care about views of tensors through storage_offset.

                data_ptr = recursive_getattr(model, param_name).data_ptr()
                tied_params_map[data_ptr] = {}

                # Note: To handle the disk offloading case, we can not simply use weights_map[param_name].data_ptr() as the reference pointer,
                # as we have no guarantee that safetensors' `file.get_tensor()` will always give the same pointer.

        attach_prefetch_hook(
            model,
            prefetch_name_map=prefetch_name_map,
            module_map=module_map,
            execution_device=execution_device,
            offload=offload,
            offload_buffers=offload_buffers,
            weights_map=weights_map,
            skip_keys=skip_keys,
            preload_module_classes=preload_module_classes,
            tied_params_map=tied_params_map,
        )

        # warn if there is any params on the meta device
        offloaded_devices_str = " and ".join(
            [device for device in set(device_map.values()) if device in ("cpu", "disk")]
        )
        if len(offloaded_devices_str) > 0:
            logging.warning(
                f"Some parameters are on the meta device because they were offloaded to the {offloaded_devices_str}."
            )

        # Attaching the hook may break tied weights, so we retie them
        retie_parameters(model, tied_params)

        # add warning to cuda and to method
        def add_warning(fn, model):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                warning_msg = "You shouldn't move a model that is dispatched using accelerate hooks."
                if str(fn.__name__) == "to":
                    to_device = torch._C._nn._parse_to(*args, **kwargs)[0]
                    if to_device is not None:
                        logging.warning(warning_msg)
                else:
                    logging.warning(warning_msg)
                for param in model.parameters():
                    if param.device == torch.device("meta"):
                        raise RuntimeError("You can't move a model that has some modules offloaded to cpu or disk.")
                return fn(*args, **kwargs)

            return wrapper

        # Make sure to update _accelerate_added_attributes in hooks.py if you add any hook
        model.to = add_warning(model.to, model)
        if is_npu_available():
            model.npu = add_warning(model.npu, model)
        elif is_mlu_available():
            model.mlu = add_warning(model.mlu, model)
        elif is_musa_available():
            model.musa = add_warning(model.musa, model)
        elif is_xpu_available():
            model.xpu = add_warning(model.xpu, model)
        else:
            model.cuda = add_warning(model.cuda, model)

        # Check if we are using multi-gpus with RTX 4000 series
        use_multi_gpu = len([device for device in set(device_map.values()) if device not in ("cpu", "disk")]) > 1
        if use_multi_gpu and not check_cuda_p2p_ib_support():
            logging.warning(
                "We've detected an older driver with an RTX 4000 series GPU. These drivers have issues with P2P. "
                "This can affect the multi-gpu inference when using accelerate device_map."
                "Please make sure to update your driver to the latest version which resolves this."
            )
    else:
        device = list(device_map.values())[0]
        # `torch.Tensor.to(<int num>)` is not supported by `torch_npu` (see this [issue](https://github.com/Ascend/pytorch/issues/16)).
        if is_npu_available() and isinstance(device, int):
            device = f"npu:{device}"
        elif is_mlu_available() and isinstance(device, int):
            device = f"mlu:{device}"
        elif is_musa_available() and isinstance(device, int):
            device = f"musa:{device}"
        elif is_xpu_available() and isinstance(device, int):
            device = f"xpu:{device}"
        if device != "disk":
            model.to(device)
        else:
            raise ValueError(
                "You are trying to offload the whole model to the disk. Please use the `disk_offload` function instead."
            )
    # Convert OrderedDict back to dict for easier usage
    model.hf_device_map = dict(device_map)
    return model