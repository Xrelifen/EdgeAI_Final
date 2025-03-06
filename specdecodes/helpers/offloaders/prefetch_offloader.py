import torch
from torch import nn
from ..model_orders import MODEL_TYPE_GET_LAYER_ORDER


def find_child(model, name: str) -> nn.Module:
    module_tree = name.split(".")
    parent = model
    for m in module_tree:
        # parent = parent._modules[m]
        parent = getattr(parent, m)
    return parent


def check_device_map(model: nn.Module, device_map: dict):
    """
    Checks a device map covers everything in a given model.

    Args:
        model (`torch.nn.Module`): The model to check the device map against.
        device_map (`Dict[str, Union[int, str, torch.device]]`): The device map to check.
    """
    all_model_tensors = [name for name, _ in model.state_dict().items()]
    for module_name in device_map.keys():
        if module_name == "":
            all_model_tensors.clear()
            break
        else:
            all_model_tensors = [
                name
                for name in all_model_tensors
                if not name == module_name and not name.startswith(module_name + ".")
            ]
    if len(all_model_tensors) > 0:
        non_covered_params = ", ".join(all_model_tensors)
        raise ValueError(
            f"The device_map provided does not give any device for the following parameters: {non_covered_params}"
        )

class PrefetchOffloader:
    def __init__(self, model, device_map, record_stream=True):
        # 1) Sanity-check the device map coverage and correctness
        check_device_map(model, device_map)

        # 2) Initialize each layer's parameters/buffers:
        #    - If device_map == 'cpu', move to pinned CPU memory
        #    - Else move to specified GPU.
        for name, dev_str in device_map.items():
            layer = find_child(model, name)
            for p in layer.parameters():
                if dev_str == 'cpu':
                    p.data = p.data.cpu().pin_memory()
                else:
                    p.data = p.data.to(dev_str)  # e.g. "cuda:0"
            for b in layer.buffers():
                if dev_str == 'cpu':
                    b.data = b.data.cpu().pin_memory()
                else:
                    b.data = b.data.to(dev_str)
        
        # Assert first layer is on a GPU
        assert model.model.embed_tokens.weight.device.type == 'cuda'
        self.gpu_device = model.model.embed_tokens.weight.device

        # 3) Save the pinned CPU or GPU "original" data references,
        #    so we can restore them for CPU layers after forward.
        self.param_dict = {p: p.data for p in model.parameters()}

        # 4) A separate CUDA stream for asynchronous CPU->GPU copies
        self.stream = torch.cuda.Stream()
        self.record_stream = record_stream

        # 5) Get an ordered list of layers (depends on model type).
        layer_order = MODEL_TYPE_GET_LAYER_ORDER[model.config.model_type](model.config)

        # 6) Walk through layers in order, registering:
        #    (a) If next layer is CPU-based, attach a forward_hook that enqueues copy -> self.gpu_device.
        #    (b) If the layer itself is CPU-based, attach:
        #       - A forward_pre_hook that waits for the copy to finish.
        #       - A forward_hook that offloads the layer back to CPU after forward.
        for i, layer_name in enumerate(layer_order):
            current_layer = find_child(model, layer_name)
            current_dev_str = device_map.get(layer_name, 'cpu')  # default to 'cpu' if not present

            # (a) Prefetch next CPU layer at the end of this layer's forward pass
            if i + 1 < len(layer_order):
                next_name = layer_order[i + 1]
                next_dev_str = device_map.get(next_name, 'cpu')
                if next_dev_str == 'cpu':
                    next_layer = find_child(model, next_name)
                    current_layer.register_forward_hook(
                        self._create_prefetch_hook(next_layer)
                    )

            # (b) If this layer is CPU-based, wait for the copy + offload back afterwards
            if current_dev_str == 'cpu':
                # Wait for the async copy right before forward
                current_layer.register_forward_pre_hook(self._create_wait_hook())
                # Move weights back to pinned CPU after forward
                current_layer.register_forward_hook(self._create_offload_hook())

    def _create_prefetch_hook(self, next_layer):
        """
        A forward_hook that runs immediately *after* the current layer's forward pass,
        scheduling an async copy of `next_layer` onto self.gpu_device.
        """
        @torch.compiler.disable()
        def prefetch_hook(module, inputs, output):
            # Copy next_layer's parameters from pinned CPU -> self.gpu_device asynchronously
            with torch.cuda.stream(self.stream):
                for p in next_layer.parameters():
                    p.data = p.data.to(self.gpu_device, non_blocking=True)
                    if self.record_stream:
                        p.data.record_stream(torch.cuda.current_stream())
                for b in next_layer.buffers():
                    b.data = b.data.to(self.gpu_device, non_blocking=True)
                    if self.record_stream:
                        b.data.record_stream(torch.cuda.current_stream())

            # IMPORTANT: Do *not* force a wait here. We want real overlap:
            # The main forward stream continues on the next ops
            # while `self.stream` is busy copying the next layer's data.
        return prefetch_hook

    def _create_wait_hook(self):
        """
        A forward_pre_hook that runs right before this CPU-based layer's forward pass.
        It ensures that any scheduled async copy in self.stream is complete
        by waiting on it from the main/forward stream.
        """
        @torch.compiler.disable()
        def wait_hook(module, inputs):
            # Wait for asynchronous copies in self.stream to complete
            torch.cuda.current_stream().wait_stream(self.stream)
        return wait_hook

    def _create_offload_hook(self):
        """
        A forward_hook that runs after this CPU-based layer finishes its forward pass,
        restoring parameters/buffers back to pinned CPU memory (self.param_dict).
        """
        @torch.compiler.disable()
        def offload_hook(module, inputs, output):
            # If we're *not* using record_stream, we need to ensure the GPU is done
            # with the forward pass before modifying p.data
            if not self.record_stream:
                torch.cuda.current_stream().synchronize()

            # Offload back to CPU pinned memory
            for p in module.parameters():
                p.data = self.param_dict[p]
            for b in module.buffers():
                b.data = self.param_dict[b]
        return offload_hook