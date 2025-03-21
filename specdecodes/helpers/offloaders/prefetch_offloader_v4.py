import torch
from torch import nn
from ..model_layer_orders import MODEL_TYPE_GET_LAYER_ORDER
from ..utils import find_child, get_tensors, check_device_map


def trim_layer_number(name: str) -> str:
    """Removes numeric fields from a dotted path (e.g. 'layer.0' -> 'layer')."""
    return ".".join(x for x in name.split(".") if not x.isdigit())

class PrefetchOffloader:
    def __init__(self, model: nn.Module, device_map: dict, record_stream=False):
        check_device_map(model, device_map)
        self.gpu_device = device_map["model.embed_tokens"]
        self.cpu_tensors = {}
        self.stream = torch.cuda.Stream()
        self.record_stream = record_stream

        self._cache_cpu_layers(model, device_map)
        assert model.model.embed_tokens.weight.device.type == "cuda"

        # V3 Prefetch Strategy: (V4: inline copy version)
        # 1. Load the first CPU layer to GPU, This layer will prefetch the next CPU layer
        # 2. The next CPU layer will prefetch the next CPU layer, and so on
        # 3. The last CPU layer will prefetch the first CPU layer

        layer_order = MODEL_TYPE_GET_LAYER_ORDER[model.config.model_type](model.config)
        cpu_layer_order = [name for name in layer_order if device_map.get(name) == "cpu"]

        # Find first 'cpu' layer
        if cpu_layer_order == []:
            raise ValueError("No CPU layer found in the model.")
        first_name = cpu_layer_order[0]
        first_cpu_layer = find_child(model, first_name)

        # Copy the first CPU layer to GPU
        for p, c in zip(get_tensors(first_cpu_layer), self.cpu_tensors[first_name]):
            p.data.copy_(c)

        # Connect subsequent CPU layers in a chain
        current_layer = first_cpu_layer
        for name in cpu_layer_order[1:]:
            if device_map.get(name) == "cpu":
                next_layer = find_child(model, name)
                current_layer.register_forward_pre_hook(self._create_prefetch_hook(next_layer, self.cpu_tensors[name]))
                current_layer = next_layer
                
                current_layer.register_forward_pre_hook(self._create_wait_hook())

        # Connect the last CPU layer to the first CPU layer
        if current_layer != first_cpu_layer: # If there is only one CPU layer, no need to prefetch
            # Set up pre-hook (wait for copy) and post-hook (offload) for the first CPU layer
            first_cpu_layer.register_forward_pre_hook(self._create_wait_hook(), prepend=True) # Prepend to ensure wait runs before prefetch
            # The last CPU layer prefetches the first CPU layer (forming a loop)
            current_layer.register_forward_pre_hook(self._create_prefetch_hook(first_cpu_layer, self.cpu_tensors[first_name]))

    def _cache_cpu_layers(self, model, device_map):
        """Moves CPU layers to pinned memory and creates GPU-shaped placeholders."""
        tensor_cache = {}
        for name, dev_str in device_map.items():
            layer = find_child(model, name)
            if dev_str == "cpu":
                trimmed = trim_layer_number(name)
                if trimmed not in tensor_cache:
                    placeholders = [torch.zeros_like(p, device=self.gpu_device)
                                    for p in get_tensors(layer)]
                    tensor_cache[trimmed] = placeholders

                pinned = []
                for i, p in enumerate(get_tensors(layer)):
                    pinned.append(p.data.cpu().pin_memory())
                    p.data = tensor_cache[trimmed][i]
                self.cpu_tensors[name] = pinned
            else:
                # Move params/buffers directly to their specified device
                for p in get_tensors(layer):
                    p.data = p.data.to(dev_str)

    def _create_prefetch_hook(self, next_layer: nn.Module, cpu_params):
        """Schedules async CPU->GPU copy for `next_layer` immediately after current layer's forward."""
        def hook(module, inputs):
            with torch.cuda.stream(self.stream):
                for p, c in zip(get_tensors(next_layer), cpu_params):
                    p.data.copy_(c, non_blocking=True)
                    if self.record_stream:
                        p.data.record_stream(torch.cuda.current_stream())
        return hook

    def _create_wait_hook(self):
        """Waits for any pending async copies in self.stream to finish before forward execution."""
        def hook(module, inputs):
            torch.cuda.current_stream().wait_stream(self.stream)
        return hook