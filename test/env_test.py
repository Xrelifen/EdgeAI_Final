import torch
print("Torch version:", torch.__version__, torch.version.cuda)
print(torch.cuda.is_available())
print(torch._C._GLIBCXX_USE_CXX11_ABI)
import flashinfer
print("FlashInfer version:", flashinfer.__version__)