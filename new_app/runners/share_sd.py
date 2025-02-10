
from configs.app import run_app
from .base import BaseRunner

from specdecodes.models.utils.modeling_utils import DraftParams
from specdecodes.models import SSM_ShareSD, ProfileShareSDWrapper

class MyRunner(BaseRunner):
    def __init__(self):
        super().__init__()
        self.llm_path = "meta-llama/Llama-3.1-8B-Instruct"
        self.draft_params = DraftParams(
            max_depth=12,
            topk_len=1,
            max_verify_tokens=64,
            min_accept_prob=1e-8,
        )
        self.offload_recipe = None
        self.vram_limit = 8 # in GB
    
    def _load_draft_model(self, model, tokenizer, draft_path):
        draft_model = SSM_ShareSD.from_pretrained(
            model,
            torch_dtype=self.dtype,
            eos_token_id=tokenizer.eos_token_id,
        )
        return draft_model
    
    def _load_pipeline_method(self, *args):
        return ProfileShareSDWrapper(*args)
    
    
if __name__ == "__main__":
    run_app(MyRunner())