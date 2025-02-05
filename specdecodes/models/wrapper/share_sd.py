import json
import logging
import os
import time
import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria
import prettytable as pt
from .sd import SDWrapper

import nvtx

class ShareSDWrapper(SDWrapper):
    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_processor: LogitsProcessorList,
        do_sample: bool,
        **model_kwargs,
    ):
        """
        Generate sequence of tokens with speculative decoding.

        This method consists of two main stages: prefill and decode.

        Prefill Stage:
        - Perform the model's initial forward pass.
        - Sample a token and append it to the input_ids.

        Decode Stage (with speculative decoding):
        - Iterate through the following steps:
            1. Perform SSM speculative sampling, returns sampled tokens in tree form.
            2. Decode the sampled tokens in parallel with the language model (LLM), generating probabilities for each token.
            3. Verify the sampled tokens by accepting or rejecting them, corresponding to the probabilities.
            4. Update the key-value cache, input_ids, and hidden_states accordingly.

        Args:
            input_ids (torch.LongTensor): The input token IDs. 
            stopping_criteria (StoppingCriteria): The criteria to stop the generation.
            logits_processor (LogitsProcessor): The processor to modify the logits.
            do_sample (bool): Whether to sample tokens during generation. If False, the generation will be deterministic.

        Returns:
            input_ids (torch.LongTensor): The generated token IDs.
        """
        assert self.llm is not None, "LLM model must be provided"
        assert self.ssm is not None, "SSM model must be provided"
        assert self.tokenizer is not None, "Tokenizer must be provided"

        # * clone input_ids 
        input_ids = input_ids.clone()
        batch_size, org_input_len = input_ids.shape
        assert batch_size == 1, "Only support batch_size=1 for now."

        # * prepare kv-cache
        # Raise error if max_length not set while using static cache
        if stopping_criteria.max_length is None:
            if self.cache_implementation == "static":
                raise ValueError(
                    "max_length is not set. Only 'dynamic' kv-cache is supported when max_length is unspecified."
                )
            
        if model_kwargs.get("past_key_values") is not None:
            past_key_values = model_kwargs["past_key_values"]
            max_cache_len = getattr(past_key_values, "max_cache_len", None)
        else:
            raise ValueError("past_key_values is not provided")
            
        
        self._init_tree_mask(self.draft_params.max_verify_tokens, max_cache_len, device=input_ids.device)
        cache_position = torch.arange(org_input_len, dtype=torch.long, device=input_ids.device)

        # * prefill stage
        with nvtx.annotate("prefill", color="orange"):
            #! Not needed after torch version=2.7, where torch.compiler.set_stance("force_eager") is introduced
            # with torch.compiler.set_stance("force_eager"):
            #     outputs = self.llm(
            outputs = self.llm.prefill_forward(
                input_ids,
                past_key_values=past_key_values,
                output_hidden_states=True, 
                cache_position=cache_position,
                num_logits_to_keep=1,
            )
            next_token_logits = outputs.logits
            hidden_states = outputs.hidden_states[-1]
            del outputs

        with nvtx.annotate("sample tokens"):
            next_tokens = self._sample_token(next_token_logits, logits_processor, do_sample)
            sampled_tokens = next_tokens

        with nvtx.annotate("update data"):
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            cache_position = torch.arange(org_input_len, org_input_len+self.draft_params.max_verify_tokens, dtype=torch.long, device=input_ids.device)

        with nvtx.annotate("decoding"):
            finished = False
            while not finished:
                # * speculate
                with nvtx.annotate("speculate", color="cyan"):
                    test_tokens = sampled_tokens[:, -1:].clone(memory_format=torch.contiguous_format)
                    tree = self._speculate(test_tokens, hidden_states, past_key_values)

                # * tree decoding
                with nvtx.annotate("tree_decoding", color="orange"):
                    prev_kv_len = past_key_values.get_seq_length()
                    outputs = self._tree_decoding(tree, past_key_values, position_offset=input_ids.shape[1]-1, cache_position=cache_position, device=hidden_states.device)
                    
                    next_token_logits = outputs.logits
                    hidden_states = outputs.hidden_states[-1]
                    del outputs

                # * verify
                with nvtx.annotate("verify"):
                    sampled_tokens, hidden_indices, _ = self._verify(
                                                            tree, next_token_logits, 
                                                            logits_processor,
                                                            do_sample
                                                        )
                    
                    sampled_tokens = sampled_tokens.to(next_tokens.device, non_blocking=True)
                    hidden_indices = hidden_indices.to(hidden_states.device, non_blocking=True)
                
                with nvtx.annotate("reorder kv"):
                    past_key_values.reorder_cache_with_offset(hidden_indices, offset=prev_kv_len, new_chunk_len=self.draft_params.max_verify_tokens, dim=2)

                # * update input_ids, hidden_states, and cache_position
                with nvtx.annotate("update data"):
                    input_ids = torch.cat([input_ids, sampled_tokens], dim=-1)
                    hidden_states = hidden_states[:, hidden_indices].clone()
                    cache_position += sampled_tokens.shape[1]
                
                # * check stopping criteria
                with nvtx.annotate("stopping criteria"):
                    finished = stopping_criteria(input_ids, None)
                    finished = finished.item()
                
        return input_ids
    

class ProfileShareSDWrapper(ShareSDWrapper):
    def __init__(self, out_dir="specdecodes/experiments/profile_data", prefix="sd", *model_args, **kwargs):
        super().__init__(*model_args, **kwargs)
        self.out_dir = out_dir
        self.prefix = prefix
        
        self.profile_data = {}
        self.sampled_count = 1 # assume first token is sampled (prefill stage)
        self.iter_count = 1 # assume first step is done (prefill stage)
        
        self.exp_log = {}
        self.draft_events = []
        self.target_events = []
        self.verify_events = []
        
        self.disable_logging = False
        
    def _speculate(self, *model_args, **kwargs):
        if self.disable_logging:
            return super()._speculate(*model_args, **kwargs)
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        root = super()._speculate(*model_args, **kwargs)
        end_event.record()
        
        self.draft_events.append((start_event, end_event))
        return root
    
    def _tree_decoding(self, *model_args, **kwargs):
        if self.disable_logging:
            return super()._tree_decoding(*model_args, **kwargs)
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        outputs = super()._tree_decoding(*model_args, **kwargs)
        end_event.record()
        
        self.target_events.append((start_event, end_event))
        return outputs
    
    def _verify(self, tree, *model_args, **kwargs):
        if self.disable_logging:
            return super()._verify(tree, *model_args, **kwargs)
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        sampled_tokens, hidden_indices, (total_len, accept_len) = super()._verify(tree, *model_args, **kwargs)
        end_event.record()
        
        self.verify_events.append((start_event, end_event))
        
        # tokenize id to text for visualization
        # nodes = list(preorder_iter(root))
        # for node in nodes:
        #     node.id = self.tokenizer.decode(torch.tensor([node.id]), clean_up_tokenization_spaces=False)
        
        # profile data
        # json_graph = tree_to_nested_dict(root, name_key="name", attr_dict={"id": "id", "prob": "prob", "global_prob": "global_prob"})
        # sampled_tokens_list = sampled_tokens.squeeze(0).tolist()
        # self.profile_data[self.iter_count] = {}
        # self.profile_data[self.iter_count]["draft_tree"] = json_graph
        # self.profile_data[self.iter_count]["sampled_tokens"] = sampled_tokens_list
        
        # create profile data if not exist
        self.profile_data['iter'] = self.profile_data.get('iter', [])
        self.profile_data['total_len'] = self.profile_data.get('total_len', [])
        self.profile_data['accept_len'] = self.profile_data.get('accept_len', [])
            
        sampled_tokens_list = sampled_tokens.squeeze(0).tolist()
        self.profile_data['iter'].append(sampled_tokens_list)
        self.profile_data['total_len'].append(total_len)
        self.profile_data['accept_len'].append(accept_len)
        # logging
        logging.debug(
            f"Total: {tree.size()},"\
            f"\tPredicted ({accept_len}/{total_len}): {self.tokenizer.batch_decode(sampled_tokens.squeeze(0), clean_up_tokenization_spaces=False)}"
        )
        
        # update stats
        self.sampled_count += len(sampled_tokens[0])
        self.iter_count += 1
        
        return sampled_tokens, hidden_indices, (total_len, accept_len)
    
    def compute_average_times(self):
        """
        Synchronize once at the end, then compute average
        draft and target times from the recorded CUDA events.
        """
        # Ensure all CUDA kernels are done
        torch.cuda.synchronize()

        # Compute total time for draft iterations
        draft_time_total_ms = 0.0
        for (start_event, end_event) in self.draft_events:
            draft_time_total_ms += start_event.elapsed_time(end_event)  # returns time in ms

        # Compute total time for target iterations
        target_time_total_ms = 0.0
        for (start_event, end_event) in self.target_events:
            target_time_total_ms += start_event.elapsed_time(end_event)
            
        # Compute total time for verify iterations
        verify_time_total_ms = 0.0
        for (start_event, end_event) in self.verify_events:
            verify_time_total_ms += start_event.elapsed_time(end_event)

        # Average times (in milliseconds)
        draft_avg_ms = draft_time_total_ms / max(len(self.draft_events), 1)
        target_avg_ms = target_time_total_ms / max(len(self.target_events), 1)
        verify_avg_ms = verify_time_total_ms / max(len(self.verify_events), 1)

        # Convert to seconds if you prefer
        draft_avg_s = draft_avg_ms / 1000.0
        target_avg_s = target_avg_ms / 1000.0
        verify_avg_s = verify_avg_ms / 1000.0

        return draft_avg_s, target_avg_s, verify_avg_s
    
    def _generate(self, input_ids: torch.LongTensor, *model_args, **kwargs):
        if self.disable_logging:
            return super()._generate(input_ids, *model_args, **kwargs)
        
        self.profile_data = {}
        self.sampled_count = 1 # assume first token is sampled (prefill stage)
        self.iter_count = 1 # assume first step is done (prefill stage)
        
        self.exp_log = {}
        self.draft_events = []
        self.target_events = []
        self.verify_events = []
        
        cur_time = time.strftime("%Y%m%d-%H%M%S")
        # prepare output directory
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)
            out_path = os.path.join(self.out_dir, f"{self.prefix}_{cur_time}.json")
        else:
            out_path = None
        
        # run generation
        org_input_len = len(input_ids[0])
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        input_ids = super()._generate(input_ids, *model_args, **kwargs)
        end_event.record()
        
        # Make sure all CUDA ops have finished before measuring
        torch.cuda.synchronize()
        
        # Elapsed time in milliseconds
        elapsed_time_ms = start_event.elapsed_time(end_event)
        elapsed_time_s = elapsed_time_ms / 1000.0
        
        # compute stats
        total_sampled = self.sampled_count
        total_iterations = self.iter_count
        avg_sampled = total_sampled / total_iterations
        depth = max(self.profile_data['total_len']) + 1
        
        # alpha (node)
        total_lens = torch.bincount( torch.tensor(self.profile_data['total_len']), minlength=depth)
        accept_lens = torch.bincount( torch.tensor(self.profile_data['accept_len']), minlength=depth)
        depth_total_cnt = total_lens + total_lens.sum() - total_lens.cumsum(dim=-1) # reverse cumsum
        depth_total_cnt = depth_total_cnt[1:] # remove first element
        depth_accept_cnt = accept_lens + accept_lens.sum() - accept_lens.cumsum(dim=-1) # reverse cumsum
        depth_accept_cnt = depth_accept_cnt[1:] # remove first element
        alpha_per_node = depth_accept_cnt.float() / depth_total_cnt.float()
        
        # aLive ratio
        depth_alive_rate = depth_total_cnt.float() / depth_total_cnt[0]
        
        # alpha (depth)
        sampled_lens = torch.tensor([len(sampled_tokens) for sampled_tokens in self.profile_data["iter"]])
        sampled_len_bins = torch.bincount(sampled_lens, minlength=depth+1)
        depth_total_cnt = sampled_len_bins + sampled_len_bins.sum() - sampled_len_bins.cumsum(dim=-1) # reverse cumsum
        depth_accept_cnt = depth_total_cnt - sampled_len_bins
        depth_total_cnt = depth_total_cnt[1:depth]
        depth_accept_cnt = depth_accept_cnt[1:depth]
        alpha_per_depth = depth_accept_cnt.float() / depth_total_cnt.float()
        
        # log stats
        tb = pt.PrettyTable()
        tb.field_names = [ "Summary \ Depth" ] + [ f"{i}" for i in range(1, depth) ]
        tb.add_row([ "Trials count" ] + [ f"{val}" for val in depth_total_cnt.tolist() ])
        tb.add_row([ "Accept count" ] + [ f"{val}" for val in depth_accept_cnt.tolist() ])
        tb.add_row([ "Alpha (node)" ] + [ f"{val:.2f}" for val in alpha_per_node.tolist() ])
        tb.add_row([ "Alpha (depth)" ] + [ f"{val:.2f}" for val in alpha_per_depth.tolist() ])
        tb.add_row([ "Alive ratio" ] + [ f"{val:.2f}" for val in depth_alive_rate.tolist() ])
        logging.info(
            f"Total sampled: {total_sampled},"\
            f"\tTotal iterations: {total_iterations},"\
            f"\tAverage sampled: {avg_sampled:.2f}"\
            f"\n{tb}"
        )
        
        # save profile data
        self.profile_data["total_sampled"] = total_sampled
        self.profile_data["total_iterations"] = total_iterations
        self.profile_data["average_sampled"] = avg_sampled
        if self.out_dir is not None:
            with open(out_path, "w") as f:
                json.dump(self.profile_data, f)
                
        # save exp_log
        avg_draft_s, avg_target_s, avg_verify_s = self.compute_average_times()
        self.exp_log['avg_draft_time'] = avg_draft_s
        self.exp_log['avg_target_time'] = avg_target_s
        self.exp_log['avg_verify_time'] = avg_verify_s
        
        self.exp_log['avg_sampled'] = avg_sampled
        self.exp_log['n_tokens'] = len(input_ids[0][org_input_len:])
        self.exp_log['tput'] = len(input_ids[0][org_input_len:]) / elapsed_time_s
        logging.info(
            f"Average draft time: {self.exp_log['avg_draft_time']:.4f},"\
            f"\tAverage target time: {self.exp_log['avg_target_time']:.4f},"\
            f"\tAverage verify time: {self.exp_log['avg_verify_time']:.4f}"
            f"\nGenerated {self.exp_log['n_tokens']} tokens in {elapsed_time_s:.2f}s, throughput: {self.exp_log['tput']:.2f} tokens/s"
        )
        
        return input_ids