import torch
import torch.nn as nn
from safetensors.torch import load_model
import os

from bigtree import Node
from bigtree import preorder_iter, levelorder_iter, shift_nodes, find_attrs

from ..utils import invert_mask, sampling_without_replacement, cuda_graph_for_sampling_without_replacement
from ..llm import modeling_llama_no_layernorm as modeling_llama


class SSM_Sequoia(nn.Module):
    def __init__(self, config):
        super().__init__()

        model = modeling_llama.LlamaModel(config)
        if hasattr(model, "embed_tokens"):
            del model.embed_tokens

        self.fc = nn.Linear(config.hidden_size*2, config.hidden_size, bias=True)
        self.model = model

        self.max_candidate_tokens = 64 #! Currently not used
        self.depth = 10
        self.topk_len = 15
        
        self.UNIQUE_ID = 1
    
    # calling .config is same as calling model.config
    @property
    def config(self):
        return self.model.config
    
    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path,
        *model_args,
        config,
        torch_dtype=torch.float32,
        **model_kwargs
    ):
        draft_model_path = os.path.join(
            pretrained_model_name_or_path, "model.safetensors")
        
        model = cls(config, *model_args, **model_kwargs)
        load_model(model, draft_model_path, strict=True)
        model.to(dtype=torch_dtype)
        return model

    # TODO: embed_tokens is likely to be on a different device in multi-gpu scenario.
    # TODO: Think of an efficient way to handle embed_tokens.
    def forward(self, hidden_states, input_ids, embed_tokens, **kwargs):
        with torch.no_grad():
            inputs_embeds = embed_tokens(input_ids)
            
        inputs_embeds = inputs_embeds.to(hidden_states.dtype)
        hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))
        
        return self.model(inputs_embeds=hidden_states, **kwargs)
    
    @torch.no_grad()
    def _sample_probs(
        self,
        logits: torch.FloatTensor,
        logits_warper,
        do_sample: bool,
        T=1.0,
    ):
        if do_sample:
            batch, seq_len, vocab_size = logits.shape
            
            logits = logits.view(-1, vocab_size)
            next_token_scores = logits_warper(None, logits)
            probs = torch.softmax(next_token_scores/T, dim=-1)
            return probs.view(batch, seq_len, vocab_size) # preserve shape
        
        else:
            return torch.softmax(logits/T, dim=-1)
    
    @torch.no_grad()
    def _update_tree_attention_data(self, depth, nodes, hidden_states, tree_mask, position_offset):
        device = hidden_states.device 
        indices = torch.tensor([node.ind for node in nodes])

        input_hidden = hidden_states[:, indices].to(device)
        
        input_ids = torch.tensor([node.id for node in nodes], device=device)[None]
        
        position_ids = torch.zeros(len(nodes), device=device)[None] + (position_offset + depth)
        
        # Generating tree masks for the new nodes, don't have to consider the old nodes
        tree_mask = tree_mask[:, :, indices]
        tree_mask = torch.concat((tree_mask, torch.eye(len(nodes), device=device, dtype=torch.bool)[None, None]), dim=3)

        return input_hidden, input_ids, position_ids, tree_mask

    @torch.no_grad()
    def speculate(self, hidden_states, input_ids, embed_tokens, lm_head, logits_warper, do_sample, past_key_values, eos_token_id=None):
        device = hidden_states.device
        
        # take out last token
        input_ids = input_ids.to(device)
        sample_token = input_ids[:, -1:]
        input_ids = input_ids[:, 1:]
        
        # get original length of input_ids
        org_input_len = input_ids.shape[1] # offset of positon_id
        
        # initialize tree and tree_mask
        tree_mask = torch.ones([1, 1, 1, org_input_len], device=device, dtype=torch.bool)
        root = Node(str(sample_token[0][0].item()), id=sample_token[0][0].item(), prob=1, global_prob=1, ind=-1)
        depth = 1 # depth starts from 1 in tree library
        prev_sample_nodes = [root]
        
        while depth < self.depth:
            if depth == 1: # first iteration
                kv_len = past_key_values.get_seq_length()
                outputs = self(
                    hidden_states, input_ids[:, kv_len:],
                    embed_tokens=embed_tokens, 
                    past_key_values=past_key_values
                )
                hidden_states = outputs.last_hidden_state[:, -1:].clone() # Only the last token's hidden state is needed.
            else:
                hidden_states, input_ids, position_ids, tree_mask = self._update_tree_attention_data(depth, next_nodes, hidden_states, tree_mask, org_input_len)
                outputs = self(
                    hidden_states, input_ids,
                    embed_tokens=embed_tokens, 
                    past_key_values=past_key_values,
                    position_ids=position_ids, 
                    attention_mask=invert_mask(tree_mask, dtype=hidden_states.dtype)
                )
                hidden_states = outputs.last_hidden_state

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

            #* Get probabilities of each token
            # sample tokens with multinomial sampling
            # sampled_probs = self._sample_probs(lm_head(hidden_states), logits_warper, do_sample=do_sample).squeeze(0)
            # sampled_probs = self._sample_probs(lm_head(hidden_states), logits_warper, do_sample=do_sample, T=T).squeeze(0)
            # sampled_probs = torch.softmax(lm_head(hidden_states)[0]/T, dim=-1)
            # sampled_indices = torch.multinomial(sampled_probs, self.topk_len)
            # sampled_probs = torch.gather(sampled_probs, 1, sampled_indices)
            
            T = 1
            logits = lm_head(hidden_states)[0]
            rand = torch.rand(logits.shape, device=logits.device)
            all_probs, sampled_probs, sampled_indices = sampling_without_replacement(logits, rand=rand, num_samples=self.topk_len, temperature=T)
            
            # sampled_probs, sampled_indices = sampling_callable(logits, rand)
            
            #! Currently building up the tree is the most time-consuming part, need optimization.
            # TODO: Build a tree with following capabilities:
            # Tree can easily obtain all nodes at a certain depth
            # Tree can O(1) access the n'th node of any depth
            # Tree can easily prune nodes with lowest global_prob
            # Fast append nodes to tree, keeping their node_id, prob, global_prob, parent_ind
            # TBH if build tree and keep data all using pytorch tensors, it should be very fast.
            #* Create nodes
            next_nodes = []
            for prev_ind, prev_node in enumerate(prev_sample_nodes):
                for i in range(self.topk_len):
                    token_id = sampled_indices[prev_ind][i].item()
                    prob = sampled_probs[prev_ind][i].item()
                    global_prob = prob * prev_node.prob
                    
                    # if prob < 1e-3:# or global_prob < 1e-6:
                    #     continue
                    
                    new_node = Node(str(self.UNIQUE_ID), order=i, id=token_id, prob=prob, global_prob=global_prob, ind=prev_ind)
                    self.UNIQUE_ID += 1 # increment node id, make sure it is unique
                    next_nodes.append(new_node)
                    
                #! new: store the child_probs in the parent node
                prev_node.sample_probs = all_probs[prev_ind]
            
            #* depth increment
            depth += 1
            
            #* Some tree pruning logic
            #! Finding top-k actually breaks probability calculation for sequoia. Treedy does not have this issue.
            # next_nodes = sorted(next_nodes, key=lambda x: x.global_prob, reverse=True)[:self.topk_len]
            # get top_k_len nodes with highest global_prob
            node_probs = torch.tensor([node.global_prob for node in next_nodes], dtype=torch.float16)
            topk_indices = torch.topk(node_probs, self.topk_len).indices
            next_nodes = [next_nodes[i] for i in topk_indices]
            
            #* Append nodes to their parent nodes
            for node in next_nodes:
                prev_sample_nodes[node.ind].append(node)
 
            #* Get the nodes as input for next iteration
            next_nodes = [node for node in next_nodes if node.id != eos_token_id] # to follow prev_nodes logic above
            prev_sample_nodes = next_nodes
            
            #* Early stop if no nodes for next iteration
            # TODO: Also break if total_global_prob < threshold, where it does not benefit to continue
            if len(next_nodes) == 0:
                break
        
        #* Keep only the top_k_len nodes with the highest global_probs (pruning is so slow using this library, need optimization)
        # remove_nodes = sorted(preorder_iter(root), key=lambda x: x.global_prob)[:-self.max_candidate_tokens]
        # shift_nodes(root, [node.path_name for node in remove_nodes], [None]*len(remove_nodes), delete_children=True, skippable=True)
        
        #* Crop the tree to the max_candidate_tokens
        past_key_values.crop(org_input_len)
        
        return root