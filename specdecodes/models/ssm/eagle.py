import torch
import torch.nn as nn
from safetensors.torch import load_model
import os

from bigtree import Node, find_attrs,  shift_nodes, yield_tree
from bigtree import preorder_iter, levelorder_iter

from ..utils import invert_mask
from .modeling_llama_no_init_weights import LlamaModel


# TODO: Rename this to EagleXXX after implementing other models
class DraftModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        model = LlamaModel(config)
        if hasattr(model, "embed_tokens"):
            del model.embed_tokens

        self.fc = nn.Linear(config.hidden_size*2, config.hidden_size, bias=True)
        self.model = model

        self.max_candidate_tokens = 60
        self.depth = 3#5
        self.topk_len = 10
    
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
    def _update_tree_attention_data(self, depth, nodes, hidden_states, tree_mask, position_offset):
        device = hidden_states.device 
        parent_indices = torch.tensor([node.parent_ind for node in nodes])

        input_hidden = hidden_states[:, parent_indices].to(device)
        
        input_ids = torch.tensor([node.id for node in nodes])[None].to(device)
        
        position_ids = torch.zeros(len(nodes), device=device)[None] + (position_offset + depth)
        
        # Generating tree masks for the new nodes, don't have to consider the old nodes
        if parent_indices[0] != -1: # if not root
            tree_mask = tree_mask[:, :, parent_indices]
        tree_mask = torch.concat((tree_mask, torch.eye(len(nodes), device=device, dtype=torch.bool)[None, None]), dim=3)

        return input_hidden, input_ids, position_ids, tree_mask

    @torch.no_grad()
    def speculate(self, hidden_states, input_ids, embed_tokens, lm_head, past_key_values, eos_token_id=None):
        device = hidden_states.device
        
        # take out last token
        input_ids = input_ids.to(device)
        sample_token = input_ids[:, -1:]
        input_ids = input_ids[:, 1:]
        
        # get original length of input_ids and kv_len
        org_input_len = input_ids.shape[1] # offset of positon_id
        kv_len = past_key_values.get_seq_length()
        
        # initialize tree and tree_mask
        tree_mask = torch.ones([1, 1, 1, org_input_len], device=device, dtype=torch.bool)
        root = Node(str(sample_token[0][0].item()), id=sample_token[0][0].item(), prob=1, global_prob=1, parent_ind=-1)
        depth = 1 # depth starts from 1 in tree library
        
        while depth < self.depth:
            if depth == 1: # first iteration
                outputs = self(
                    hidden_states, input_ids[:, kv_len:], 
                    embed_tokens=embed_tokens, 
                    past_key_values=past_key_values
                )
                out_hidden = outputs.last_hidden_state[:, -1:].clone() # Only the last token's hidden state is needed.
            else:
                hidden_states, input_ids, position_ids, tree_mask = self._update_tree_attention_data(depth, next_nodes, hidden_states, tree_mask, org_input_len)
                outputs = self(
                    hidden_states, input_ids, 
                    embed_tokens=embed_tokens, 
                    past_key_values=past_key_values, 
                    position_ids=position_ids, 
                    attention_mask=invert_mask(tree_mask, dtype=hidden_states.dtype)
                )
                out_hidden = outputs.last_hidden_state
                #TODO: keep used parent indices, don't crop everything
                # newly generated kv cache not needed
                past_key_values.crop(org_input_len) 

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

            #* Get probabilities of each token
            sampled_probs = nn.functional.softmax(lm_head(out_hidden)[0], dim=-1) # [0] removes batch dimension
            
            # sample top_k tokens, and their probabilities
            topk_tokens = torch.topk(sampled_probs, self.topk_len, dim=-1)
            topk_index, topk_prob = topk_tokens.indices, topk_tokens.values

            #! Currently building up the tree is the most time-consuming part, need optimization.
            # TODO: Build a tree with following capabilities:
            # Tree can easily obtain all nodes at a certain depth
            # Tree can O(1) access the n'th node of any depth
            # Tree can easily prune nodes with lowest global_prob
            # Fast append nodes to tree, keeping their node_id, prob, global_prob, parent_ind
            # TBH if build tree and keep data all using pytorch tensors, it should be very fast.
            #* Append nodes ready for next iteration
            prev_nodes = list(find_attrs(root, "depth", depth))
            prev_nodes = [node for node in prev_nodes if node.prob > 1e-2 and node.id != eos_token_id]
            for idx, node in enumerate(prev_nodes):
                for i in range(self.topk_len):
                    token_id = topk_index[idx][i].item()
                    prob = topk_prob[idx][i].item()
                    global_prob = prob * node.prob
                    if global_prob > 1e-2:
                        new_node = Node(str(token_id), id=token_id, prob=prob, global_prob=global_prob, parent_ind=idx)
                        node.append(new_node)

            #* depth increment
            depth += 1
            
            #* Some tree pruning logic
            # keep the top_k_len nodes with the highest global_probs
            # added_nodes = list(find_attrs(root, "depth", depth))
            # remove_nodes = sorted(added_nodes, key=lambda x: x.global_prob)[:-self.topk_len]
            # shift_nodes(root, [node.path_name for node in remove_nodes], [None]*len(remove_nodes))
 
            #* Get the nodes as input for next iteration
            next_nodes = list(find_attrs(root, "depth", depth))
            next_nodes = [node for node in next_nodes if node.prob > 1e-2 and node.id != eos_token_id] # to follow prev_nodes logic above

            #* Early stop if no nodes for next iteration
            # TODO: Also break if total_global_prob < threshold, where it does not benefit to continue
            if len(next_nodes) == 0:
                break
        
        #* Keep only the top_k_len nodes with the highest global_probs
        remove_nodes = sorted(preorder_iter(root), key=lambda x: x.global_prob)[:-self.max_candidate_tokens]
        shift_nodes(root, [node.path_name for node in remove_nodes], [None]*len(remove_nodes))#, delete_children=True, skippable=True)

        return root