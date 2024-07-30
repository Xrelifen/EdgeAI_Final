import torch
import torch.nn as nn

from utils import invert_mask

from bigtree import Node, find_attrs,  shift_nodes, yield_tree
from bigtree import preorder_iter, levelorder_iter

# TODO: Fix the model to use the new tree structure
class DraftModel(nn.Module):
    def __init__(self, config, model = None):
        super().__init__()
        if hasattr(model, "embed_tokens"):
            del model.embed_tokens

        self.fc = nn.Linear(config.hidden_size*2, config.hidden_size, bias=True)
        self.model = model
        self.lm_head = None
        self.embed_tokens = None

        self.max_candidate_tokens = 60
        self.depth = 3#5
        self.topk_len = 10
        self.threshold = 1.0
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)

    def set_head_and_embed(self, lm_head, embed_tokens):
        self.lm_head = lm_head
        self.embed_tokens = embed_tokens
        for param in self.lm_head.parameters():
            param.requires_grad = False
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    def discard_head_and_embed(self):
        lm_head = self.lm_head
        embed_tokens = self.embed_tokens
        self.lm_head = None
        self.embed_tokens = None

        return lm_head, embed_tokens

    def forward(self, hidden_states, input_ids, embed_tokens=None, **kwargs):
        if embed_tokens is None:
            if self.embed_tokens is None:
                raise ValueError("embed_tokens is not provided")
            embed_tokens = self.embed_tokens

        with torch.no_grad():
            inputs_embeds = embed_tokens(input_ids)  # [Multiple GPU Support]

        inputs_embeds = inputs_embeds.to(hidden_states.dtype)
        hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))
        # returns hidden_states only
        return self.model(inputs_embeds=hidden_states, **kwargs)
    
    @torch.no_grad()
    def _update_tree_attention_data(self, nodes, hidden_states, tree_mask, depth, org_input_len, device):
        parent_indices = torch.tensor([node.parent_ind for node in nodes])
        
        input_hidden = hidden_states[:, parent_indices].to(device)
        input_ids = torch.tensor([node.id for node in nodes])[None].to(device)
        position_ids = torch.zeros(len(nodes), device=device) + (org_input_len + depth)
        
        # Generating tree masks for the new nodes, don't have to consider the old nodes
        if depth > 1:
            tree_mask = tree_mask[:, :, parent_indices]
        tree_mask = torch.concat((tree_mask, torch.eye(len(nodes), device=device, dtype=torch.bool)[None, None]), dim=3)
        tree_mask = invert_mask(tree_mask, dtype=hidden_states.dtype)

        return input_hidden, input_ids, tree_mask, position_ids

    @torch.no_grad()
    def speculate(self, hidden_states, input_ids):
        # take out last token
        kv_len = self.past_key_values.get_seq_length()
        sample_token = input_ids[:, None, -1].to(hidden_states.device)
        input_ids = input_ids[:, 1:].to(hidden_states.device)
        org_input_len = input_ids.shape[1] # offset of positon_id
        
        # generate tree
        root = Node(str(sample_token[0][0].item()), id=sample_token[0][0].item(), prob=1, global_prob=1, parent_ind=-1)
        depth = 0 + 1 # depth starts from 1 in tree library
        tree_mask = torch.ones([1, 1, 1, org_input_len], device=hidden_states.device, dtype=torch.bool)
        
        while depth < self.depth:
            if depth == 0 + 1: # first iteration
                outputs = self(hidden_states, input_ids[:, kv_len:], past_key_values=self.past_key_values)
                out_hidden = outputs.last_hidden_state[:, None, -1] # Only the last token's hidden state is needed.
                self.past_key_values = outputs.past_key_values
            else:
                hidden_states, input_ids, tree_mask, position_ids = self._update_tree_attention_data(next_nodes, hidden_states, tree_mask, depth, org_input_len, hidden_states.device)
                
                outputs = self(hidden_states, input_ids, past_key_values=self.past_key_values, position_ids=position_ids[None], attention_mask=tree_mask)
                out_hidden = outputs.last_hidden_state.clone()
                self.past_key_values.crop(org_input_len) # newly generated kv cache not needed

            #* Get probabilities of each token
            # out_hidden[0] removes batch dimension
            # maybe can use logsoftmax if needed to build faster trees.
            sampled_probs = self.softmax(self.lm_head(out_hidden[0])) 
            
            # sample top_k tokens, and their probabilities
            topk_tokens = torch.topk(sampled_probs, self.topk_len, dim=-1)
            topk_index, topk_prob = topk_tokens.indices, topk_tokens.values

            #! Currently building up the tree is the most time-consuming part, need optimization.
            # TODO: Build a tree with following capabilities:
            # Tree can easily obtain all nodes at a certain depth
            # Tree can O(1) access the n'th node of any depth
            # Fast append nodes to tree, keeping their node_id, prob, global_prob, parent_ind
            # Fast pruning, remove nodes with lowest global_prob in O(nlogn) time
            # TBH if build tree and keep data all using pytorch tensors, it should be very fast.
            #* Append nodes ready for next iteration
            prev_nodes = list(find_attrs(root, "depth", depth))
            prev_nodes = [node for node in prev_nodes if node.prob > 1e-2] # to follow next_nodes logic below
            for idx, node in enumerate(prev_nodes):
                for i in range(self.topk_len):
                    token_id = topk_index[idx][i].item()
                    prob = topk_prob[idx][i].item()
                    global_prob = prob * node.prob
                    if global_prob > 1e-3:
                        new_node = Node(str(token_id), id=token_id, prob=prob, global_prob=global_prob, parent_ind=idx)
                        node.append(new_node)

            #* depth increment
            depth += 1
            
            #* Some tree pruning logic
            # keep the top_k_len nodes with the highest global_probs
            added_nodes = find_attrs(root, "depth", depth+1)
            remove_nodes = sorted(added_nodes, key=lambda x: x.global_prob)[:-self.topk_len]
            shift_nodes(root, [node.path_name for node in remove_nodes], [None]*len(remove_nodes))
 
            #* Get final nodes ready for next iteration
            # get nodes ready for next iteration
            next_nodes = list(find_attrs(root, "depth", depth))
            next_nodes = [node for node in next_nodes if node.prob > 1e-2]

            #* Some stop logic
            if len(next_nodes) == 0:
                break
        
        # prune out (total_tokens) nodes with the lowest global_probs
        remove_nodes = sorted(preorder_iter(root), key=lambda x: x.global_prob)[:-self.max_candidate_tokens]
        shift_nodes(root, [node.path_name for node in remove_nodes], [None]*len(remove_nodes))#, delete_children=True, skippable=True)

        return root