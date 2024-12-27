import torch
from typing import Tuple, Dict, List, Optional

import threading
import queue

class TreeNode:
    """
    Simple node structure for a CPU-based tree.
    
    Attributes:
        parent (Optional[int]): Index of the parent node, or None if this is the root.
        children (List[int]): List of child node indices.
        depth (int): Depth in the tree (0 for root).
        token_id (int): The token ID.
        cumulative_probability (float): The cumulative probability up to this node.
        extra_data (int): Any extra integer data associated with the node.
        has_been_sampled (bool): Whether this node has already been sampled.
    """
    __slots__ = (
        'parent',
        'children',
        'depth',
        'token_id',
        'cumulative_probability',
        'extra_data',
        'has_been_sampled'
    )
    
    def __init__(
        self,
        parent: Optional[int],
        token_id: int,
        cumulative_probability: float,
        depth: int,
        extra_data: int = -1
    ):
        self.parent = parent
        self.children: List[int] = []
        self.depth = depth
        self.token_id = token_id
        self.cumulative_probability = cumulative_probability
        self.extra_data = extra_data
        self.has_been_sampled = False


class Tree:
    """
    CPU-based tree structure with linked (parent-child) nodes.

    Provides methods to add new nodes, prune the tree, retrieve data,
    and create an attention mask based on ancestor relationships.
    """
    
    __slots__ = (
        'prob_dtype',
        'nodes',
        'current_size',
        'available_leaves',
    )

    def __init__(
        self, 
        root_token_id: torch.Tensor,
        prob_dtype: torch.dtype = torch.float32,
    ):
        self.prob_dtype = prob_dtype
        self.nodes: List[TreeNode] = []

        # Create root node
        root_token_id = root_token_id.item()
        root = TreeNode(
            parent=None,
            token_id=root_token_id,
            cumulative_probability=1.0,
            depth=0,
            extra_data=-1
        )
        self.nodes.append(root)
        
        self.current_size = 1
        self.available_leaves: List[int] = [0]

    def add_nodes(
        self, 
        token_ids: torch.Tensor, 
        probabilities: torch.Tensor, 
        local_parent_indices: torch.Tensor,
        # valid_flags: torch.Tensor
    ):
        """
        Add new child nodes (one per valid position) to the tree.
        """
        # # Indices of valid new positions
        # valid_indices = valid_flags.nonzero(as_tuple=False).squeeze(-1)
        # num_new_nodes = valid_indices.numel()
        num_new_nodes = token_ids.size(0)
        # if num_new_nodes == 0:
        #     # No valid expansions; mark current leaves as sampled
        #     for leaf_idx in self.available_leaves:
        #         self.nodes[leaf_idx].has_been_sampled = True
        #     self.available_leaves = []
        #     return

        # Mark existing leaves as sampled
        for leaf_idx in self.available_leaves:
            self.nodes[leaf_idx].has_been_sampled = True

        # Gather subset
        # valid_parents = local_parent_indices[valid_indices].tolist()
        # valid_tokens = token_ids[valid_indices].tolist()
        # valid_probs = probabilities[valid_indices].tolist()
        valid_parents = local_parent_indices.tolist()
        valid_tokens = token_ids.tolist()
        valid_probs = probabilities.tolist()

        # Create new nodes
        old_size = self.current_size
        new_nodes = []
        new_leaves = []
        for i, (loc_par, t_id, prob) in enumerate(zip(valid_parents, valid_tokens, valid_probs)):
            parent_idx = self.available_leaves[loc_par]
            parent_node = self.nodes[parent_idx]

            node = TreeNode(
                parent=parent_idx,
                token_id=t_id,
                cumulative_probability=prob,
                depth=parent_node.depth + 1,
                extra_data=-1
            )
            new_nodes.append(node)
            parent_node.children.append(old_size + i)
            new_leaves.append(old_size + i)

        # Extend
        self.nodes.extend(new_nodes)
        self.current_size += num_new_nodes
        self.available_leaves = new_leaves

    def prune_to_top_n(self, n: int) -> torch.Tensor:
        """
        Keep only top-n nodes (by cumulative_probability) + all their ancestors.
        Remove others. Return 1D CPU tensor of old indices that were kept.
        """
        if n == -1 or self.current_size <= n:
            # No pruning needed
            return torch.arange(self.current_size, device='cpu')

        # Gather probabilities in one tensor
        probs = torch.tensor(
            [nd.cumulative_probability for nd in self.nodes],
            dtype=self.prob_dtype
        )

        # top-k
        # NOTE: If you expect many thousands of nodes, topk is O(n log k), 
        # which is typically faster than a full sort O(n log n).
        keep_vals, keep_idx = torch.topk(probs, k=n, sorted=False)

        # Convert to Python set
        top_indices = set(keep_idx.tolist())

        # BFS/DFS up the tree to get ancestors
        ancestors_set = set(top_indices)
        stack = list(top_indices)
        while stack:
            cur = stack.pop()
            parent = self.nodes[cur].parent
            if parent is not None and parent not in ancestors_set:
                ancestors_set.add(parent)
                stack.append(parent)

        # Build new index mapping
        nodes_to_keep = sorted(ancestors_set)
        old_to_new = {old_idx: i for i, old_idx in enumerate(nodes_to_keep)}

        # Build new node list
        new_nodes = []
        for old_idx in nodes_to_keep:
            old_node = self.nodes[old_idx]
            parent_new = old_to_new[old_node.parent] if (old_node.parent is not None) and (old_node.parent in old_to_new) else None

            new_node = TreeNode(
                parent=parent_new,
                token_id=old_node.token_id,
                cumulative_probability=old_node.cumulative_probability,
                depth=old_node.depth,
                extra_data=old_node.extra_data
            )
            new_node.has_been_sampled = old_node.has_been_sampled
            new_nodes.append(new_node)

        # Re-link children
        for new_idx, old_idx in enumerate(nodes_to_keep):
            old_children = self.nodes[old_idx].children
            for c in old_children:
                if c in old_to_new:
                    new_nodes[new_idx].children.append(old_to_new[c])

        # Overwrite
        self.nodes = new_nodes
        self.current_size = len(new_nodes)

        # Recompute available leaves
        is_parent = [False]*self.current_size
        for i, node in enumerate(self.nodes):
            if node.children:
                is_parent[i] = True

        # Leaves = not parent + not sampled
        self.available_leaves = [
            i for i, node in enumerate(self.nodes)
            if (not is_parent[i]) and (not node.has_been_sampled)
        ]

        return torch.tensor(nodes_to_keep, dtype=torch.long)

    def get_available_leaf_node_data(self) -> Dict[str, torch.Tensor]:
        """
        Retrieve data for current available leaves.
        """
        leaf_data = {
            'available_leaf_indices': [],
            'token_ids': [],
            'cumulative_probabilities': [],
            'depths': [],
            'parent_indices': [],
            'extra_data': [],
        }
        leaves = self.available_leaves
        for leaf_idx in leaves:
            nd = self.nodes[leaf_idx]
            leaf_data['available_leaf_indices'].append(leaf_idx)
            leaf_data['token_ids'].append(nd.token_id)
            leaf_data['cumulative_probabilities'].append(nd.cumulative_probability)
            leaf_data['depths'].append(nd.depth)
            leaf_data['parent_indices'].append(nd.parent if nd.parent is not None else -1)
            leaf_data['extra_data'].append(nd.extra_data)

        return {
            k: torch.tensor(
                v,
                dtype=(self.prob_dtype if 'probabilities' in k else torch.long),
                device='cpu'
            )
            for k, v in leaf_data.items()
        }

    def get_node_data(self) -> Dict[str, torch.Tensor]:
        """
        Retrieve data for all nodes currently in the tree.
        """
        token_ids = []
        cum_probs = []
        depths = []
        parents = []
        extra_data = []
        sampled = []

        for nd in self.nodes:
            token_ids.append(nd.token_id)
            cum_probs.append(nd.cumulative_probability)
            depths.append(nd.depth)
            parents.append(nd.parent if nd.parent is not None else -1)
            extra_data.append(nd.extra_data)
            sampled.append(nd.has_been_sampled)

        return {
            'token_ids': torch.tensor(token_ids, dtype=torch.long),
            'cumulative_probabilities': torch.tensor(cum_probs, dtype=self.prob_dtype),
            'depths': torch.tensor(depths, dtype=torch.long),
            'parent_indices': torch.tensor(parents, dtype=torch.long),
            'extra_data': torch.tensor(extra_data, dtype=torch.long),
            'has_been_sampled': torch.tensor(sampled, dtype=torch.bool)
        }

    def set_node_extra_data(self, node_indices: torch.Tensor, data: torch.Tensor):
        """
        Assign extra integer data to specified nodes.
        """
        node_indices = node_indices.tolist()
        data_list = data.tolist()
        for idx, d in zip(node_indices, data_list):
            self.nodes[idx].extra_data = d

    def get_node_extra_data(self, node_indices: torch.Tensor) -> torch.Tensor:
        """
        Retrieve extra integer data from specified nodes.
        """
        node_indices = node_indices.tolist()
        result = [self.nodes[idx].extra_data for idx in node_indices]
        return torch.tensor(result, dtype=torch.long)

    def get_children_indices(self, node_index: int) -> torch.Tensor:
        """
        Retrieve all direct children of a given node index.
        
        Args:
            node_index (int): The index of the node in `self.nodes`.
        
        Returns:
            torch.Tensor: 1D CPU tensor of child indices.
        """
        return torch.tensor(
            self.nodes[node_index].children,
            dtype=torch.long,
            device='cpu'
        )
    
    def get_available_leaf_node_data(self) -> Dict[str, torch.Tensor]:
        """
        Retrieve data for all current available leaves (unsampled).
        
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing tensors of:
                'available_leaf_indices', 'token_ids', 
                'cumulative_probabilities', 'depths', 
                'parent_indices', and 'extra_data'.
        """
        leaf_data = {
            'available_leaf_indices': [],
            'token_ids': [],
            'cumulative_probabilities': [],
            'depths': [],
            'parent_indices': [],
            'extra_data': []
        }
        
        for leaf_idx in self.available_leaves:
            node = self.nodes[leaf_idx]
            leaf_data['available_leaf_indices'].append(leaf_idx)
            leaf_data['token_ids'].append(node.token_id)
            leaf_data['cumulative_probabilities'].append(node.cumulative_probability)
            leaf_data['depths'].append(node.depth)
            leaf_data['parent_indices'].append(node.parent) # if node.parent is not None else -1)
            leaf_data['extra_data'].append(node.extra_data)
        
        # Convert lists to tensors
        return {
            key: torch.tensor(
                val,
                dtype=(self.prob_dtype if 'probabilities' in key else torch.long),
                device='cpu'
            )
            for key, val in leaf_data.items()
        }

    def get_node_data(self) -> Dict[str, torch.Tensor]:
        """
        Retrieve data for all nodes currently in the tree.
        
        Returns:
            Dict[str, torch.Tensor]: A dictionary with keys:
                'token_ids', 'cumulative_probabilities', 'depths',
                'parent_indices', 'extra_data', 'has_been_sampled'.
        """
        token_ids = []
        cum_probs = []
        depths = []
        parents = []
        extra_data = []
        sampled = []
        
        for node in self.nodes:
            token_ids.append(node.token_id)
            cum_probs.append(node.cumulative_probability)
            depths.append(node.depth)
            parents.append(node.parent if node.parent is not None else -1)
            extra_data.append(node.extra_data)
            sampled.append(node.has_been_sampled)
        
        return {
            'token_ids': torch.tensor(token_ids, dtype=torch.long, device='cpu'),
            'cumulative_probabilities': torch.tensor(cum_probs, dtype=self.prob_dtype, device='cpu'),
            'depths': torch.tensor(depths, dtype=torch.long, device='cpu'),
            'parent_indices': torch.tensor(parents, dtype=torch.long, device='cpu'),
            'extra_data': torch.tensor(extra_data, dtype=torch.long, device='cpu'),
            'has_been_sampled': torch.tensor(sampled, dtype=torch.bool, device='cpu')
        }
    
    def get_node_indices(self) -> torch.Tensor:
        """
        Return a CPU tensor of all valid node indices (0 to current_size - 1).
        """
        return torch.arange(self.current_size, device='cpu')
    
    def get_available_leaf_position_ids(self) -> torch.Tensor:
        """
        Return a 1D CPU tensor of position IDs for all available leaves.
        
        Args:
            offset (int): An optional offset to add to each depth value.
        
        Returns:
            torch.Tensor: 1D tensor of position IDs.
        """
        return torch.tensor(
            [node.depth for node in self.nodes if not node.has_been_sampled],
            dtype=torch.long,
            device='cpu'
        )
        
    def get_max_depth(self) -> torch.Tensor:
        """
        Return the maximum depth of the tree as a 1D CPU tensor.
        """
        if self.nodes:
            max_d = max(node.depth for node in self.nodes)
        else:
            max_d = 0
        return torch.tensor(max_d, dtype=torch.long, device='cpu')

    def create_attention_mask(self, prefix_length: int = 0, device: str = 'cpu') -> torch.Tensor:
        """
        Create an attention mask indicating which nodes can attend to which.
        Specifically, each node can attend to itself and all of its ancestors.

        If prefix_length > 0, an additional prefix dimension is prepended,
        and these positions are fully attendable (mask = True).

        The final mask uses large negative values (float('-inf')) in places
        where attention is blocked, so that adding this mask to logits
        effectively prevents attending to those positions.

        Returns:
            torch.Tensor: A mask tensor of shape [1, 1, num_nodes, prefix_length + num_nodes].
                        False (→ large negative) means blocked; True (→ 0) means attendable.
        """
        num_nodes = self.current_size

        # Edge case: If the tree is empty, return an empty tensor on the correct device
        if num_nodes == 0:
            return torch.empty((1, 1, 0, prefix_length), dtype=self.prob_dtype, device=device)
        
        # Step 1: Build an ancestor matrix on CPU as a list of lists.
        #         Each row i will mark the ancestors of node i (including i itself).
        ancestor_matrix = [[False] * num_nodes for _ in range(num_nodes)]
        for i in range(num_nodes):
            # A node always attends to itself
            ancestor_matrix[i][i] = True
            
            # Walk up the parent chain, marking ancestors
            parent_idx = self.nodes[i].parent
            while parent_idx is not None:
                ancestor_matrix[i][parent_idx] = True
                parent_idx = self.nodes[parent_idx].parent
        
        # Step 2: Convert the Python list-of-lists to a boolean tensor on the desired device
        ancestor_tensor = torch.tensor(ancestor_matrix, dtype=torch.bool, device=device)
        
        # Step 3: If prefix_length > 0, prepend a (num_nodes x prefix_length) block of True
        #         so that each node can also attend those prefix positions
        if prefix_length > 0:
            prefix_part = torch.ones((num_nodes, prefix_length), dtype=torch.bool, device=device)
            ancestor_tensor = torch.cat([prefix_part, ancestor_tensor], dim=1)
        
        # Step 4: Convert the boolean attend-mask to a negative-infinity mask
        #         (False => large negative; True => 0)
        inverted_mask = (~ancestor_tensor).to(self.prob_dtype)
        inverted_mask *= torch.finfo(self.prob_dtype).min
        
        # Final shape: [batch=1, heads=1, seq_len=num_nodes, hidden=(prefix_length + num_nodes)]
        return inverted_mask.unsqueeze(0).unsqueeze(0)

    def set_node_extra_data(self, node_indices: torch.Tensor, data: torch.Tensor):
        """
        Assign extra integer data to the specified nodes.
        
        Args:
            node_indices (torch.Tensor): A 1D tensor of node indices.
            data (torch.Tensor): A 1D tensor of data values to assign.
        """
        node_indices = node_indices.to('cpu').tolist()
        data_list = data.to('cpu').tolist()
        
        for idx, d in zip(node_indices, data_list):
            self.nodes[idx].extra_data = d

    def get_node_extra_data(self, node_indices: torch.Tensor) -> torch.Tensor:
        """
        Retrieve the extra integer data from the specified nodes.
        
        Args:
            node_indices (torch.Tensor): A 1D tensor of node indices.
        
        Returns:
            torch.Tensor: A 1D CPU tensor of extra_data values.
        """
        node_indices = node_indices.to('cpu').tolist()
        result = [self.nodes[idx].extra_data for idx in node_indices]
        return torch.tensor(result, dtype=torch.long, device='cpu')

    def print_tree_structure(self, show_token_id: bool = True, show_probability: bool = False):
        """
        Print a text-based representation of the tree.
        
        Args:
            show_token_id (bool): Whether to display the node's token_id.
            show_probability (bool): Whether to display the node's cumulative_probability.
        
        Raises:
            ValueError: If both show_token_id and show_probability are False.
        """
        if not (show_token_id or show_probability):
            raise ValueError("At least one of show_token_id or show_probability must be True.")
        
        # Precompute children for easy traversal
        children_list = [[] for _ in range(self.current_size)]
        for i, node in enumerate(self.nodes):
            for c in node.children:
                children_list[i].append(c)
        
        def recurse_print(node_idx: int, prefix=''):
            child_nodes = children_list[node_idx]
            for i, child_idx in enumerate(child_nodes):
                is_last = (i == len(child_nodes) - 1)
                connector = '└── ' if is_last else '├── '
                
                child_node = self.nodes[child_idx]
                pieces = []
                if show_token_id:
                    pieces.append(str(child_node.token_id))
                if show_probability:
                    pieces.append(f"{child_node.cumulative_probability:.4f}")
                node_repr = " ".join(pieces)
                
                print(prefix + connector + node_repr)
                extension = '    ' if is_last else '│   '
                recurse_print(child_idx, prefix + extension)
        
        # Print the root
        root_node = self.nodes[0]
        root_pieces = []
        if show_token_id:
            root_pieces.append(str(root_node.token_id))
        if show_probability:
            root_pieces.append(f"{root_node.cumulative_probability:.4f}")
        
        print(" ".join(root_pieces))
        recurse_print(0)

    def __repr__(self):
        return f"LinkedCPUTree(num_nodes={self.current_size}, device='cpu')"
    

class TreeBuilderWorker(threading.Thread):
    """
    A background worker that:
      1) Receives GPU-based expansions via a Queue.
      2) Converts them to CPU in this thread.
      3) Updates a CPU-based Tree with each expansion.
    Once a sentinel `None` is placed on the queue, it stops.
    """

    def __init__(
        self,
        root_id: int,
        tree_prob_dtype: torch.dtype,
        max_tokens: int
    ):
        """
        Args:
            root_id: Token ID for the root node (if needed).
            tree_prob_dtype: Dtype used in the CPU tree.
            max_tokens: Maximum tokens for pruning in the Tree.
        """
        super().__init__()
        self.tree = Tree(root_token_id=root_id, prob_dtype=tree_prob_dtype)
        self.expansion_queue = queue.Queue()
        self.max_tokens = max_tokens

    def run(self):
        """
        Continuously fetch expansions from the queue, blocking until one is available.
        If `None` is encountered, we exit immediately.
        """
        while True:
            # Block until an expansion is available
            expansion = self.expansion_queue.get()  # blocks indefinitely
            if expansion is None:
                break  # Sentinel => done

            # expansion is (token_ids_gpu, probs_gpu, parents_gpu, valid_flags_gpu).
            token_ids_cpu, probs_cpu, parents_cpu = [ #, valid_flags_cpu = [
                x.to('cpu', non_blocking=True) for x in expansion
            ]
            torch.cuda.synchronize()
            
            # Update the CPU-based tree
            self.tree.add_nodes(token_ids_cpu, probs_cpu, parents_cpu) #, valid_flags_cpu)
            # self.tree.prune_to_top_n(self.max_tokens)

    def close_queue(self):
        """
        Indicate no more expansions will arrive by sending a sentinel (None).
        """
        self.expansion_queue.put(None)

    def put_expansion(self, expansion):
        """
        Enqueue a single expansion tuple of GPU tensors:
          (token_ids_gpu, probs_gpu, parents_gpu, valid_flags_gpu)
        """
        self.expansion_queue.put(expansion)

    def get_tree(self):
        """
        Retrieve the final CPU-based Tree after the thread has joined.
        """
        return self.tree