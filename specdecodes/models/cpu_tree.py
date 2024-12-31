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
        has_been_sampled (bool): Whether this node has already been sampled.
    """
    __slots__ = (
        'parent',
        'children',
        'depth',
        'token_id',
        'cumulative_probability',
        'has_been_sampled'
    )
    
    def __init__(
        self,
        parent: Optional[int],
        token_id: int,
        cumulative_probability: float,
        depth: int,
    ):
        self.parent = parent
        self.children: List[int] = []
        self.depth = depth
        self.token_id = token_id
        self.cumulative_probability = cumulative_probability
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
        )
        self.nodes.append(root)
        
        self.current_size = 1
        self.available_leaves: List[int] = [0]

    def add_nodes(
        self, 
        token_ids: torch.Tensor, 
        token_probs: torch.Tensor, 
        local_parent_indices: torch.Tensor,
    ):
        """
        Add new child nodes (one per valid position) to the tree.
        """
        batch_size, num_new_nodes = token_ids.shape
        assert batch_size == 1, "Currently only batch_size=1 is supported."

        # Mark existing leaves as sampled
        for leaf_idx in self.available_leaves:
            self.nodes[leaf_idx].has_been_sampled = True

        # Gather subset, assuming batch_size=1
        parents = local_parent_indices[0].tolist()
        tokens = token_ids[0].tolist()
        probs = token_probs[0].tolist()

        # Create new nodes
        old_size = self.current_size
        new_nodes = []
        new_leaves = []
        for i, (loc_par, t_id, prob) in enumerate(zip(parents, tokens, probs)):
            parent_idx = self.available_leaves[loc_par]
            parent_node = self.nodes[parent_idx]

            node = TreeNode(
                parent=parent_idx,
                token_id=t_id,
                cumulative_probability=prob,
                depth=parent_node.depth + 1,
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

    def get_node_data(self) -> Dict[str, torch.Tensor]:
        """
        Retrieve data for all nodes currently in the tree.
        
        Returns:
            Dict[str, torch.Tensor]: A dictionary with keys:
                'token_ids', 'cumulative_probabilities', 
                'depths', 'parent_indices'
        """
        token_ids = []
        cum_probs = []
        depths = []
        parents = []
        
        for node in self.nodes:
            token_ids.append(node.token_id)
            cum_probs.append(node.cumulative_probability)
            depths.append(node.depth)
            parents.append(node.parent if node.parent is not None else -1)
        
        return {
            'token_ids': torch.tensor(token_ids, dtype=torch.long, device='cpu'),
            'cumulative_probabilities': torch.tensor(cum_probs, dtype=self.prob_dtype, device='cpu'),
            'depths': torch.tensor(depths, dtype=torch.long, device='cpu'),
            'parent_indices': torch.tensor(parents, dtype=torch.long, device='cpu'),
        }
        
    def get_max_depth(self) -> torch.Tensor:
        """
        Return the maximum depth of the tree as a 1D CPU tensor.
        """
        if self.nodes:
            max_d = max(node.depth for node in self.nodes)
        else:
            max_d = 0
        return torch.tensor(max_d, dtype=torch.long, device='cpu')
    
    def size(self) -> int:
        """
        Return the current number of nodes in the tree.
        """
        return self.current_size

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

    def print_tree_structure(self, show_token_id: bool = True, show_probability: bool = True):
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