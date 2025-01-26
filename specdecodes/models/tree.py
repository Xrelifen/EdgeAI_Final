import torch
from typing import Tuple, Dict

class Tree:
    def __init__(
        self, 
        root_token_id: int, 
        device: str = 'cuda', 
        prob_dtype: torch.dtype = torch.float32, 
        max_nodes: int = 1_000_000
    ):
        """
        Optimized tree structure for GPU operations with preallocated tensors.

        Args:
            root_token_id (int): Token ID of the root node.
            device (str): Device to store tensors ('cuda' or 'cpu').
            prob_dtype (torch.dtype): Data type for probabilities.
            max_nodes (int): Maximum number of nodes the tree can hold.
        """
        self.device = device
        self.prob_dtype = prob_dtype
        self.max_nodes = max_nodes
        self.current_size = 1  # Start with root node

        # Preallocate tensors
        self.parent_indices = torch.full((max_nodes,), -1, dtype=torch.long, device=device)
        self.depths = torch.zeros(max_nodes, dtype=torch.long, device=device)
        self.token_ids = torch.empty(max_nodes, dtype=torch.long, device=device)
        self.token_ids[0] = root_token_id
        self.cumulative_probabilities = torch.zeros(max_nodes, dtype=prob_dtype, device=device)
        self.cumulative_probabilities[0] = 1.0
        self.extra_data = torch.full((max_nodes,), -1, dtype=torch.long, device=device)
        self.has_been_sampled = torch.zeros(max_nodes, dtype=torch.bool, device=device)

        # Initialize available leaves with the root node
        self.available_leaves = torch.tensor([0], dtype=torch.long, device=device)

    # def add_nodes(
    #     self, 
    #     token_ids: torch.Tensor, 
    #     probabilities: torch.Tensor, 
    #     parent_indices: torch.Tensor,
    #     valid_flags: torch.Tensor
    # ):
    #     """
    #     Add new nodes as children based on sampled tokens and validity flags.

    #     Args:
    #         token_ids (torch.Tensor): Tensor of token IDs to add. Shape: [sample_k]
    #         probabilities (torch.Tensor): Tensor of corresponding probabilities. Shape: [sample_k]
    #         parent_indices (torch.Tensor): Tensor of parent leaf indices for each sampled token. Shape: [sample_k]
    #         valid_flags (torch.Tensor): Tensor of boolean flags indicating valid tokens. Shape: [sample_k]
    #     """
    #     # Filter only valid tokens based on valid_flags
    #     valid_tokens = valid_flags.nonzero(as_tuple=True)[0]  # Shape: [num_valid]

    #     num_new_nodes = valid_tokens.size(0)
    #     if num_new_nodes == 0:
    #         # Mark all current available_leaves as sampled
    #         self.has_been_sampled[self.available_leaves] = True
    #         self.available_leaves = torch.empty((0,), dtype=torch.long, device=self.device)
    #         return

    #     # Assign new node indices
    #     new_node_indices = torch.arange(
    #         self.current_size, 
    #         self.current_size + num_new_nodes, 
    #         device=self.device, 
    #         dtype=torch.long
    #     )

    #     # Update tensors with new nodes
    #     self.parent_indices[self.current_size:self.current_size + num_new_nodes] = parent_indices[valid_tokens]
    #     self.depths[self.current_size:self.current_size + num_new_nodes] = self.depths[parent_indices[valid_tokens]] + 1
    #     self.token_ids[self.current_size:self.current_size + num_new_nodes] = token_ids[valid_tokens]
    #     self.cumulative_probabilities[self.current_size:self.current_size + num_new_nodes] = probabilities[valid_tokens]
    #     self.extra_data[self.current_size:self.current_size + num_new_nodes] = -1  # Initialize extra_data

    #     # Mark all current available_leaves as sampled
    #     self.has_been_sampled[self.available_leaves] = True

    #     # Update available_leaves to the newly added nodes
    #     self.available_leaves = new_node_indices

    #     # Update current size
    #     self.current_size += num_new_nodes
        
    def add_nodes(
        self, 
        token_ids: torch.Tensor, 
        probabilities: torch.Tensor, 
        parent_indices: torch.Tensor,
        valid_flags: torch.Tensor
    ):
        """
        Add new nodes as children based on sampled tokens and validity flags.

        Args:
            token_ids (torch.Tensor): Tensor of token IDs to add. Shape: [sample_k]
            probabilities (torch.Tensor): Tensor of corresponding probabilities. Shape: [sample_k]
            parent_indices (torch.Tensor): Tensor of parent leaf indices for each sampled token. Shape: [sample_k]
            valid_flags (torch.Tensor): Tensor of boolean flags indicating valid tokens. Shape: [sample_k]
        """
        # Ensure all input tensors are contiguous and on the correct device
        valid_flags = valid_flags.contiguous()
        parent_indices = parent_indices.contiguous()
        token_ids = token_ids.contiguous()
        probabilities = probabilities.contiguous()

        # Calculate the number of new nodes
        num_new_nodes = valid_flags.sum().item()

        if num_new_nodes == 0:
            # Mark all current available_leaves as sampled
            self.has_been_sampled[self.available_leaves] = True
            self.available_leaves = torch.empty((0,), dtype=torch.long, device=self.device)
            return

        # Assign new node indices
        new_node_indices = torch.arange(
            self.current_size, 
            self.current_size + num_new_nodes, 
            device=self.device, 
            dtype=torch.long
        )

        # Efficiently gather valid parent indices, token IDs, and probabilities using boolean masking
        self.parent_indices[self.current_size:self.current_size + num_new_nodes] = parent_indices[valid_flags]
        self.depths[self.current_size:self.current_size + num_new_nodes] = self.depths[parent_indices[valid_flags]] + 1
        self.token_ids[self.current_size:self.current_size + num_new_nodes] = token_ids[valid_flags]
        self.cumulative_probabilities[self.current_size:self.current_size + num_new_nodes] = probabilities[valid_flags]
        self.extra_data[self.current_size:self.current_size + num_new_nodes] = -1  # Initialize extra_data

        # Mark all current available_leaves as sampled
        self.has_been_sampled[self.available_leaves] = True

        # Update available_leaves to the newly added nodes
        self.available_leaves = new_node_indices

        # Update current_size
        self.current_size += num_new_nodes
        
    def prune_to_top_n(self, n: int) -> torch.Tensor:
        """
        Retain the top `n` nodes by cumulative probability and prune the rest.

        Args:
            n (int): Number of top nodes to retain. If -1, retain all nodes.

        Returns:
            torch.Tensor: Indices of nodes retained after pruning.
        """
        if n == -1 or self.current_size <= n:
            return torch.arange(self.current_size, device=self.device)

        # Select top n nodes based on cumulative probability
        top_probs, top_indices = torch.topk(
            self.cumulative_probabilities[:self.current_size], n, dim=0, largest=True, sorted=True
        )
        top_indices = top_indices.detach()

        # Initialize a mask for nodes to keep
        keep_mask = torch.zeros(self.max_nodes, dtype=torch.bool, device=self.device)
        keep_mask[top_indices] = True

        # Initialize a tensor to hold nodes to process (parents)
        current_nodes = top_indices.clone()

        # Iteratively mark all ancestors
        while True:
            # Get parents of the current nodes
            parents = self.parent_indices[current_nodes]
            # Filter out -1 (root has no parent)
            valid_parents = parents[parents != -1]
            # Determine which parents are new (not already kept)
            new_parents = ~keep_mask[valid_parents]
            if new_parents.sum() == 0:
                break
            # Update the keep_mask with new parents
            keep_mask[valid_parents[new_parents]] = True
            # Update current_nodes to new parents for next iteration
            current_nodes = parents[new_parents]

        # Get all node indices to keep
        nodes_to_keep = keep_mask.nonzero(as_tuple=True)[0]

        # Create a mapping from old indices to new indices
        num_keep = nodes_to_keep.size(0)
        index_mapping = torch.full((self.current_size,), -1, dtype=torch.long, device=self.device)
        index_mapping[nodes_to_keep] = torch.arange(num_keep, device=self.device)

        # Update parent indices with new mapping
        new_parent_indices = index_mapping[self.parent_indices[nodes_to_keep]]
        new_parent_indices[self.parent_indices[nodes_to_keep] == -1] = -1

        # Update tensors with kept nodes
        self.parent_indices[:num_keep] = new_parent_indices
        self.depths[:num_keep] = self.depths[nodes_to_keep]
        self.token_ids[:num_keep] = self.token_ids[nodes_to_keep]
        self.cumulative_probabilities[:num_keep] = self.cumulative_probabilities[nodes_to_keep]
        self.extra_data[:num_keep] = self.extra_data[nodes_to_keep]
        self.has_been_sampled[:num_keep] = self.has_been_sampled[nodes_to_keep]

        # Recompute available_leaves
        # A node is a parent if any node has it as a parent
        is_parent = torch.zeros(num_keep, dtype=torch.bool, device=self.device)
        parent_indices = self.parent_indices[:num_keep]
        parent_valid = parent_indices != -1
        is_parent[parent_indices[parent_valid]] = True
        # Available leaves are nodes that are not parents and have not been sampled
        self.available_leaves = (~is_parent & ~self.has_been_sampled[:num_keep]).nonzero(as_tuple=True)[0]

        # Update current size
        self.current_size = num_keep

        return nodes_to_keep

    # def prune_to_top_n(self, n: int) -> torch.Tensor:
    #     """
    #     Retain the top `n` nodes by cumulative probability and prune the rest.

    #     Args:
    #         n (int): Number of top nodes to retain. If -1, retain all nodes.

    #     Returns:
    #         torch.Tensor: Indices of nodes retained after pruning.
    #     """
    #     if n == -1 or self.current_size <= n:
    #         return torch.arange(self.current_size, device=self.device)

    #     # Select top n nodes based on cumulative probability
    #     top_probs, top_indices = torch.topk(
    #         self.cumulative_probabilities[:self.current_size], n, dim=0, largest=True, sorted=True
    #     )
    #     nodes_to_keep = self.get_all_ancestor_indices(top_indices)

    #     # Create a mask and mapping for nodes to keep
    #     mask = torch.zeros(self.max_nodes, dtype=torch.bool, device=self.device)
    #     mask[nodes_to_keep] = True
    #     old_indices = mask[:self.current_size].nonzero(as_tuple=True)[0]
    #     new_size = old_indices.size(0)
    #     index_mapping = torch.full((self.current_size,), -1, dtype=torch.long, device=self.device)
    #     index_mapping[old_indices] = torch.arange(new_size, device=self.device)

    #     # Update parent indices with new mapping
    #     mapped_parent_indices = index_mapping[self.parent_indices[:self.current_size][old_indices]]
    #     mapped_parent_indices[self.parent_indices[:self.current_size][old_indices] == -1] = -1

    #     # Update tensors with kept nodes
    #     self.parent_indices[:new_size] = mapped_parent_indices
    #     self.depths[:new_size] = self.depths[old_indices]
    #     self.token_ids[:new_size] = self.token_ids[old_indices]
    #     self.cumulative_probabilities[:new_size] = self.cumulative_probabilities[old_indices]
    #     self.extra_data[:new_size] = self.extra_data[old_indices]
    #     self.has_been_sampled[:new_size] = self.has_been_sampled[old_indices]

    #     # Recompute available_leaves
    #     is_parent = torch.zeros(new_size, dtype=torch.bool, device=self.device)
    #     parent_mask = self.parent_indices[:new_size] != -1
    #     parent_indices = self.parent_indices[:new_size][parent_mask]
    #     is_parent[parent_indices] = True
    #     self.available_leaves = (~is_parent & ~self.has_been_sampled[:new_size]).nonzero(as_tuple=True)[0]

    #     # Update current size
    #     self.current_size = new_size

    #     return nodes_to_keep

    def get_all_ancestor_indices(self, node_indices: torch.Tensor) -> torch.Tensor:
        """
        Retrieve all ancestor indices (including the nodes themselves) for given nodes.

        Args:
            node_indices (torch.Tensor): Indices of target nodes. Shape: [n]

        Returns:
            torch.Tensor: Unique sorted ancestor indices.
        """
        all_nodes = node_indices.clone()
        current_nodes = node_indices.clone()

        while True:
            parents = self.parent_indices[current_nodes]
            valid_parents = parents[parents != -1]
            if valid_parents.numel() == 0:
                break
            all_nodes = torch.cat([all_nodes, valid_parents])
            current_nodes = valid_parents.unique()

        return all_nodes.unique()

    def get_children_indices(self, node_index: int) -> torch.Tensor:
        """
        Retrieve all children of a given node.

        Args:
            node_index (int): Parent node index.

        Returns:
            torch.Tensor: Indices of child nodes.
        """
        return torch.nonzero(self.parent_indices[:self.current_size] == node_index, as_tuple=True)[0]

    def get_available_leaf_node_data(self) -> Dict[str, torch.Tensor]:
        """
        Fetch data for all current available leaf nodes (nodes that haven't been sampled yet).

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing available leaf node data.
        """
        leaves = self.available_leaves
        return {
            'available_leaf_indices': leaves,
            'token_ids': self.token_ids[leaves],
            'cumulative_probabilities': self.cumulative_probabilities[leaves],
            'depths': self.depths[leaves],
            'parent_indices': self.parent_indices[leaves],
            'extra_data': self.extra_data[leaves]
        }

    def get_node_data(self) -> Dict[str, torch.Tensor]:
        """
        Retrieve data for all nodes in the tree.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing all node data.
        """
        return {
            'token_ids': self.token_ids[:self.current_size],
            'cumulative_probabilities': self.cumulative_probabilities[:self.current_size],
            'depths': self.depths[:self.current_size],
            'parent_indices': self.parent_indices[:self.current_size],
            'extra_data': self.extra_data[:self.current_size],
            'has_been_sampled': self.has_been_sampled[:self.current_size]
        }

    def get_node_indices(self) -> torch.Tensor:
        """
        Get indices of all nodes in the tree.

        Returns:
            torch.Tensor: Tensor of node indices.
        """
        return torch.arange(self.current_size, device=self.device)

    def get_max_depth(self) -> torch.Tensor:
        """
        Determine the maximum depth of the tree.

        Returns:
            torch.Tensor: Scalar tensor representing the maximum depth.
        """
        return self.depths[:self.current_size].max()

    def create_attention_mask(self, prefix_length: int = 0) -> torch.Tensor:
        """
        Create an attention mask allowing nodes to attend to themselves and their ancestors.

        Args:
            prefix_length (int, optional): Length of the prefix sequence to prepend. Defaults to 0.

        Returns:
            torch.Tensor: Attention mask of shape (1, 1, num_nodes, num_nodes + prefix_length).
        """
        num_nodes = self.current_size
        tree_mask = torch.eye(num_nodes, dtype=torch.bool, device=self.device)

        # Initialize ancestor_matrix with direct parent relationships as float16
        ancestor_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float16, device=self.device)
        parent_indices = self.parent_indices[:num_nodes]
        valid_parents = parent_indices != -1
        ancestor_matrix[valid_parents, parent_indices[valid_parents]] = 1.0

        # Compute transitive closure using float16 matrix multiplication
        while True:
            # Perform matrix multiplication
            new_relations = torch.matmul(ancestor_matrix, ancestor_matrix)
            # Threshold to determine new ancestor relationships
            new_relations = (new_relations > 0).to(torch.float16)
            # Identify newly discovered relationships
            new_relations = new_relations - ancestor_matrix
            new_relations[new_relations < 0] = 0.0  # Ensure no negative values
            if torch.sum(new_relations) == 0:
                break
            # Update ancestor_matrix with new relations
            ancestor_matrix += new_relations
            # Ensure binary representation
            ancestor_matrix = (ancestor_matrix > 0).to(torch.float16)

        # Convert ancestor_matrix to boolean
        ancestor_matrix_bool = ancestor_matrix > 0

        # Combine the direct ancestor relationships with self-attention
        tree_mask |= ancestor_matrix_bool

        # Optionally prepend a prefix mask
        if prefix_length > 0:
            prefix_mask = torch.ones((num_nodes, prefix_length), dtype=torch.bool, device=self.device)
            tree_mask = torch.cat([prefix_mask, tree_mask], dim=1)

        # Convert mask for attention (True -> no mask, False -> large negative)
        inverted_mask = (~tree_mask).to(self.prob_dtype) * torch.finfo(self.prob_dtype).min
        return inverted_mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, num_nodes, num_nodes + prefix_length)



    def set_node_extra_data(self, node_indices: torch.Tensor, data: torch.Tensor):
        """
        Assign extra data to specified nodes.

        Args:
            node_indices (torch.Tensor): Indices of target nodes.
            data (torch.Tensor): Data to assign.
        """
        self.extra_data[node_indices] = data

    def get_node_extra_data(self, node_indices: torch.Tensor) -> torch.Tensor:
        """
        Retrieve extra data from specified nodes.

        Args:
            node_indices (torch.Tensor): Indices of target nodes.

        Returns:
            torch.Tensor: Extra data of the specified nodes.
        """
        return self.extra_data[node_indices]

    def print_tree_structure(self, show_token_id: bool = True, show_probability: bool = False):
        """
        Display the tree structure in a readable format.

        Args:
            show_token_id (bool, optional): Whether to show token IDs. Defaults to True.
            show_probability (bool, optional): Whether to show cumulative probabilities. Defaults to False.
        """
        if not (show_token_id or show_probability):
            raise ValueError("At least one of show_token_id or show_probability must be True.")

        # Transfer data to CPU and convert to lists for easier processing
        parent = self.parent_indices[:self.current_size].cpu().tolist()
        tokens = self.token_ids[:self.current_size].cpu().tolist()
        probs = self.cumulative_probabilities[:self.current_size].cpu().tolist()

        # Initialize a list of empty lists for children
        children = [[] for _ in range(self.current_size)]

        # Populate the children list
        for child_idx in range(1, self.current_size):
            parent_idx = parent[child_idx]
            if parent_idx != -1:
                children[parent_idx].append(child_idx)

        def recurse_print(node, prefix=''):
            children_nodes = children[node]
            for i, child in enumerate(children_nodes):
                is_last = i == len(children_nodes) - 1
                connector = '└── ' if is_last else '├── '
                if show_token_id and show_probability:
                    node_repr = f"{tokens[child]} ({probs[child]:.4f})"
                elif show_token_id:
                    node_repr = f"{tokens[child]}"
                elif show_probability:
                    node_repr = f"({probs[child]:.4f})"
                else:
                    node_repr = ""
                print(prefix + connector + node_repr)
                extension = '    ' if is_last else '│   '
                recurse_print(child, prefix + extension)

        # Print root node
        if show_token_id and show_probability:
            root_repr = f"{tokens[0]} ({probs[0]:.4f})"
        elif show_token_id:
            root_repr = f"{tokens[0]}"
        elif show_probability:
            root_repr = f"({probs[0]:.4f})"
        else:
            root_repr = ""
        print(root_repr)

        # Start recursive printing from root
        recurse_print(0)

    def __repr__(self):
        return f"EfficientGPUTree(num_nodes={self.current_size}, device='{self.device}')"
