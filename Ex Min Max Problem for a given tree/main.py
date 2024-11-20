def minimax(node, depth, is_maximizing, tree):
    """
    General Minimax algorithm.

    :param node: Current node in the tree.
    :param depth: Depth of the current node in the game tree.
    :param is_maximizing: True if the current player is the maximizing player, False if minimizing.
    :param tree: Dictionary representation of the game tree.
    :return: The optimal value at the current node.
    """
    # Base case: If the node is a leaf node (no children), return its value
    if node not in tree:
        return node  # Node itself is the value for leaf nodes

    if is_maximizing:
        best_value = float('-inf')
        for child in tree[node]:
            val = minimax(child, depth - 1, False, tree)
            best_value = max(best_value, val)
        return best_value
    else:
        best_value = float('inf')
        for child in tree[node]:
            val = minimax(child, depth - 1, True, tree)
            best_value = min(best_value, val)
        return best_value

# Example Game Tree
game_tree = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F', 'G'],
    'D': [3],  # Leaf node
    'E': [5],  # Leaf node
    'F': [6],  # Leaf node
    'G': [9]   # Leaf node
}

# Example Usage
root = 'A'  # Starting node
optimal_value = minimax(root, 3, True, game_tree)
print(f"The optimal value is: {optimal_value}")
