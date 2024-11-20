def water_jug_dfs(jug1, jug2, target):
    """
    Solves the Water Jug Problem using a simple DFS approach.

    Args:
        jug1 (int): Capacity of the first jug.
        jug2 (int): Capacity of the second jug.
        target (int): Target amount of water.

    Returns:
        str: Solution path or "No solution."
    """
    visited = set()
    stack = [(0, 0)]  # Initial state with both jugs empty

    while stack:
        x, y = stack.pop()

        # If the target amount is achieved
        if x == target or y == target:
            return f"Solution: Jug1 = {x}, Jug2 = {y}"

        # Mark current state as visited
        if (x, y) in visited:
            continue
        visited.add((x, y))

        # Generate possible states
        next_states = [
            (jug1, y),  # Fill Jug 1
            (x, jug2),  # Fill Jug 2
            (0, y),     # Empty Jug 1
            (x, 0),     # Empty Jug 2
            (min(x + y, jug1), max(0, x + y - jug1)),  # Pour Jug 2 -> Jug 1
            (max(0, x + y - jug2), min(x + y, jug2))   # Pour Jug 1 -> Jug 2
        ]

        # Add unvisited states to the stack
        for state in next_states:
            if state not in visited:
                stack.append(state)

    return "No solution."


# Main function
if __name__ == "__main__":
    jug1 = int(input("Enter capacity of Jug 1: "))
    jug2 = int(input("Enter capacity of Jug 2: "))
    target = int(input("Enter target amount: "))

    # Check if the target is valid
    if target > max(jug1, jug2):
        print("Target cannot be greater than the capacity of the largest jug.")
    else:
        result = water_jug_dfs(jug1, jug2, target)
        print(result)
