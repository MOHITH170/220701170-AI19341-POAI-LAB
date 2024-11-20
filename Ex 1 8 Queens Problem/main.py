 You are given an 8x8 board; find a search technique to place 8 queens so no queen can attack
any other queen on the chessboard. A queen can only be attacked if it lies on the same row, or same
column, or the same diagonal as any other queen. Print all the possible configurations.


//main.py//

def print_board(board):
    for row in board:
        print(" ".join("Q" if col == 1 else "." for col in row))
    print()

def is_safe(board, row, col, n):
    # Check left side of the current row
    for i in range(col):
        if board[row][i] == 1:
            return False

    # Check upper diagonal on left side
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    # Check lower diagonal on left side
    for i, j in zip(range(row, n), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    return True

def solve_n_queens(board, col, n, solutions):
    # If all queens are placed, store the solution
    if col >= n:
        solutions.append([row[:] for row in board])  # Deep copy of the board
        return

    for i in range(n):
        if is_safe(board, i, col, n):
            # Place the queen
            board[i][col] = 1
            # Recurse for the next column
            solve_n_queens(board, col + 1, n, solutions)
            # Backtrack
            board[i][col] = 0

def n_queens(n=8):
    board = [[0 for _ in range(n)] for _ in range(n)]  # Initialize an n x n chessboard
    solutions = []
    solve_n_queens(board, 0, n, solutions)
    return solutions

# Solve the 8-Queens problem
solutions = n_queens(8)

# Print all solutions
print(f"Total Solutions: {len(solutions)}\n")
for idx, solution in enumerate(solutions, 1):
    print(f"Solution {idx}:")
    print_board(solution)
