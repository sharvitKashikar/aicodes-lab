```markdown
# Tic-Tac-Toe Game

This module (`tic.py`) implements a classic text-based Tic-Tac-Toe game for two players. It's a simple command-line game where players take turns marking spaces in a 3x3 grid with their respective signs (X or O) to achieve three in a row to win.

## Game Play

1.  **Board Initialization:** The game starts with an empty 3x3 board.
2.  **Player Turns:** Players 'X' and 'O' take turns to make moves.
3.  **Input:** Players are prompted to choose a position (1-9) on the board.
4.  **Input Validation:** The game validates inputs to ensure they are numbers between 1 and 9 and that the chosen position is not already taken.
5.  **Winning Condition:** After each move, the game checks if the current player has achieved three of their marks in a row (horizontally, vertically, or diagonally).
6.  **Tie Game:** If all 9 positions are filled without a winner, the game is declared a tie.

## Example Usage

To play the Tic-Tac-Toe game, execute the `tic.py` script directly from your terminal:

```python
# Step 1: Create the board
board = [" " for _ in range(9)]

# Step 2: Function to print the board
def print_board():
    print(f"{board[0]} | {board[1]} | {board[2]}")
    print("--+---+--")
    print(f"{board[3]} | {board[4]} | {board[5]}")
    print("--+---+--")
    print(f"{board[6]} | {board[7]} | {board[8]}")

# Step 3: Function to check for winner
def check_winner(player):
    win_positions = [
        [0,1,2], [3,4,5], [6,7,8],  # rows
        [0,3,6], [1,4,7], [2,5,8],  # columns
        [0,4,8], [2,4,6]            # diagonals
    ]
    for pos in win_positions:
        if board[pos[0]] == board[pos[1]] == board[pos[2]] == player:
            return True
    return False

# Step 4: Main game loop
def play_game():
    current_player = "X"
    for turn in range(9):
        print_board()

        # Input validation loop
        while True:
            move_input = input(f"{current_player}'s turn. Choose position (1-9): ")
            if not move_input.isdigit():
                print("❗ Please enter a valid number between 1 and 9.")
                continue
            move = int(move_input) - 1
            if move < 0 or move > 8:
                print("❗ Invalid position! Choose a number between 1 and 9.")
                continue
            if board[move] != " ":
                print("❗ Position already taken! Try again.")
                continue
            break  # valid move

        board[move] = current_player
        if check_winner(current_player):
            print_board()
            print(f"🎉 {current_player} wins!")
            return

        current_player = "O" if current_player == "X" else "X"

    print_board()
    print("It's a tie!")

# Step 5: Start the game
play_game()
```

## Sample Game Flow (User Interactions)

Upon running `tic.py`, you will be prompted to enter moves. Here's an example interaction:

```
  |   |  
--+---+--
  |   |  
--+---+--
  |   |  
X's turn. Choose position (1-9): 1

X |   |  
--+---+--
  |   |  
--+---+--
  |   |  
O's turn. Choose position (1-9): 5

X |   |  
--+---+--
  | O |  
--+---+--
  |   |  
X's turn. Choose position (1-9): 2

X | X |  
--+---+--
  | O |  
--+---+--
  |   |  
O's turn. Choose position (1-9): 3

X | X | O
--+---+--
  | O |  
--+---+--
  |   |  
X's turn. Choose position (1-9): 8

X | X | O
--+---+--
  | O |  
--+---+--
  | X |  
O's turn. Choose position (1-9): 6

X | X | O
--+---+--
  | O | O
--+---+--
  | X |  
X's turn. Choose position (1-9): 4

X | X | O
--+---+--
X | O | O
--+---+--
  | X |  
O's turn. Choose position (1-9): 7

X | X | O
--+---+--
X | O | O
--+---+--
O | X |  
X's turn. Choose position (1-9): 9

X | X | O
--+---+--
X | O | O
--+---+--
O | X | X
It's a tie!
```
