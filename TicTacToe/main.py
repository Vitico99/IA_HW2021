# Tic Tac Toe game with GUI
# using tkinter

# importing all necessary libraries
import random
import tkinter
from tkinter import *
from functools import partial
from tkinter import messagebox
from copy import deepcopy
from player import *

EMPTY = ' '
PLAYER1 = 'X'
PLAYER2 = 'O'

# sign variable to decide the turn of which player
CURRENT = PLAYER1

# Creates an empty board
global board
board = [[ EMPTY for x in range(3)] for y in range(3)]
global p2
global result
p2 = MinimaxPlayer()

# Check if the player can push the button or not
def isfree(i, j):
	return board[i][j] == EMPTY
    
def get_pc(i,j,gb,l1,l2):
    global CURRENT
    if board[i][j] == EMPTY:
        if CURRENT == PLAYER1:
            l1.config(state=DISABLED)
            l2.config(state=ACTIVE)
            board[i][j] = PLAYER1
        else:
            button[i][j].config(state=ACTIVE)
            l2.config(state=DISABLED)
            l1.config(state=ACTIVE)
            board[i][j] = PLAYER2
        CURRENT = PLAYER1 if CURRENT == PLAYER2 else PLAYER2
        button[i][j].config(text=board[i][j])
    result = Player.game_over(board, PLAYER1)
    if result != PLAYING:
        if result == WIN:
            box = messagebox.showinfo("Winner", "Player1 won the match")
        elif result == LOSE:
            box = messagebox.showinfo("Loser", "Player2 won the match" )
        elif result == DRAW:
            box = messagebox.showinfo("Draw", "Tie match")
        withpc(gb, p2)
    else:
        if CURRENT == PLAYER2:
            move = p2.move(board, PLAYER2)
            button[move[0]][move[1]].config(state=DISABLED)
            get_pc(move[0], move[1], gb, l1, l2)

# Create the GUI of game board for play along with system
def gameboard_pc(game_board, l1, l2):
	global button
	button = []
	for i in range(3):
		m = 3+i
		button.append(i)
		button[i] = []
		for j in range(3):
			n = j
			button[i].append(j)
			get_t = partial(get_pc, i, j, game_board, l1, l2)
			button[i][j] = Button(
				game_board, bd=5, command=get_t, height=4, width=8)
			button[i][j].grid(row=m, column=n)
	game_board.mainloop()

def withpc(game_board, ia):
    global board
    global CURRENT
    global p2
    p2 = ia
    board = [[EMPTY for j in range(3)] for i in  range(3)]
    CURRENT = PLAYER1
    game_board.destroy()
    game_board = Tk()
    game_board.title("Tic Tac Toe")
    l1 = Button(game_board, text="Player : X", width=10)
    l1.grid(row=1, column=1)
    l2 = Button(game_board, text="Computer : O", width=10, state=DISABLED)
    l2.grid(row=2, column=1)
    gameboard_pc(game_board, l1, l2)



def play():
    menu = Tk()
    menu.geometry("250x250")
    menu.title("Tic Tac Toe")
    wpc_minimax = partial(withpc, menu, MemoMinimaxPlayer())
    wpc_tarfunc = partial(withpc, menu, TrainedPlayer(train=True))

    head = Button(menu, text = "---Welcome to Tic-Tac-Toe---", 
                    activeforeground='red', 
                    activebackground="yellow", bg="red", fg="yellow", width=500, font="summer", bd=5)

    B1 = Button(menu, text = "MiniMaxPlayer", command=wpc_minimax, 
                    activeforeground='red', 
                    activebackground="yellow", bg="red", fg="yellow", width=500, font="summer", bd=5)

    B2 = Button(menu, text = "TrainedPlayer", command=wpc_tarfunc, 
                    activeforeground='red', 
                    activebackground="yellow", bg="red", fg="yellow", width=500, font="summer", bd=5)

    B3 = Button(menu, text = "Quit", command=menu.quit, 
                    activeforeground='red', 
                    activebackground="yellow", bg="red", fg="yellow", width=500, font="summer", bd=5)

    head.pack(side='top')
    B1.pack(side='top')
    B2.pack(side='top')
    B3.pack(side='top')
    menu.mainloop()


# Call main function
if __name__ == '__main__':
	play()
