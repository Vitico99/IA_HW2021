from sklearn import linear_model
import numpy as np

EMPTY = ' '
PLAYER1 = 'X'
PLAYER2 = 'O'

WIN = 1
LOSE = -1
DRAW = 0
PLAYING = 2


class Player:

    """
    This method performs a move, selected by some strategy, in a game of tic tac toe ,represented by a 3x3 matrix,
    as the player represented by 'idp'.
     
    """
    def move(self, board, idp):
        move, _  = self.best_move(board, idp)
        return move

    """
    This method selects a move from the set of all available moves according to some strategy, the simplest one just select a random
    move from all available moves
    """
    def best_move(self, board, idp):
        empty_cells = set()
        for i in range(3):
            for j in range(3):
                if board[i][j] == EMPTY:
                    empty_cells.add((i,j))
        return empty_cells.pop(), None
    
    """
    This method checks if the given game of tic tac toe represented by a 3x3 matrix is over.
    In positive case returns the result for the player represented by 'idp'
    """
    @staticmethod
    def game_over(board, idp): 
        
        empty = 0
        ldiag = board[0][0]; rdiag = board[0][2]
        
        for i in range(3):
            if board[i][i] != ldiag: 
                ldiag = -1
            if board[i][2 - i] != rdiag:
                rdiag = -1

            row = board[i][0]; col = board[0][i]
            for j in range(3):
                if board[i][j] != row:
                    row = -1
                if board[j][i] != col:
                    col = -1
                if board[i][j] == EMPTY:
                    empty += 1

            if row != -1 and row != EMPTY:
                return WIN if row == idp else LOSE

            if col != -1 and col != EMPTY:
                return WIN if col == idp else LOSE
            
        if ldiag != -1 and ldiag != EMPTY:
            return WIN if ldiag == idp else LOSE

        if rdiag != -1 and rdiag != EMPTY:
            return WIN if rdiag == idp else LOSE
        
        return PLAYING if empty else DRAW
    

def match(p1, p2):
    board = [[EMPTY for j in range(3)] for i in range(3)]
    current = PLAYER1
    result = p1.move(board, current)
    print( np.array(board))
    while result == PLAYING:
        current = PLAYER1 if current == PLAYER2 else PLAYER2
        if current == PLAYER1:
            result = p1.move(board, current)
        else:
            result = p2.move(board, current)
        print(np.array(board))

    if result == DRAW:
        return DRAW
    elif result == WIN:
        return current
    else:
        return PLAYER1 if current == PLAYER2 else PLAYER2

"""
This player plays optimally every time by computing the value of all possible moves every time
"""
class MinimaxPlayer(Player):

    def best_move(self, board, idp):
        move = None; score = LOSE
        for i in range(3):
            for j in range(3):
                if board[i][j] == EMPTY: # for each possible play
                    board[i][j] = idp
                    result = self.game_over(board, idp) # if it is a final board its result is easy to compute
                    if result != PLAYING:
                        if result >= score:    
                            move  = (i, j)
                            score = result
                    else: # hte board is an intermediate state so we compute the best play of our opponent and our score is the opposite of that
                        _, result = self.best_move(board, PLAYER1 if idp == PLAYER2 else PLAYER2) 
                        if -result > score: 
                            move = (i, j)
                            score = -result
                        if -result == score and np.random.rand(1) < 0.5:
                            move = (i, j)
                            score = -result
                    board[i][j] = EMPTY
        return move, score


"""
This player stores the best possible move for each board previously analyzed to improve time efficiency
"""
class MemoMinimaxPlayer(MinimaxPlayer):
    def __init__(self):
        self.moves = {}
    
    def best_move(self, board, idp):
        board_str = self.board_to_string(board)
        try:
            return self.moves[board_str]
        except KeyError:
            self.moves[board_str] = MinimaxPlayer.best_move(self, board, idp)
            return self.moves[board_str]

    def board_to_string(self, board):
        result = ""
        for i in range(3):
            for j in range(3):
                result += str(board[i][j])
        return result


class TrainedPlayer(Player):

    def __init__(self, train=False):
        self.weights = np.random.rand(7)
        if train:
            self.train(5000)

    def best_move(self, board, idp):
        move = None; score = float('-inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == EMPTY:
                    board[i][j] = idp
                    approx = np.dot(self.weights, self.board_features(board, idp)) # approximate f(b)
                    if approx > score:
                        move = (i, j)
                        score = approx
                    if approx == score and np.random.rand(1) < 0.5:
                        move = (i, j)
                        score = approx
                    board[i][j] = EMPTY
        return move, score

    def train(self, iters, opponent=None):
        alpha = 0.001
        for i in range(iters):
            board = [[EMPTY for j in range(3)] for i in range(3)]
            current = PLAYER1 if i % 2 == 0 else PLAYER2 # lets rotate who starts

            pre_features = self.board_features(board, current) # current board
            pre_approx = np.dot(self.weights, pre_features) # actual approximation

            move = self.move(board, current)
            board[move[0]][move[1]] = current
            result = Player.game_over(board, current)

            while result == PLAYING:
                current = PLAYER1 if current == PLAYER2 else PLAYER2
                if current == PLAYER1:
                    succ_features = self.board_features(board, current) # next board
                    succ_approx = np.dot(self.weights, self.board_features(board, current)) # next approximation
                    
                    move = self.move(board, current)
                    board[move[0]][move[1]] = current
                    result = Player.game_over(board, current)
                    
                    self.update_weights(alpha, succ_approx, pre_approx, pre_features) # update the weights
                    pre_features = succ_features # udpdate current board
                    pre_approx = np.dot(self.weights, self.board_features(board, current)) # update current approx with new weights
                else:
                    move = self.move(board, current)
                    board[move[0]][move[1]] = current
                    result = Player.game_over(board, current)

            if result == WIN:
                succ_approx = 1 if current == (PLAYER1 if i % 2 == 0 else PLAYER2) else -1
            elif result == LOSE:
                succ_approx = -1 if current == (PLAYER1 if i % 2 == 0 else PLAYER2) else 1
            else:
                succ_approx = 0
            
            self.update_weights(alpha, succ_approx, pre_approx, pre_features)
                    

    def update_weights(self, learning_rate, train_v, approx, features): # weight update LMS rule
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + learning_rate * (train_v - approx) * features[i]


    def board_features(self, board, idp):
        x0 = 1  # Constant
        x1 = 0  # Number of rows/columns/diagonals with two of our own pieces and one emtpy field
        x2 = 0  # Number of rows/columns/diagonals with two of opponent's pieces and one empty field
        x3 = 0  # Number of rows/columns/diagonals with one own piece and two empty fields
        x4 = 0  # Number of rows/columns/diagonals with one opponent's piece and two empty fields
        x5 = 0  # Number of rows/columns/diagonals with three own pieces
        x6 = 0  # Number of rows/columns/diagonals with three opponent's pieces

        for i in range(3):
            own_rows = 0
            own_columns = 0
            enemy_rows = 0
            enemy_columns = 0
            empty_rows = 0
            empty_columns = 0
            for j in range(3):
                if board[i][j] == EMPTY:
                    empty_rows += 1
                elif board[i][j] == idp:
                    own_rows += 1
                else:
                    enemy_rows += 1
                if board[j][i] == 0:
                    empty_columns += 1
                elif board[j][i] == idp:
                    own_columns += 1
                else:
                    enemy_columns += 1

            if own_rows == 2 and empty_rows == 1:
                x1 += 1
            if enemy_rows == 2 and empty_rows == 1:
                x2 += 1
            
            if own_columns == 2 and empty_columns == 1:
                x1 += 1
            if enemy_columns == 2 and empty_columns == 1:
                x2 += 1

            if own_rows == 1 and empty_rows == 2:
                x3 += 1
            if own_columns == 1 and empty_columns == 2:
                x3 += 1
            
            if enemy_rows == 1 and empty_rows == 2:
                x4 += 1
            if enemy_columns == 1 and empty_columns == 2:
                x4 += 1

            if own_rows == 3:
                x5 += 1
            if own_columns == 3:
                x5 += 1

            if enemy_rows == 3:
                x6 += 1
            if enemy_columns == 3:
                x6 += 1

        for i in range(2):
            own_diagonal = 0
            enemy_diagonal = 0
            empty_diagonal = 0
            for j in range(3):
                if i == 0:
                    diagonal = board[2-j][j]
                else:
                    diagonal = board[j][j]
                if diagonal == EMPTY:
                    empty_diagonal += 1
                elif diagonal == idp:
                    own_diagonal += 1
                else:
                    enemy_diagonal += 1
            if own_diagonal == 2 and empty_diagonal == 1:
                x1 += 1
            if enemy_diagonal == 2 and empty_diagonal == 1:
                x2 += 1
            if own_diagonal == 1 and empty_diagonal == 2:
                x3 += 1
            if enemy_diagonal == 1 and empty_diagonal == 2:
                x4 += 1
            if own_diagonal == 3:
                x5 += 1
            if enemy_diagonal == 3:
                x6 += 1


        return [x0, x1, x2, x3, x4, x5, x6]
        
