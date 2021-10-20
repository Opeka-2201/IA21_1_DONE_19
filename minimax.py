# Complete this class for all parts of the project

import math
from pacman_module.game import Agent
from pacman_module.pacman import Directions

def succMoves(state, player):
    """
    Returns succesors states and moves available to player (ghosts or Pacman).

    Arguments:
    ----------
    - `state`: current gameState, see class
                `pacman.gameState`.
    - `player`: player's index, 0 = Pacman, >0 = ghosts.

    Return:
    -------
    - Returns succesors states and moves available to player
    """

    if player == 0:
        return state.generatePacmanSuccessors()
    else:
        return state.generateGhostSuccessors(player)

def keyHash(state):
    """
    Returns a unique hash to identifie the game's state (food, positions of ghosts 
    (+ directions) and Pacman).

    Arguments:
    ----------
    - `state`: current gameState, see class
                `pacman.gameState`.

    Return:
    -------
    Returns a unique hash to identifie the game's state.
    """

    return state.getFood(), state.getPacmanPosition(), state.getGhostPosition(1), state.getGhostDirection(1)

class PacmanAgent(Agent):
    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.args = args

        # We want to remember previously computed nodes
        # To implement this transposition table, we will use a dictionary
        # This method ensures we don't compute twice the same node
        # The key is a combination of an hash of the state and the score

        self.computedNodes = dict()
        self.depth = 0

    def get_action(self, state):
        """
        Given a pacman game state, returns a legal move.

        Arguments:
        ----------
        - `state`: the current game state. See FAQ and class
                   `pacman.GameState`.

        Return:
        -------
        - A legal move as defined in `game.Directions`.
        """

        # To avoid visiting twice a node in recursion, we'll keep track of visited nodes in a set

        visitedNodes = set()

        # player = 0, pacman plays first
        # alpha and beta are initially infinite bounds to compute alpha beta pruning

        return self.minimax(state, 0, -math.inf, +math.inf, visitedNodes)[1]
    
    def minimax(self, state, player, alpha, beta, visitedNodes):
        """
        Computes the Minimax value for a node and the action to take on the node
        If the state = currentState and if player = Pacman, tells the player the best move.

        Arguments:
        ----------
        - `state`: current gameState, see class
                    `pacman.gameState`.
        - `player`: player's index, 0 = Pacman, >0 = ghosts.
        - `alpha`: Bound in Alpha-Beta pruning (low).
        - `beta: Bound in Alpha-Beta pruning (hign).
        - `visitedNodes`: Set that keeps track of already visited nodes.

        Return:
        -------
        - Minimax value for a node and best action relative to the node.
        """

        if state.isLose() or state.isWin():
            return self.utilityFunction(state), None
            # if the state is terminal, no moves are required

        if (keyHash(state), state.getScore()) in self.computedNodes:
            return self.computedNodes[(keyHash(state), state.getScore())]
        
        if player == 1:
            opponent = 0
        else:
            opponent = 1
        
        # initiate the minimax algorithm
        bestMove = Directions.STOP
        if player == 0:
            bestValue = -math.inf
        else:
            bestValue = +math.inf

        self.depth = 0

        for succResult, succAction in succMoves(state, player):
            if self.depth > 12:
                return bestValue, bestMove
            if keyHash(succResult) not in visitedNodes:
                visitedNodes.add(keyHash(succResult))
                computedValue = self.minimax(succResult,opponent,alpha,beta,visitedNodes)[0]
                visitedNodes.remove(keyHash(succResult))

                if player == 0:
                    # Pacman -> max utility
                    if computedValue > bestValue:
                        bestValue = computedValue
                        bestMove = succAction

                    # Alpha-Beta pruning
                    if computedValue >= beta:
                        self.computedNodes[(keyHash(state), state.getScore())] = computedValue, succAction
                        return computedValue, succAction
                    
                    alpha = max(alpha, computedValue)
                
                else:
                    # Ghost -> min utility
                    if computedValue < bestValue:
                        bestValue = computedValue
                        bestMove = succAction

                    # Alpha-Beta pruning
                    if computedValue <= alpha:
                        self.computedNodes[(keyHash(state), state.getScore())] = computedValue, succAction
                        return computedValue, succAction
                    
                    beta = min(beta, computedValue)
            
            self.depth += 1

        self.computedNodes[(keyHash(state), state.getScore())] = bestValue, bestMove
        return bestValue, bestMove
    
    def utilityFunction(self, state):
        """
        Returns utility function's value, to maximize by Pacman and minimize by the ghosts.

        Arguments:
        ----------
        - `state` : current gameState, see class
                    `pacman.gameState`.

        Return:
        -------
        - Returns utility function's value.
        """

        return state.getScore()