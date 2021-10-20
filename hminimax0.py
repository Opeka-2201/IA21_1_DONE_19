# Complete this class for all parts of the project

import math
from pacman_module.game import Agent
from pacman_module.pacman import Directions
from pacman_module.util import manhattanDistance as distance

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
        self.it = 0
        # Variable to stock the number of iterations of get_action

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
        # player = 0, pacman plays first
        # on each call of get_action, it is incremented
        # at first iteration, alpha and beta are set on infinity for Alpha-Beta pruning
        visitedNodes = set()
        self.it += 1
        return self.hminimax0(state, 0, -math.inf, +math.inf, 0, visitedNodes)[1]

    def hminimax0(self, state, player, alpha, beta, treeDepth, visitedNodes):
        """
        Computes the HMinimax value for a node and the action to take on the node
        If the state = currentState and if player = Pacman, tells the player the best move.

        Arguments:
        ----------
        - `state`: current gameState, see class
                    `pacman.gameState`.
        - `player`: player's index, 0 = Pacman, >0 = ghosts.
        - 'alpha`: Bound in Alpha-Beta pruning (low).
        - `beta: Bound in Alpha-Beta pruning (hign).
        - `treeDepth`: recursion tree's depth.
        - `visitedNodes`: Set that keeps track of already visited nodes.

        Return:
        -------
        - HMinimax value for a node and best action relative to the node.
        """

        # check cutoff
        if self.cutoff(state,treeDepth):
            return self.compute(state), None
        
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
        
        for succResult, succMove in succMoves(state, player):
            if keyHash(succResult) not in visitedNodes:
                visitedNodes.add(keyHash(succResult))
                computedValue = self.hminimax0(succResult, opponent, alpha, beta, treeDepth + 1, visitedNodes)[0]
                visitedNodes.remove(keyHash(succResult))

                # if pacman -> max utility, if ghost -> min utility
                if player == 0:
                    if computedValue > bestValue:
                        bestValue = computedValue
                        bestMove = succMove
                
                # Alpha-Beta Pruning
                if computedValue >= beta:
                    return computedValue, succMove
                
                alpha = max(alpha, computedValue)
            
            else:
                if computedValue < bestValue:
                    bestValue = computedValue
                    bestMove = succMove
                
                if computedValue <= alpha:
                    return computedValue, succMove
                
                beta = min(beta, computedValue)
            
        return bestValue, bestMove

    def quiescent(self, state):
        """
        Returns a boolean to check if the state is not going to drasticly change in
        next states (if the state is quiescent)

        Arguments:
        ----------
        - `state`: current gameState, see class
                    `pacman.gameState`.

        Return:
        -------
        - Boolean to check if the state is quiescent
        """
        
        pacmanPosition = state.getPacmanPosition()
        ghostPosition = state.getGhostPosition(1)
        xPacman, yPacman = state.getPacmanPosition()
        xGhost, yGhost = state.getGhostPosition(1)
        ghostOrientation = state.getGhostDirection(1)
        food = state.getFood().asList().copy()
        foodFar = True
        ghostClose = False

        # Is Pacman able to eat next turn
        for i in food:
            if distance(pacmanPosition, i) == 1:
                foodFar = False
        
        # Is Ghost able to eat Pacman next turn
        if distance(pacmanPosition, ghostPosition) == 1:
            if (ghostOrientation and yGhost < yPacman is Directions.NORTH) \
                    or (ghostOrientation and xGhost < xPacman is Directions.EAST) \
                    or (ghostOrientation and yGhost > yPacman is Directions.SOUTH) \
                    or (ghostOrientation and xGhost > xPacman is Directions.WEST):
                ghostClose = True
        
        boolean = not ghostClose and foodFar
        return boolean



    def cutoff(self, state, treeDepth):
        """
        Returns a boolean to check if we should stop computing

        Arguments:
        ----------
        - `state`: current gameState, see class
                    `pacman.gameState`.
        - `treeDepth`: recursion tree's depth.

        Return:
        -------
        - Boolean to check if we should stop computing
        """

        if state.isWin() or state.isLose():
            return True

        boolean = treeDepth > 9 or (treeDepth > 7 and self.quiescent(state))
        return boolean

    def compute(self, state):
        """
        Computes value of given state ~ utility score for Pacman

        Arguments:
        ----------
        - `state`: current gameState, see class
                    `pacman.gameState`.

        Return:
        -------
        - The approximation of utility score of state
        """

        pacmanPosition = state.getPacmanPosition()
        ghostPosition = state.getGhostPosition(1)

        # check if it is better for Pacman to kill itself
        if self.it > 4 * (state.getFood().height + state.getFood().width):
            return state.getScore() - 5 * distance(pacmanPosition, ghostPosition)

        food = state.getFood().asList().copy()
        distanceLeft = 0
        closest = pacmanPosition

        while food:
            distanceMin = +math.inf
            foodClose = None

            for i in food:
                distanceCurrent = distance(i, closest)
                if distanceCurrent < distanceMin:
                    distanceMin = distanceCurrent
                    foodClose = i
            
            distanceLeft += distanceMin
            closest = foodClose
            food.remove(foodClose)

        # eval of utility score
        return state.getScore() + 10 * state.getNumFood() - distanceLeft