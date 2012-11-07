# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (newFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    dist_to_food = []
    dist_to_ghost = []
    food_list = newFood.asList()
    for food in food_list:
        dist_to_food += [manhattanDistance(food, newPos)]
    for ghost in newGhostStates:
        dist_to_ghost += [manhattanDistance(ghost.getPosition(), newPos)]
    score_to_send = 0
    current_score = successorGameState.getScore()
    if len(food_list) > 0:
        return current_score + (min(dist_to_ghost))/((min(dist_to_food)*2))
    return current_score

    #return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"

    pm_state = gameState.getPacmanState()
    pm_actions = gameState.getLegalPacmanActions()
    list_actions = {}
    depth = 0
    for action in pm_actions:
      if (action == Directions.STOP): continue
      pm_state = gameState.generateSuccessor(0, action)
      value = self.minimax(self.index+1, pm_state, depth)
      list_actions[action] = value
    return max(list_actions, key=list_actions.get)

  def minimax(self, a_index, state, depth):
    if (depth == self.depth) or state.isWin() or state.isLose():
      return self.evaluationFunction(state)
    if (a_index >= state.getNumAgents()):
      depth += 1
      a_index = 0
    if a_index != 0:
      return self.min_value(a_index, state, depth)
    else:
      return self.max_value(a_index, state, depth)

  def max_value(self, a_index, state, depth):
    v = float('-inf')
    legal_actions = state.getLegalActions(a_index)
    for action in legal_actions:
      if action == Directions.STOP: continue
      successor = state.generateSuccessor(a_index, action)
      temp = self.minimax(a_index+1, successor, depth)
      if temp > v:
        v = temp
    return v

  def min_value(self, a_index, state, depth):
    v = float('+inf')
    legal_actions = state.getLegalActions(a_index)
    for action in legal_actions:
      if action == Directions.STOP: continue
      successor = state.generateSuccessor(a_index, action)
      temp = self.minimax(a_index+1, successor, depth)
      if temp < v:
        v = temp
    return v

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    pm_actions = gameState.getLegalActions(self.index)
    list_actions = {}
    alpha = float('-inf')
    beta = float('+inf')
    value = alpha
    depth = 0
    act = 0
    for action in pm_actions:
       if action == Directions.STOP: continue
       pm_state = gameState.generateSuccessor(0, action)
       value = self.alpha_beta(self.index+1, pm_state, alpha, beta, depth+1)
       alpha = max(alpha, value)
       list_actions[action] = value
    #print list_actions
#    exit()
    #print max(list_actions, key=list_actions.get)
    return max(list_actions, key=list_actions.get)
    #util.raiseNotDefined()
#    v = self.alpha_beta(self.index, gameState, alpha, beta, depth+1, act)
    #print v[0]
#    return v[1]

  def alpha_beta(self, a_index, state, alpha, beta, depth):
    if (depth == self.depth) or state.isWin() or state.isLose():
        return self.evaluationFunction(state)
    if a_index == state.getNumAgents():
        depth += 1
        a_index = 0
    if a_index != 0:
       return self.min_value(a_index, state, alpha, beta, depth)
    else:
       return self.max_value(a_index, state, alpha, beta, depth)

  def max_value(self, a_index, state, alpha, beta, depth):
    #print "max a_index:", a_index," alpha:", alpha, " beta:",beta
    v = float('-inf')
    legal_actions = state.getLegalActions(a_index)
    for action in legal_actions:
      if action == Directions.STOP: continue
      successor = state.generateSuccessor(a_index, action)
      temp = self.alpha_beta(a_index+1, successor, alpha, beta, depth)
      v = max(temp, v)
      if v >= beta:
          #print "max prune at ", v
          return v
      alpha = max(alpha, v)
      #print "max value:", v
    return v

  def min_value(self, a_index, state, alpha, beta, depth):
    #print "min a_index:", a_index," alpha:", alpha, " beta:",beta
    v = float('+inf')
    legal_actions = state.getLegalActions(a_index)
    for action in legal_actions:
      if action == Directions.STOP: continue
      successor = state.generateSuccessor(a_index, action)
      temp = self.alpha_beta(a_index+1, successor, alpha, beta, depth)
      v = min(temp, v)
      if v <= alpha:
          #print "min prune at ", v
          return v
      beta = min(beta, v)
      #print "min value:", v
    return v

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    pm_actions = gameState.getLegalActions(self.index)
    list_actions = {}
    depth = 0
    for action in pm_actions:
        if (action == Directions.STOP): continue
        pm_state = gameState.generateSuccessor(0, action)
        value = self.expectimax(self.index+1, pm_state, depth+1)
        list_actions[action] = value
    return max(list_actions, key=list_actions.get)

  def expectimax(self, a_index, state, depth):
      if (depth == self.depth) or state.isWin() or state.isLose():
          return self.evaluationFunction(state)
      if a_index == state.getNumAgents():
          depth += 1
          a_index = 0
      if a_index != 0:
          return self.exp_value(a_index, state, depth)
      else:
          return self.max_value(a_index, state, depth)

  def max_value(self, a_index, state, depth):
      v = float('-inf')
      legal_actions = state.getLegalActions(a_index)
      for action in legal_actions:
          if action == Directions.STOP: continue
          successor = state.generateSuccessor(a_index, action)
          temp = self.expectimax(a_index+1, successor, depth)
          v = max(temp, v)
      return v

  def exp_value(self, a_index, state, depth):
      v = 0
      legal_actions = state.getLegalActions(a_index)
      if Directions.STOP in legal_actions:
          legal_actions.remove(Directions.STOP)
      for action in legal_actions:
          successor = state.generateSuccessor(a_index, action)
          p = 1.0/len(legal_actions)
          temp = self.expectimax(a_index+1, successor, depth)
          v += (p * temp)
      return v

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newCapsulesPos = currentGameState.getCapsules()

    dist_to_food = []
    dist_to_ghost = []
    dist_to_capsule = []
    scared_times = []
    food_score = 0
    food_list = newFood.asList()
    for food in food_list:
        dist_to_food += [manhattanDistance(food, newPos)]
    for ghost in newGhostStates:
        dist_to_ghost += [manhattanDistance(ghost.getPosition(), newPos)]
    for capsule in newCapsulesPos:
        dist_to_capsule += [manhattanDistance(capsule, newPos)]
#    for time in newScaredTimes:
#        if not time:
#            dist_to_ghost = [9999]
        #print scared_times
    score = currentGameState.getScore()

    if len(newGhostStates) > 0:
        if min(dist_to_ghost) == 0:
            ghost_score = 0.1
        else:
            ghost_score = 1.0 - 1.0/min(dist_to_ghost)
    else:
        ghost_score = 0
    if len(food_list) > 0:
        if min(dist_to_food) == 0:
            food_score = 1.1
        else:
            food_score = 1.0/min(dist_to_food)
    if len(dist_to_capsule) > 0:
        if min(dist_to_capsule) == 0:
            capsule_score = 1.1
        else:
            capsule_score = 1.0/min(dist_to_capsule)
    else:
        capsule_score = 0


    return score + ghost_score + food_score + capsule_score

    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

#class TestState:
#    def __init__(self, tree):
#        self.tree = tree
#    def getNumAgents(self):
#        return 2
#    def generateSuccessor(self,id, ac):
#        return TestState(self.tree[ac])
#    def getLegalActions(self,agent):
#        if isinstance(self.tree, list): return xrange(len(self.tree))
#        else: return []
#    def getScore(self):
#        return self.tree
#
#if __name__ == '__main__':
#    state = TestState([[[4,6],[7,9]],[[1,2],[0,1]],[[8,1],[9,2]]])
#    agent = AlphaBetaAgent(depth=2)
#    action = agent.getAction(state)
