# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()

def getPathFromParent(cur_state, parents):
    cur_state = (cur_state)
    path = []
    while cur_state != None:
        from game import Directions
        prev_state = parents[cur_state]
        if prev_state is None:
            break
        if prev_state[0] > cur_state[0]:
            path.append(Directions.WEST)
        elif prev_state[0] < cur_state[0]:
            path.append(Directions.EAST)
        elif prev_state[1] > cur_state[1]:
            path.append(Directions.SOUTH)
        elif prev_state[1] < cur_state[1]:
            path.append(Directions.NORTH)
        cur_state = prev_state
    path.reverse()
    return path

def getPathCostFromParent(cur_state, path_cost_parent):
    cur_state = (cur_state, 0)
    path_cost = 0
    while cur_state != None:
        print cur_state
        print path_cost_parent
        cur_state = path_cost_parent[cur_state[0]]
        if cur_state is None:
            break
        path_cost += cur_state[1]
    return path_cost   

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"

    #key:((state),direction,cost,cumulative cost, current key, parent key)
    node_dict = {}
    fringe = []
    fringe.append((problem.getStartState(), 'LStart', 0, 0, 0, 0))
    closed_set = {}
    node_dict[0] = (problem.getStartState(), 'LStart', 0, 0, 0, 0)
    while len(fringe) > 0:
        cur_state = fringe.pop()
        if problem.isGoalState(cur_state[0]):
            path = []
            while True:
                path.append(cur_state[1])
                cur_state = node_dict[cur_state[5]]
                if not cur_state[4]:
                    break
            path.reverse()
            return path
        closed_set[cur_state[0]] = cur_state[4]
        for successor in problem.getSuccessors(cur_state[0]):
            if successor[0] not in closed_set:
                new_key = len(node_dict)
                node_dict[new_key] = (successor[0], successor[1], successor[2], successor[2] + node_dict[cur_state[4]][3], new_key, cur_state[4])
                fringe.append(node_dict[new_key])
    return None

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"
    #key:((state),direction,cost,cumulative cost, current key, parent key)
    node_dict = {}
    fringe = []
    fringe.insert(0, (problem.getStartState(), 'LStart', 0, 0, 0, 0))
    closed_set = {}
    node_dict[0] = (problem.getStartState(), 'LStart', 0, 0, 0, 0)
    while len(fringe) > 0:
        cur_state = fringe.pop()
        closed_set[cur_state[0]] = cur_state[4]

        for successor in problem.getSuccessors(cur_state[0]):

            if successor[0] in closed_set: continue
            if successor[0] in fringe: continue

            new_key = len(node_dict)
            node_dict[new_key] = (successor[0], successor[1], successor[2], successor[2] + node_dict[cur_state[4]][3], new_key, cur_state[4])

            if problem.isGoalState(successor[0]):
                path = []
                cur_state = node_dict[new_key]
                while True:
                    path.append(cur_state[1])
                    cur_state = node_dict[cur_state[5]]
                    if not cur_state[4]:
                        break
                path.reverse()
                return path

            fringe.insert(0, node_dict[new_key])

    return None

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    #key:((state),direction,cost,cumulative cost, current key, parent key)
    node_dict = {}
    fringe = util.PriorityQueue()
    fringe.push((problem.getStartState(), 'LStart', 0, 0, 0, 0), 0)
    closed_set = {}
    node_dict[0] = (problem.getStartState(), 'LStart', 0, 0, 0, 0)
    while not fringe.isEmpty():
        cur_state = fringe.pop()
        if problem.isGoalState(cur_state[0]):
            path = []
            cur_state = node_dict[cur_state[4]]
            while True:
                path.append(cur_state[1])
                cur_state = node_dict[cur_state[5]]
                if not cur_state[4]:
                    break
            path.reverse()
            return path
        closed_set[cur_state[0]] = cur_state[4]

        for successor in problem.getSuccessors(cur_state[0]):

            if successor[0] in closed_set: continue
            #if successor[0] in fringe: continue

            new_key = len(node_dict)
            node_dict[new_key] = (successor[0], successor[1], successor[2], successor[2] + node_dict[cur_state[4]][3], new_key, cur_state[4])

            fringe.push(node_dict[new_key], node_dict[new_key][3])

    return None

def nullHeuristic(position, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return None

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    #key:((state),direction,cost,cumulative cost, current key, parent key)
    node_dict = {}
    fringe = util.PriorityQueue()
    fringe.push((problem.getStartState(), 'LStart', 0, 0, 0, 0), 0)
    closed_set = {}
    node_dict[0] = (problem.getStartState(), 'LStart', 0, 0, 0, 0)
    while not fringe.isEmpty():
        cur_state = fringe.pop()
        if problem.isGoalState(cur_state[0]):
            path = []
            cur_state = node_dict[cur_state[4]]
            while True:
                path.append(cur_state[1])
                cur_state = node_dict[cur_state[5]]
                if not cur_state[4]:
                    break
            path.reverse()
            return path
        closed_set[cur_state[0]] = cur_state[4]

        for successor in problem.getSuccessors(cur_state[0]):

            if successor[0] in closed_set: continue
            #if successor[0] in fringe: continue

            new_key = len(node_dict)
            node_dict[new_key] = (successor[0], successor[1], successor[2], successor[2] + node_dict[cur_state[4]][3], new_key, cur_state[4])
            h = heuristic(node_dict[new_key][0], problem)
            fringe.push(node_dict[new_key], node_dict[new_key][3] + h)

    return None
    


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
