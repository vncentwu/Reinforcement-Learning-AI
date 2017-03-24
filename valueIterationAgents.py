# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        # populate values dict
        for _ in xrange(self.iterations):
          copy = util.Counter()
          for tState in self.mdp.getStates():
            # if no more legal moves
            if self.mdp.isTerminal(tState):
              copy[tState] = 0
            else:
              highestQVal = float("-inf")
              for action in self.mdp.getPossibleActions(tState):
                currQVal = self.computeQValueFromValues(tState, action)
                highestQVal = max(highestQVal, currQVal)
                copy[tState] = highestQVal
          self.values = copy




    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        qVal = 0
        # Q_k+1 (s, a) <- Sum(T(s, a, s')[R(s, a, s') + y * max(Q_k(s', a'))])
        for nextState, probability in self.mdp.getTransitionStatesAndProbs(state, action):
          reward = self.mdp.getReward(state, action, nextState)
          discount = self.discount
          nextVal = self.values[nextState]
          qVal += probability * (reward + discount * nextVal)
        return qVal

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        # No more legal actions
        if self.mdp.isTerminal(state):
          return None

        # If not terminal state, return action with highest QValue
        highestQVal = float('-inf')
        highestAction = None

        # Loop through and call computeQValue for each state, action pair
        for action in self.mdp.getPossibleActions(state):
          qVal = self.computeQValueFromValues(state, action)
          if qVal > highestQVal:
            highestQVal = qVal            
            highestAction = action
        return highestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
