import json, os
import numpy as np
import cvxpy as cp

class MDP:
  transitions = {}
  states = {}
  rewards = []
  actions = []

  def __init__(self, step_cost):
    self.step_cost = step_cost
    self.num_states = 60
    self.setupInitialProb()
    self.numberStates()
    self.setupTransition()
    self.setupRewards()

  def numberStates(self):
    for h in range(5):
      for a in range(4):
        for s in range(3):
          self.states[(h, a, s)] = 12 * h + 3 * a + s

  def setupInitialProb(self):
    self.alpha = [0] * (self.num_states)
    self.alpha[59] = 1

  def setupRewards(self):
    for action in self.actions:
      if action[1] == 'TERMINAL':
        self.rewards.append(0)
      else:
        self.rewards.append(self.step_cost)

  def setupTransition(self):
    for state in self.states.keys():
      self.transitions[state] = {'SHOOT' : {}, 'DODGE' : {}, 'RECHARGE' : {}}
      state_h, state_a, state_s = state
      
      if state_h == 0:
        self.actions.append([state, 'TERMINAL'])
      
      if state_h > 0:       
        if state_s > 0:
          if state_a > 0:
            self.actions.append([state, 'SHOOT'])
            self.transitions[state]['SHOOT'][(state_h - 1, state_a - 1, state_s - 1)] = 0.5
            self.transitions[state]['SHOOT'][(state_h - 0, state_a - 1, state_s - 1)] = 0.5
          
          self.actions.append([state, 'DODGE'])
          
          if state_s == 2:
            if state_a != 3:
              self.transitions[state]['DODGE'][(state_h, state_a + 0, state_s - 1)] = 0.8 * 0.2
              self.transitions[state]['DODGE'][(state_h, state_a + 1, state_s - 1)] = 0.8 * 0.8
              self.transitions[state]['DODGE'][(state_h, state_a + 0, state_s - 2)] = 0.2 * 0.2
              self.transitions[state]['DODGE'][(state_h, state_a + 1, state_s - 2)] = 0.2 * 0.8
            else:       
              self.transitions[state]['DODGE'][(state_h, state_a, state_s - 1)] = 0.8
              self.transitions[state]['DODGE'][(state_h, state_a, state_s - 2)] = 0.2
          else:
            if state_a != 3:
              self.transitions[state]['DODGE'][(state_h, state_a + 1, state_s - 1)] = 0.8
              self.transitions[state]['DODGE'][(state_h, state_a + 0, state_s - 1)] = 0.2
            else:
              self.transitions[state]['DODGE'][(state_h, state_a + 0, state_s - 1)] = 1.0
         
        if state_s != 2:
          self.actions.append([state, 'RECHARGE'])
          self.transitions[state]['RECHARGE'][(state_h, state_a, state_s + 1)] = 0.8
          self.transitions[state]['RECHARGE'][(state_h, state_a, state_s + 0)] = 0.2

class LP:

  def __init__(self, MDP):
    self.states = MDP.states
    self.transitions = MDP.transitions
    self.num_actions = len(MDP.actions)
    self.num_states = MDP.num_states
    self.actions = MDP.actions
    self.alpha = MDP.alpha
    self.A = []
    self.rewards = MDP.rewards
    self.setupMatrixA()
    self.setupVectors()
    self.outputs = self.computeSimplex()

  def setupMatrixA(self):
    self.A = [[0] * self.num_actions  for i in range(self.num_states)]

    for action_num, action in enumerate(self.actions):
      state = action[0]
      state_num = self.states[state]

      if action[1] == 'TERMINAL':
        self.A[state_num][action_num] += 1
      else:  
        for transition in self.transitions[state][action[1]].items():
          to_state, trans_prob = transition
          to_state_num = self.states[to_state]
          self.A[to_state_num][action_num] -= trans_prob
          self.A[state_num][action_num] += trans_prob

  def setupVectors(self):
    self.A_matrix = np.reshape(self.A, (self.num_states, self.num_actions))
    self.alpha_vector = np.reshape(self.alpha, (self.num_states, 1))
    self.rewards_vector = np.reshape(self.rewards, (1, self.num_actions))

  def computeSimplex(self):
    X_vector = cp.Variable((self.num_actions, 1))
    constraints = [(self.A_matrix * X_vector) == self.alpha_vector, X_vector >= 0]
    objective = cp.Maximize(self.rewards_vector * X_vector)
    problem = cp.Problem(objective, constraints)
    problem.solve()

    return problem.status, problem.value, X_vector.value

def createOutput(LPObject):
  status, prob_value, x_values = LPObject.outputs
  actions = LPObject.actions
  states = LPObject.states
  policy_draft = {}
  state_values = {}

  for i in range(len(actions)):
    state = actions[i][0]
    action = actions[i][1]

    if state not in state_values:
      state_values[state] = x_values[i]
      
      if action == 'TERMINAL':
        policy_draft[state] = 'NOOP'
      else:
        policy_draft[state] = action

    elif state_values[state] < x_values[i]:
      state_values[state] = x_values[i]

      if action == 'TERMINAL':
        policy_draft[state] = 'NOOP'
      else:
        policy_draft[state] = action

  policy = [ [key, value] for key, value in policy_draft.items() ]

  output = {
      "a": np.round(LPObject.A_matrix, 3).tolist(),
      "r": LPObject.rewards_vector.tolist(),
      "alpha": LPObject.alpha_vector.tolist(),
      "x": np.round(x_values, 3).tolist(),
      "policy": policy,
      "objective": np.round(prob_value, 3)
  }

  return output

def storeOutput(output):
  try:
    current_directory = os.getcwd()
    
    final_directory = os.path.join(current_directory, 'outputs')
    if not os.path.exists(final_directory):
      os.makedirs("outputs")
    with open(current_directory + "/outputs/output.json", "w") as out_file:
      json.dump(output, out_file)
  except:
    pass
  

if __name__ == "__main__":
  MDPObject = MDP(-5)
  LPObject = LP(MDPObject)
  output = createOutput(LPObject)
  storeOutput(output)

