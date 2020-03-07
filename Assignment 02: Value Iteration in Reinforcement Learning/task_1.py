nStates = 60
nActions = 3

arr = [0.5, 1, 2]
X = 29 #Team Number

states = []
actions = ['SHOOT', 'DODGE', 'RECHARGE']
transition_model = {}
gamma = 0.99
delta = 0.001
penalty = -10/arr[X % 3]
reward = {}

for i in range(0, 5):
    for j in range(0, 3):
        for k in range(0, 4):
            states.append((i, j, k))

for state in states:
    h, a, s = state
    reward[state] = penalty if h != 0 else 10



for state in states:
    """For each of the states in the Markov Decision Process, we assign 
    the transition pobabilities to the next possible states according
    to the rules given in the question
    """
    transition = [[], [], []]
    h, a, s = state
    if a > 0:
        if s - 1 >= 0 and h - 1 >= 0:
            transition[0].append((0.5, (h-1, a-1, s-1)))
        if s - 1 >= 0:
            transition[0].append((0.5, (h, a-1, s-1)))
    if s == 2:
        if a + 1 <= 2:
            transition[1].append((0.64, (h, a+1, s-1)))
            transition[1].append((0.16, (h, a+1, s-2)))
        transition[1].append((0.16, (h, a, s-1)))
        transition[1].append((0.04, (h, a, s-2)))
    if s == 1:
        if a + 1 <= 2:
            transition[1].append((0.8, (h, a+1, s-1)))
        transition[1].append((0.2, (h, a, s-1)))
    if s + 1 <= 3:
        transition[2].append((0.8, (h, a, s+1)))
    transition[2].append((0.2, (h, a, s)))

    transition_model[state] = transition



def print_transition_model():
    """For each state of the Markov Decision Process environment, the
    transition model is of the form ACTION: (probability, state), <(probability, state), ...>
    """
    for key, val in transition_model.items():
        print("State: " + str(key))
        print("SHOOT: " + str(val[0]))
        print("DODGE: " + str(val[1]))
        print("RECHARGE: " + str(val[2]) + "\n")



