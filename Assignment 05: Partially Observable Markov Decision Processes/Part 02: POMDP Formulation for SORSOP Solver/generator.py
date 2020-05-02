# Initializing basic parameters for POMDP
actions = ['stay', 'up', 'down', 'left', 'right']
cells = []
states = []
transitions = []

# For determining the probability for the given transition
def find_prob(action, i, j):
    return 1.0

# For generating all the cells of the matrix
for i in range(0,3):
    for j in range(0, 3):
        cells.append("(" + str(i) + "," + str(j) + ")")

# For generating all the states of the POMDP
for i in cells:
    for j in cells:
        for call in range(0,2):
            states.append((i, j, call))

# # For generating different types of  transistions
for i in states:
    for j in states:
        for action in actions:
            transitions.append("T: " + action + " : " + str(i) + " : " + str(j) + " " + str(find_prob(action, i, j)))

# For generating the initial belief state
for state in states:
    if state[1] == '(1,1)':
        if state[0] == '(0,0)' or state[0] == '(0,2)' or state[0] == '(2,2)' or state[0] =='(2,0)':
            print("0.125", end = " ")
        else:
            print("0.0", end = " ")
    else:
        print("0.0", end = " ")