# Initializing basic parameters for POMDP
actions = ['stay', 'up', 'down', 'left', 'right']
cells = []
states = []
transitions = []
initial_belief = []

# For determining the probability for the given transition
def find_prob(action, i, j):  
    px = 0.79
    pox = 0.21

    a1x = i[0][0]
    a1y = i[0][1]
    a2x = j[0][0]
    a2y = j[0][1]  

    t1x = i[1][0]
    t1y = i[1][1]
    t2x = j[1][0]
    t2y = j[1][1]

    p = 0.0

    # Target's movement
    if t1x == t2x:
        if abs(t2y - t1y) == 1:
            p = 0.15
    if t1y == t2y:
        if abs(t2x - t1x) == 1:
            p = 0.15
    if t1x == t2x and t1y == t2y:
        if t1x == 1 and t1y == 1:
            p = 0.55
        else:
            p = 0.40

    # Call status
    if i[2] == 0:
        if j[2] == 0:
            p *= 0.60
        elif j[2] == 1:
            p *= 0.40
    elif i[2] == 1:
        if j[2] == 0:
            p *= 0.20
        elif j[2] == 1:
            p *= 0.80 

    # Agent's movement
    if action == 'stay':
        if a1x == a2x and a1y == a2y:
            p *= 1.0
        else:
            p *= 0.0
    elif action == 'up':
        if a1x == a2x:
            if a1x == 2 and a1y == a2y:
                p *= px
            elif a2y - a1y == 1:
                p *= px
            elif a1y - a2y == 1:
                p *= pox
        else:
            p *= 0.0
    elif action == "down":
        if a1x == a2x:
            if a1x == 0 and a1y == a2y:
                p *= px
            elif a1y - a2y == 1:
                p *= px
            elif a2y - a1y == 1:
                p *= pox
        else:
            p *= 0.0
    elif action == "left":
        if a1y == a2y:
            if a1y == 0 and a1x == a2x:
                p *= px
            elif a1x - a2x == 1:
                p *= px
            elif a2x - a1x == 1:
                p *= pox
        else:
            p *= 0.0
    elif action == "right":
        if a1y == a2y:
            if a1x == 2 and a1y == a2y:
                p *= px
            elif a2x - a1x == 1:
                p *= px
            elif a1x - a2x == 1:
                p *= pox
        else:
            p *= 0.0

    return p       

# For generating all the cells of the matrix
for i in range(0,3):
    for j in range(0, 3):
        cells.append((i, j))

# For generating all the states of the POMDP
for i in cells:
    for j in cells:
        for call in range(0,2):
            states.append((i, j, call))

# # For generating different types of  transistions
for i in states:
    for j in states:
        for action in actions:
                print("T: " + action + " : " + str(i) + " : " + str(j) + " " + str(find_prob(action, i, j)))

# For generating the initial belief state
for state in states:
    if state[1] == '(1,1)':
        if state[0] == '(0,0)' or state[0] == '(0,2)' or state[0] == '(2,2)' or state[0] =='(2,0)':
            initial_belief.append("0.125")
        else:
            initial_belief.append("0.0")
    else:
        initial_belief.append("0.0")