import matrix

# Initialize a state vector
initial_position = 0 # meters
velocity = 50 # m/s

# Notice the syntax for creating a state column vector ([ [x], [v] ])
# Commas separate these items into rows and brackets into columns
initial_state = matrix.Matrix([ [initial_position], 
                                [velocity] ])

def predict_state_mtx(state, dt):
    tx_matrix = matrix.Matrix([ [1, dt], 
                            [0, 1] ])
    predicted_state = tx_matrix * state
    
    return predicted_state


print('The initial state is: ' + str(initial_state))

# after 2 seconds make a prediction using the new function
state_est1 = predict_state_mtx(initial_state, 2)

print('State after 2 seconds is: ' + str(state_est1))
# after 3 more
state_est2 = predict_state_mtx(state_est1, 3)

print('State after 3 more seconds is: ' + str(state_est2))

# after 3 more
state_est3 = predict_state_mtx(state_est2, 3)

print('Final state after 3 more seconds is: ' + str(state_est3))