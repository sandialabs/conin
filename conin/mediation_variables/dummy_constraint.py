#Ddummy constraint for debugging

def update_fun(k, r, k_past, r_past):
    return 1

def init_fun(k, r):

    return 1

def cst_fun(k, r, sat):
    '''
    Constraint is a boolean emissions of the final auxiliary state. In this case, is just m1^T: ie. tau_a >= tau_b for all time.
    '''
    return 1


forbidden_emissions = [ ]

forbidden_transitions = []

knowledge_state = 'dummy'

aug_size = 1

name = 'dummy'
