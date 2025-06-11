#have_data_on_he

def update_fun(k, r, k_past, r_past):
    '''
    r^t = (m_1^t, m_2^t)
    m1^t = (k == ('COL', ('HE', 'img/post'))) or r_past[0]  # tracks if the data state has occurred yet
    forbidden_transitions = (k_past[0] == 'COL' and k[0] == 'EXF')
    forbidden_emissions = (k == ('EXF', ('HE', 'img/query')))
    m2 = (m1 or (not (forbidden_transitions and forbidden_emissions))) and r_past[1]
    '''
    m1 = (k == ('COL', ('HE', 'img/post'))) or r_past[0]  # tracks if the data state has occurred yet
    forbidden_transitions = (k_past[0] == 'COL' and k[0] == 'EXF')
    forbidden_emissions = (k == ('EXF', ('HE', 'img/query')))
    m2 = (m1 or (not (forbidden_transitions and forbidden_emissions))) and r_past[1]

    return int(r == (m1, m2))

def init_fun(k, r):
    '''
    Initial "prob" of r = (m1, m2) from k. Is just indicator
    '''
    m1 = k == ('COL', ('HE', 'img/post'))
    m2 = not (k == ('EXF', ('HE', 'img/query')))  # at first time, can only violate the emission constraint.

    return int(r == (m1, m2))

def cst_fun(k, r, sat):
    '''
    Constraint is a boolean emissions of the final auxiliary state. In this case, is just m1^T: ie. tau_a >= tau_b for all time.
    '''
    return int(r[1] == sat)

dependency = 'have_data_on_hi'

forbidden_emissions = [ ('EXF', ('HE', 'img/query'))]

forbidden_transitions = [('COL','EXF'), ('WAIT_COL','EXF')]

knowledge_state = ('COL', ('HE', 'img/post'))

aug_size = 2

name = 'have_data_on_he'

states = [(True, True),(False,True)] #second variables tracks \tau_A > \tau_B at every t.