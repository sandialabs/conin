#learn_where_data_stored

def update_fun(k, r, k_past, r_past):
    '''
    r^t = (m_1^t, m_2^t)
    m1^t = (k == ('DI', ('DS', 'syslog/ls'))) or r_past[0]  # tracks if the data storage state has occurred yet
    forbidden_transitions = (k_past[0] == 'DI' and k[0] == 'COL')
    forbidden_emissions = (k == ('COL', ('DS', 'syslog/nano')))
    m2 = (m1 or (not (forbidden_transitions and forbidden_emissions))) and r_past[1]
    '''
    m1 = (k == ('DI', ('DS', 'syslog/ls'))) or r_past[0]  # tracks if the data storage state has occurred yet
    forbidden_transitions = ((k_past[0] == 'DI' or k_past[0] == 'WAIT_DI') and (k[0] == 'COL' or k[0] == 'WAIT_COL'))
    forbidden_emissions = (k == ('COL', ('DS', 'syslog/nano')))
    m2 = (m1 or (not (forbidden_transitions and forbidden_emissions))) 

    return int(r == (m1, ) and m2)

def init_fun(k, r):
    '''
    Initial "prob" of r = (m1, m2) from k. Is just indicator
    '''
    m1 = k == ('DI', ('DS', 'syslog/ls'))
    m2 = not (k == ('COL', ('DS', 'syslog/nano')))  # at first time, can only violate the emission constraint.

    return int(r == (m1, ) and m2)

def cst_fun(k, r, sat):
    '''
    Constraint is a boolean emissions of the final auxiliary state. In this case, is just m1^T: ie. tau_a >= tau_b for all time.
    '''
    return 1

dependency = 'have_sally_credential'

forbidden_emissions = [ ('COL', ('DS', 'syslog/nano'))]

forbidden_transitions = [('DI','COL'), ('WAIT_DI','COL')]

knowledge_state = ('DI', ('DS', 'syslog/ls'))

aug_size = 1

name= 'learn_where_data_stored'

states = [(True, True),(False,True)] #second variables tracks \tau_A > \tau_B at every t.