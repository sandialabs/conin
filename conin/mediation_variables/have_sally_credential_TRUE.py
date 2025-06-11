#have_sally_credential

def update_fun(k, r, k_past, r_past):
    '''
    r^t = (m_1^t, m_2^t)
    m1^t = (k == ('CA', ('HI', 'usr/query'))) or r_past[0]  # tracks if the credential state has occurred yet
    forbidden_emissions = (k == ('EX', ('V', 'access/sally'))) or (k == ('EX', ('DS', 'syslog/nano'))) or (k == ('DI', ('DS', 'syslog/ls')))
    m2 = (m1 or not forbidden_emissions) and r_past[1]
    '''
    m1 = (k == ('CA', ('HI', 'usr/query'))) or r_past[0]  # tracks if the credential state has occurred yet
    forbidden_emissions = (k == ('EX', ('V', 'access/sally'))) or (k == ('EX', ('DS', 'syslog/nano'))) or (k == ('DI', ('DS', 'syslog/ls')))
    m2 = (m1 or not forbidden_emissions)

    return int(r == (m1, ) and m2)

def init_fun(k, r):
    '''
    Initial "prob" of r = (m1, m2) from k. Is just indicator
    '''
    m1 = k == ('CA', ('HI', 'usr/query'))
    m2 = not ((k == ('EX', ('V', 'access/sally'))) or (k == ('EX', ('DS', 'syslog/nano'))) or (k == ('DI', ('DS', 'syslog/ls'))))  # at first time, can only violate the emission constraint.

    return int(r == (m1, ) and m2)

def cst_fun(k, r, sat):
    '''
    Constraint is a boolean emissions of the final auxiliary state. In this case, is just m1^T: ie. tau_a >= tau_b for all time.
    '''
    return 1

dependency = 'know_sally_exists'

aug_size = 1

forbidden_emissions = [ ('EX', ('V', 'access/sally')), ('EX', ('DS', 'syslog/nano')), ('DI', ('DS', 'syslog/ls'))]

forbidden_transitions = []

knowledge_state =  ('CA', ('HI', 'usr/query'))

name = 'have_sally_credential'

states = [(True, True),(False,True)] #second variables tracks \tau_A > \tau_B at every t.