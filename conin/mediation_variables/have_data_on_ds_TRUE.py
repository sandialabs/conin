#have_data_on_ds

#Constraint is always true

def update_fun(k, r, k_past, r_past):
    '''
    r^t = (m_1^t, m_2^t)
    m1^t = (k == ('COL', ('DS', 'syslog/nano'))) or r_past[0]  # tracks if the data state has occurred yet
    forbidden_emissions = (k == ('COL', ('HI', 'img/post')))
    m2 = (m1 or not forbidden_emissions) and r_past[1]
    '''
    m1 = (k == ('COL', ('DS', 'syslog/nano'))) or r_past[0]  # tracks if the data state has occurred yet
    forbidden_emissions = (k == ('COL', ('HI', 'img/post')))
    m2 = (m1 or not forbidden_emissions) and r_past[1]

    return int(r == (m1 and m2,))

def init_fun(k, r):
    '''
    Initial "prob" of r = (m1, m2) from k. Is just indicator
    '''
    m1 = k == ('COL', ('DS', 'syslog/nano'))
    m2 = not (k == ('COL', ('HI', 'img/post')))  # at first time, can only violate the emission constraint.

    return int(r == (m1, m2))

def cst_fun(k, r, sat):
    '''
    Constraint is a boolean emissions of the final auxiliary state. In this case, is just m1^T: ie. tau_a >= tau_b for all time.
    '''
    return int(r[1] == sat)

dependency = 'learn_where_data_stored'

aug_size = 1

name = 'have_data_on_ds'

states = [(True, True),(False,True)] #second variables tracks \tau_A > \tau_B at every t.