#know_sally_exists

def update_fun(k,r ,k_past, r_past):
    '''
    r^t = (m_1^t, m_2^t)
    m1^t = tau^t. the hitting time of (DI, (HI, usr/query))
    m2^t = [1- (1 - tau^t_a) AND c)] AND m2^{t-1} = [tau^t_a or (1 - c)] AND m2^{t-1} #tracks if the arrival time of a is before c
    '''
    m1 = (k == ('DI',('HI','usr/query'))) or r_past[0] #tracks if knowledge state has occured yet
    forbidden_transitions = ((k_past[0] == 'EX' or k_past[0] == 'WAIT_EX') and (k[0] == 'CA' or k[0] == 'WAIT_CA')) or ((k_past[0] == 'DI' or k_past[0] == 'WAIT_DI') and (k[0] == 'CA' or k[0] == 'WAIT_CA'))
    forbidden_emissions = (k == ('CA',('HI','usr/query')))
    m2 = (m1 or (not (forbidden_transitions and forbidden_emissions))) and r_past[1] 
          
    #at the time of first hit, all forbidden transitions/emissions are impossible. current logical formulation ok.

    return int(r == (m1,m2))

def init_fun(k, r):
    '''
    initial "prob" of r = (m1,m2) from k. is just indicator
    '''
    m1 = k == ('DI',('HI','usr/query'))
    m2 = not (k == ('CA',('HI','usr/query'))) #at first time, can only violate the emission constraint.


    return int(r == (m1,m2))
    
def cst_fun(k,r, sat):
    '''
    Constraint is a boolean emissions of the final auxillary state. In this case, is just m1^T: ie. tau_a >= tau_b for all time.
    '''
    return int(r[1] == sat) 

aug_size = 2

name = 'know_sally_exists'

forbidden_emissions = [ ('CA',('HI','usr/query'))]

forbidden_transitions = [('EX','CA'),('WAIT_EX','CA'),('DI','CA'),('WAIT_DI','CA')]

knowledge_state = ('DI',('HI','usr/query'))
#states = [(True, True),(False,True)] #second variables tracks \tau_A > \tau_B at every t.