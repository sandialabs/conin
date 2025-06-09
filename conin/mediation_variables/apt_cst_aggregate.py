from munch import Munch

def create_updatefun(zip_list, cst_ix):
    def update_fun_agg(k,r,k_past,r_past):
        val = 1
        for cst, ix, depend in zip_list:
            val *= cst.update_fun(k,tuple(r[ix[0]:ix[1]]),k_past,tuple(r_past[ix[0]:ix[1]]))
            if depend:
                val *= int(r[cst_ix[depend]] or (not r[cst_ix[cst.name]])) #either the dependcy is satisifed or the current knowledge hasn't been attained.
        return val
    return update_fun_agg

def create_initfun(zip_list):
    def init_fun_agg(k,r):
        val = 1
        for cst,ix, depend in zip_list:
            val *= cst.init_fun(k,tuple(r[ix[0]:ix[1]]))
        return val
    return init_fun_agg

def create_cstfun(zip_list):
    def cst_fun_agg(k,r,sat):
        val = 1
        it = 0
        for cst,ix, depend in zip_list:
            val*= cst.cst_fun(k,tuple(r[ix[0]:ix[1]]),sat[it])
            it += 1
        return val
    return cst_fun_agg

def apt_cst_aggregate(cst_list, debug = False):
    '''
    Assumes that the first Boolean of a constraint tracks whether the knowledge state has been hit.
    '''
    l_ix = 0
    r_ix = 0
    ix_list = []
    name_list = []
    dependency_list = []
    cst_ix = {}
    for cst in cst_list:
        name_list.append(cst.name)
        cst_ix[cst.name] = l_ix
        r_ix = l_ix + cst.aux_size
        ix_list.append((l_ix,r_ix)) #tuple of indices of the aux stats that correspond to each state
        l_ix = r_ix
        if hasattr(cst, 'dependency'):
            dependency_list.append(cst.dependency)
        else:
            dependency_list.append(None)
    zip_list = list(zip(cst_list,ix_list, dependency_list))

    cst_combined = Munch(name = name_list, aux_size = r_ix, update_fun = create_updatefun(zip_list, cst_ix), \
                         init_fun = create_initfun(zip_list), cst_fun = create_cstfun(zip_list))

    if debug:
        return cst_combined, zip_list, cst_ix
    return cst_combined