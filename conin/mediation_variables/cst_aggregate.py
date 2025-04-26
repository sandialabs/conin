from munch import Munch

def create_updatefun(zip_list):
    def update_fun_agg(r,k,r_past):
        val = 1
        for cst, ix in zip_list:
            val *= cst.update_fun(tuple(r[ix[0]:ix[1]]),k,tuple(r_past[ix[0]:ix[1]]))
        return val
    return update_fun_agg

def create_initfun(zip_list):
    def init_fun_agg(k,r):
        val = 1
        for cst,ix in zip_list:
            val *= cst.init_fun(k,tuple(r[ix[0]:ix[1]]))
        return val
    return init_fun_agg

def create_cstfun(zip_list):
    def cst_fun_agg(r,sat):
        val = 1
        it = 0
        for cst,ix in zip_list:
            val*= cst.cst_fun(tuple(r[ix[0]:ix[1]]),sat[it])
            it += 1
        return val
    return cst_fun_agg

def cst_aggregate(cst_list):
    l_ix = 0
    r_ix = 0
    ix_list = []
    name_list = []
    for cst in cst_list:
        r_ix = l_ix + cst.aux_size
        ix_list.append((l_ix,r_ix)) #tuple of indices of the aux stats that correspond to each state
        l_ix = r_ix
        name_list.append(cst.name)
    zip_list = list(zip(cst_list,ix_list))

    cst_combined = Munch(name = name_list, aux_size = r_ix, update_fun = create_updatefun(zip_list), \
                         init_fun = create_initfun(zip_list), cst_fun = create_cstfun(zip_list))
    
    return cst_combined