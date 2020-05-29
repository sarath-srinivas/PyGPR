import numpy as np
import numpy.ctypeslib as nct
import ctypes as ct
import os

arr = nct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
ul = ct.c_ulong
ui = ct.c_uint
si = ct.c_int
cdbl = ct.c_double
cvoid = ct.c_void_p
cdbl_p = ct.POINTER(cdbl) 

lib = nct.load_library(os.environ['LIB'], ".")

covars = { "exp" : lib.get_krn_se_ard,
          "symm_exp" : lib.get_symm_covar }

covar_jacs = { "exp" : lib.get_dkrn_se_ard,
              "symm_exp" : lib.get_symm_covar_jac }

for i in covars:
    covars[i].argtypes = [cdbl_p, cdbl_p, cdbl_p, ul, ul, ui, cdbl_p, ui, cvoid]
    covars[i].restype = cvoid
    covar_jacs[i].argtypes = [cdbl_p, ui, cdbl_p, cdbl_p, ul, ui, cdbl_p, ui, cvoid]
    covar_jacs[i].restype = cvoid

covar_fptr = ct.CFUNCTYPE(cvoid, cdbl_p, cdbl_p, cdbl_p, ul, ul, ui, cdbl_p, ui, cvoid)
covar_jac_fptr = ct.CFUNCTYPE(cvoid, cdbl_p, ui, cdbl_p, cdbl_p, ul, ui, cdbl_p, ui, cvoid)

lib.gpr_interpolate_wrap.argtypes = [arr, arr, ul, arr, arr, ul, ui, arr, ui, arr, si, covar_fptr, covar_jac_fptr, cvoid]
lib.gpr_interpolate_wrap.restype = cvoid

lib.gpr_interpolate.argtypes = [arr, arr, ul, arr, arr, ul, ui, arr, ui, arr, si]
lib.gpr_interpolate.restype = cvoid

def interpolate(xp, x, y, p,  krn='exp', is_opt=1):
    ns = len(y)
    nt = xp.shape[0]

    if len(xp.shape) == 1:
        dim = 1
    else:
        dim = xp.shape[1]

    npar = len(p)

    assert npar == dim + 1

    yp = np.empty(nt, dtype=np.double)
    var = np.empty(nt*nt, dtype=np.double)

    lib.gpr_interpolate_wrap(xp.ravel(), yp, nt, x.ravel(), y, ns, dim, p, npar, var, is_opt, covar_fptr(covars[krn]), 
            covar_jac_fptr(covar_jacs[krn]), None)

#    lib.gpr_interpolate(xp.ravel(), yp, nt, x.ravel(), y, ns, dim, p, npar, var, is_opt)

    return [ yp, np.reshape(var, (nt,nt)), p ]


lib.get_hyper_param_ard.argtypes = [arr, ui, arr, arr, ul, ui, covar_fptr, covar_jac_fptr, cvoid]
lib.get_hyper_param_ard.restype = cvoid 

def get_hyperparam(hp, x, y, krn='exp'):
    
    hp_in = np.copy(hp)
    nhp = len(hp)

    n = len(y)

    if len(x.shape) == 1:
        dim = 1
    else:
        dim = x.shape[1]

    assert nhp == dim + 1

    lib.get_hyperparam_ard(hp_in, nhp, x.ravel(), y, n, dim, covars[krn], covar_jacs[krn], None)  

    return hp_in


lib.sample_gp.argtypes = [arr, arr, arr, ul, si]
lib.sample_gp.restype = cvoid

def sample_gpr_fun(mn, covar, seed):
    n = len(mn)

    y = np.empty(n, dtype=np.float64)

    lib.sample_gp(y, mn, covar.ravel(), n, seed) 

    return y


