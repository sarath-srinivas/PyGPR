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

covars = { "sq_exp" : lib.get_krn_se_ard,
           "exp" : lib.get_krn_modexp_ard,
           "sin" : lib.get_krn_sin_ard,
           "symm_exp" : lib.get_symm_covar }

covar_jacs = { "sq_exp" : lib.get_dkrn_se_ard,
               "exp" : lib.get_dkrn_modexp_ard,
               "sin" : lib.get_dkrn_sin_ard,
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

    if krn=='sin':
        assert npar == 2 * dim + 1
    else:
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

def sample_gpr_fun(kxx, mn, seed):
    n = kxx.shape[0]

    if mn == None:
        mn = np.zeros(n)

    y = np.empty(n, dtype=np.float64)

    lib.sample_gp(y, mn, kxx.ravel(), n, seed) 

    return y

def get_covar(x, covar, args):
    n = x.shape[0]

    kxx = np.empty((n,n), dtype=np.float64)

    for i in range(n): 
        for j in range(n): 
            kxx[i,j] = covar(x[i], x[j], args)

    return kxx


#void gpr_interpolate_experts(double *yp, double *var_yp, const double *xp, unsigned long np,
#			     const double *x, const double *y, unsigned long ns, unsigned long nc,
#			     unsigned int dim, double *hp, unsigned long nhp, int is_opt,
#			     void covar(double *krn, const double *x, const double *xp,
#					unsigned long nx, unsigned long nxp, unsigned int dim,
#					const double *p, unsigned int npar, void *dat),
#			     void covar_jac(double *dK, unsigned int k, const double *x,
#					    const double *kxx, unsigned long nx, unsigned int dim,
#					    const double *p, unsigned int np, void *dat),
#			     void *dat, unsigned int gate);

lib.gpr_interpolate_experts.argtypes = [arr, arr, arr, ul, arr, arr, ul, arr, ul, ui, arr, ul, si, covar_fptr, covar_jac_fptr, cvoid, ui]
lib.gpr_interpolate_experts.restype = cvoid 

gates = {'PoE' : 0, 
         'gPoE' : 1,
         'BCM' : 2,
         'rBCM' : 3 }


def interpolate_experts(xt, x, y, xc, nc, hp, is_opt=1, krn='exp', gate='PoE'):
    nt = xt.shape[0]
    ns = x.shape[0]
    dim = x.shape[1]
    nsc = ns / nc
    nhp = hp.shape[0]
    nhps = nhp / nc

    assert ns % nc == 0
    assert nhp % nc == 0

    yt = np.empty(nt, dtype=np.float64)

    var_yt = np.empty(nt * nt, dtype=np.float64)

    lib.gpr_interpolate_experts(yt, var_yt, xt.ravel(), nt, x.ravel(), y, ns, xc.ravel(), nc, dim, hp, nhp, is_opt, covar_fptr(covars[krn]),
            covar_jac_fptr(covar_jacs[krn]), None, gates[gate])

    return [yt, var_yt.reshape(nt,nt)]


def diagnostic(yp, yt, covar, is_diag=False):
    dgn = {}
    var = np.diag(covar)
    n = yp.shape[0]
    err = yp - yt

    dgn['RMSE'] = np.sqrt(np.mean(np.sum(err**2)))

    dgn['SDSUM'] = np.sqrt(np.mean(np.sum(var)))

    dgn['RCHI-SQ'] = (1.0/n) * np.sum((err**2)/var)

    if is_diag == True:
       dgn['LLHD'] =  -0.5 * np.sum(np.log(var)) - 0.5 * np.log(2 * np.pi) - n * dgn['RCHI-SQ']
    else:
       eig = np.linalg.eigvalsh(covar)
       sol = np.linalg.solve(covar, err)
       md = np.dot(err, sol)

       dgn['LLHD'] = -0.5 * np.sum(np.log(eig)) - 0.5 * np.log(2 * np.pi) - md

       dgn['MD'] = (1.0/n) * md

    return dgn




