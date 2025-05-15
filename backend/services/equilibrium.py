# equilibrium.py

import math
import numpy as np
from scipy.optimize import fsolve
from .thermo_parser import R

def solve_equilibrium_energy(species_list, element_balance,
                             U0, V0, P_ref=101325.0,
                             tol=1e-8):
    elems = [e for e in element_balance if e != '_idx']
    N_el = len(elems)
    N_sp = len(species_list)
    x0 = np.array([2000.0] + [1.0]*N_sp)

    def eq(vars):
        T = vars[0]
        n = vars[1:]
        res = []
        for e in elems:
            res.append(sum(n[i] * species_list[i].composition.get(e,0)
                           for i in range(N_sp))
                       - element_balance[e])
        U_calc = sum(n[i] * (species_list[i].enthalpy(T) - R*T)
                     for i in range(N_sp))
        res.append(U_calc - U0)
        n_tot = n.sum()
        P  = n_tot * R * T / V0
        mus = []
        for i, sp in enumerate(species_list):
            H = sp.enthalpy(T)
            S = sp.entropy(T)
            pj = max(n[i]/n_tot * P, 1e-20)
            mu = (H - T*S) + R*T*math.log(pj / P_ref)
            mus.append(mu)
        mu_ref = mus[-1]
        R_eq = N_sp - N_el
        for i in range(R_eq):
            res.append(mus[i] - mu_ref)
        return np.array(res)

    def jac(vars):
        """Численный якобиан ∂eq_i/∂vars_j методом центральных разностей."""
        f0 = eq(vars)
        J = np.zeros((len(f0), len(vars)))
        h = 1e-6
        for j in range(len(vars)):
            dv = np.zeros_like(vars)
            dv[j] = h
            f1 = eq(vars + dv)
            f2 = eq(vars - dv)
            J[:,j] = (f1 - f2) / (2*h)
        return J

    sol = fsolve(eq, x0, fprime=jac, xtol=tol, maxfev=2000)
    T_sol = sol[0]
    n_sol = sol[1:]
    return T_sol, n_sol


def solve_equilibrium_composition(species_list, element_balance,
                                  T, V0, P_ref=101325.0, tol=1e-8):
    """
    Решает только n_i при фиксированном T и V0, с численным якобианом.
    """
    elems = [e for e in element_balance if e != '_idx']
    N_el = len(elems)
    N_sp = len(species_list)
    x0 = np.ones(N_sp)

    def eq(n):
        res = []
        # балансы
        for e in elems:
            res.append(sum(n[i] * species_list[i].composition.get(e,0)
                           for i in range(N_sp))
                       - element_balance[e])
        # хим. равновесие
        n_tot = n.sum()
        P  = n_tot * R * T / V0
        mus = []
        for i, sp in enumerate(species_list):
            H = sp.enthalpy(T)
            S = sp.entropy(T)
            pj = max(n[i]/n_tot * P, 1e-20)
            mu = (H - T*S) + R*T*math.log(pj / P_ref)
            mus.append(mu)
        mu_ref = mus[-1]
        R_eq = N_sp - N_el
        for i in range(R_eq):
            res.append(mus[i] - mu_ref)
        return np.array(res)

    def jac(n):
        f0 = eq(n)
        J = np.zeros((len(f0), len(n)))
        h = 1e-6
        for j in range(len(n)):
            dn = np.zeros_like(n)
            dn[j] = h
            f1 = eq(n + dn)
            f2 = eq(n - dn)
            J[:,j] = (f1 - f2) / (2*h)
        return J

    sol = fsolve(eq, x0, fprime=jac, xtol=tol, maxfev=2000)
    return sol
