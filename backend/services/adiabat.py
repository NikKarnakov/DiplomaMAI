# adiabat.py

import numpy as np
from scipy.optimize import brentq
from .thermo_parser import R
from .equilibrium import solve_equilibrium_composition

def calc_adiabat(species_list, element_balance, CJ_res, init_comp, P1, T1, points=100):
    T_CJ = CJ_res['T_CJ']
    P_CJ = CJ_res['P_CJ']
    V_CJ = CJ_res['V_CJ']
    n_CJ = CJ_res['n_CJ']

    # Энтропия CJ
    S_CJ = sum(n_CJ[i] * species_list[i].entropy(T_CJ)
               for i in range(len(species_list)))

    # Если P_CJ или P1 некорректны, падаем на равномерный линейный сет:
    if not (np.isfinite(P_CJ) and np.isfinite(P1) and P_CJ > 0 and P1 > 0):
        P_vals = np.linspace(max(P_CJ, 1e-6), max(P1, 1e-6), points)
    else:
        P_vals = np.logspace(np.log10(P_CJ), np.log10(P1), points)

    # Отфильтруем на всякий случай нулевые/отрицательные/нечисла
    P_vals = P_vals[np.isfinite(P_vals) & (P_vals > 0)]
    if len(P_vals) == 0:
        raise RuntimeError("Невозможно построить адиабату: все P_vals неверны")

    V_out, T_out, P_out = [], [], []

    for P in P_vals:
        def F(T_try):
            V_try = sum(init_comp.values()) * R * T_try / P
            n_eq = solve_equilibrium_composition(
                species_list, element_balance, T_try, V_try
            )
            S = sum(n_eq[i] * species_list[i].entropy(T_try)
                    for i in range(len(species_list)))
            return S - S_CJ

        # границы
        T_low, T_high = T1, T_CJ
        f_low, f_high = F(T_low), F(T_high)

        if f_low * f_high > 0:
            T_sol = T_low if abs(f_low) < abs(f_high) else T_high
        else:
            T_sol = brentq(F, T_low, T_high)

        V_sol = sum(init_comp.values()) * R * T_sol / P

        # сохраняем только конечные валидные точки
        if np.isfinite(V_sol) and np.isfinite(T_sol):
            V_out.append(V_sol)
            P_out.append(P)
            T_out.append(T_sol)

    return np.array(V_out), np.array(P_out), np.array(T_out)
