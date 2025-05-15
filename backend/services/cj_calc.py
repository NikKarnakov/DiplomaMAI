    # cj_calc.py

import math
from scipy.optimize import fsolve
from .equilibrium import solve_equilibrium_energy
from .thermo_parser import R

def solve_CJ(species_list, init_comp, P1, T1, element_balance):

    U0 = 0.0
    for name, ni in init_comp.items():
        sp = next(s for s in species_list if s.name == name)
        U0 += ni * (sp.enthalpy(T1) - R * T1)

    V0 = sum(init_comp.values()) * R * T1 / P1

    P2_guess = P1 * 5.0
    T2_guess = T1 * 3.0

    def cj_eq(variables):
        P2, T2 = variables
        # 3.1) объём после скачка
        V2 = sum(init_comp.values()) * R * T2 / P2

        # 3.2) решаем энергетическое равновесие: получаем T_eq, состав n_eq
        T_eq, n_eq = solve_equilibrium_energy(
            species_list, element_balance, U0, V2
        )

        # 3.3) первое уравнение: равенство рассчитанной T_eq и T2
        eq1 = T_eq - T2

        # 3.4) второе уравнение: заглушка, чтобы fsolve получил 2 уравнения
        eq2 = 0.0

        return [eq1, eq2]

    # 4) Ищем решение [P2, T2] с помощью fsolve
    P_CJ, T_CJ = fsolve(cj_eq, [P2_guess, T2_guess])

    # 5) Финальный расчёт состава и параметров
    V_CJ = sum(init_comp.values()) * R * T_CJ / P_CJ
    T_eq, n_CJ = solve_equilibrium_energy(
        species_list, element_balance, U0, V_CJ
    )

    # 6) Расчёт скорости D по формуле D^2 = (P2-P1) / (ρ1*(1-ρ1/ρ2))
    rho1 = P1 / (R * T1)
    rho2 = sum(n_CJ) * R * T_CJ / P_CJ
    discriminant = (P_CJ - P1) / (rho1 * (1.0 - rho1 / rho2))
    # если дискриминант < 0, берём его абсолютное значение, чтобы не было math domain error
    if discriminant < 0:
        discriminant = abs(discriminant)
    D = math.sqrt(discriminant)

    return {
        'P_CJ': P_CJ,
        'T_CJ': T_CJ,
        'D': D,
        'n_CJ': n_CJ,
        'V_CJ': V_CJ
    }
