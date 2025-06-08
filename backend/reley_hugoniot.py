# reley_hugoniot.py

import os
import cantera as ct
import numpy as np
from .postshock import PostShock_eq
from .thermo import soundspeed_eq

def solve_reley_hugoniot(
    P1: float,
    T1: float,
    q: str,
    U1: float,
    n_steps: int = 50
) -> dict:

    mech_filename = "airNASA9ions.yaml"
    base_dir = os.path.dirname(__file__)
    mech_path = os.path.join(base_dir, mech_filename)

    if not os.path.isfile(mech_path):
        raise FileNotFoundError(f"Механизм '{mech_filename}' не найден в папке '{base_dir}'")

    gas1 = ct.Solution(mech_path)
    gas1.TPX = T1, P1, q
    r1 = gas1.density
    v1 = 1.0 / r1

    # 2) Frozen post-shock при скорости U1
    gas_fr_ps = PostShock_eq(U1, P1, T1, q, mech_path)
    v_ps = 1.0 / gas_fr_ps.density

    # 3) Rayleigh Line: строим vR от maxv = v1 до minv = 0.9 · v_ps
    minv = 0.9 * v_ps
    maxv = 1.00 * v1
    stepv = 0.01 * v1
    n_r = int(np.ceil((maxv - minv) / stepv)) + 1

    vR = np.linspace(maxv, minv, n_r)
    PR = []
    for v2 in vR:
        p_ray = P1 - (r1**2) * (U1**2) * (v2 - v1)
        PR.append(p_ray / ct.one_atm)

    # 4) Frozen Hugoniot: первая точка (v1, P1), потом n_steps точек
    vH = [v1]
    PH = [P1 / ct.one_atm]

    gas1_for_cs = ct.Solution(mech_path)
    gas1_for_cs.TPX = T1, P1, q
    U_min = 1.1 * soundspeed_eq(gas1_for_cs)

    stepU = ((1.1 * U1) - U_min) / float(n_steps)
    for i in range(n_steps):
        U = U_min + stepU * i
        gas_fr = PostShock_eq(U, P1, T1, q, mech_path)
        vH.append(1.0 / gas_fr.density)
        PH.append(gas_fr.P / ct.one_atm)

    return {
        "vR": vR.tolist(),
        "PR": PR,
        "vH": vH,
        "PH": PH
    }
