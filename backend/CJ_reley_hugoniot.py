# backend/CJ_reley_hugoniot.py

import os
import numpy as np
import cantera as ct
from scipy.optimize import fsolve

from .postshock import CJspeed, PostShock_eq, hug_eq

def solve_cj_reley_hugoniot(
    P1: float,
    T1: float,
    q: str,
    v_steps: int = 100,
    v_min_factor: float = 0.3,
    v_max_factor: float = 1.7
) -> dict:
    """
    Решение задачи 2: построение прямой Релея + адиабаты Гюгонио с точкой Чепмена–Жуге.
    Теперь в словаре 'products' попадают все виды, у которых ν > 1e-8, независимо от того,
    входили они в исходную смесь или нет.
    """
    mech_filename = "mevel2018.yaml"
    base_dir = os.path.dirname(__file__)
    mech_path = os.path.join(base_dir, mech_filename)
    if not os.path.isfile(mech_path):
        raise FileNotFoundError(f"Механизм '{mech_filename}' не найден в '{base_dir}'")

    # ----------------------------------------------------------------
    # 1) Инициализируем начальное состояние (до удара)
    # ----------------------------------------------------------------
    gas1 = ct.Solution(mech_path)
    gas1.TPX = T1, P1, q
    h1 = gas1.enthalpy_mass
    r1 = gas1.density
    v1 = 1.0 / r1

    # ----------------------------------------------------------------
    # 2) Вычисляем CJ-скорость и состояние CJ через PostShock_eq
    # ----------------------------------------------------------------
    D_cj = CJspeed(P1, T1, q, mech_path)
    gas_cj = PostShock_eq(D_cj, P1, T1, q, mech_path)
    vcj = 1.0 / gas_cj.density
    Pcj = gas_cj.P / ct.one_atm      # переводим в атм
    Tcj = gas_cj.T

    # ----------------------------------------------------------------
    # 3) Строим диапазон v для прямой Релея
    # ----------------------------------------------------------------
    v_min = v_min_factor * vcj
    v_max = v_max_factor * vcj
    vinc = 0.01 * vcj
    n_r = int(np.ceil((v_max - v_min) / vinc)) + 1
    vR = np.linspace(v_min, v_max, n_r)
    PR = [
        (P1 - r1**2 * D_cj**2 * (v2 - v1)) / ct.one_atm
        for v2 in vR
    ]

    # ----------------------------------------------------------------
    # 4) Строим равновесную Hugoniot-кривую, двигаясь вниз от Ta
    # ----------------------------------------------------------------
    gas_eq = ct.Solution(mech_path)
    gas_eq.TPX = T1, P1, q
    gas_eq.equilibrate('UV')
    Ta = gas_eq.T
    va = 1.0 / gas_eq.density

    vH = [va]
    PH = [gas_eq.P / ct.one_atm]

    vb = va
    idx = 0
    # Двигаемся вниз по v, пока vb > v_min
    while vb > v_min:
        idx += 1
        vb = va - idx * 0.01
        Tf = fsolve(hug_eq, Ta, args=(vb, h1, P1, v1, gas_eq))
        gas_eq.TD = Tf, 1.0 / vb
        PH.append(gas_eq.P / ct.one_atm)
        vH.append(vb)
    # ----------------------------------------------------------------
    # 5) Получаем состав продуктов в CJ-точке
    # ----------------------------------------------------------------
    gas_products = ct.Solution(mech_path)
    gas_products.TP = Tcj, Pcj * ct.one_atm
    gas_products.equilibrate('TP')

    species_names = gas_products.species_names     # список всех видов механизма
    nu_array      = gas_products.X                 # массив мольных долей
    pi_array      = nu_array * gas_products.P      # частичные давления (Па)

    # Формируем итоговый список продуктов: все виды с nu > порог
    products = []
    small_threshold = 1e-8
    for name, nu, p in zip(species_names, nu_array, pi_array):
        if nu > small_threshold:
            products.append({
                "name": name,
                "nu": float(nu),
                "p": float(p)
            })

    # ----------------------------------------------------------------
    # 6) Собираем всё в словарь и возвращаем
    # ----------------------------------------------------------------
    return {
        "cj": {
            "speed":  D_cj,
            "vcj":    vcj,
            "Pcj":    Pcj,
            "Tcj":    Tcj,
        },
        "vR":       vR.tolist(),
        "PR":       PR,
        "vH":       vH,
        "PH":       PH,
        "products": products
    }
