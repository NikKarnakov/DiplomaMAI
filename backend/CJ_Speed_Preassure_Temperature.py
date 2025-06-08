# backend/cj_speed_pressure_temperature.py

import os
import numpy as np
import cantera as ct
from .postshock import CJspeed, PostShock_eq

def solve_cj_speed_pressure_temperature(
    P1: float,
    T1: float,
    q_template: str,
    u_values: list[float]
) -> dict:

    mech_filename = "mevel2017.yaml"
    base_dir = os.path.dirname(__file__)
    mech_path = os.path.join(base_dir, mech_filename)
    if not os.path.isfile(mech_path):
        raise FileNotFoundError(f"Механизм '{mech_filename}' не найден в '{base_dir}'")

    speeds = []
    pressures = []
    temperatures = []

    for u in u_values:
        # подставляем текущее u в шаблон
        if "{u}" not in q_template:
            raise ValueError("В q_template должно быть '{u}', например 'H2:1 O2:{u}'")
        q = q_template.replace("{u}", str(u))

        cj_speed = CJspeed(P1, T1, q, mech_path)
        gas = PostShock_eq(cj_speed, P1, T1, q, mech_path)

        speeds.append(cj_speed)
        pressures.append(gas.P / ct.one_atm)
        temperatures.append(gas.T)

    return {
        "u": u_values,
        "speeds": speeds,
        "pressures": pressures,
        "temperatures": temperatures
    }
