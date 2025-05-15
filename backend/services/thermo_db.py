# thermo_db.py

import sqlite3
import re
from .thermo_parser import Species

def parse_formula(formula: str) -> dict[str,int]:
    pattern = r'([A-Z][a-z]?)(\d*)'
    comp = {}
    for elem, count in re.findall(pattern, formula):
        comp[elem] = int(count) if count else 1
    return comp

def get_species_from_db(db_path: str, names: list[str]) -> list[Species]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    species_list = []

    for name in names:
        # 1) Достаём запись о веществе
        cur.execute(
            "SELECT SubtanceID, Formula, MolecularMass, DHf298 "
            "FROM dbo_ChemSubtances "
            "WHERE Formula = ?",
            (name,)
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"В базе нет вещества с формулой '{name}'")
        subst_id, formula, M, Hf0 = row

        # 2) Парсим формулу в словарь состава
        composition = parse_formula(formula)

        # 3) Достаём NASA-коэффициенты из PolyConstant
        cur.execute(
            "SELECT Tmin, Tmax, k1, k2, k3, k4, k5, k6, k7 "
            "FROM dbo_PolyConstant "
            "WHERE SubtanceID = ? "
            "ORDER BY Tmin",
            (subst_id,)
        )
        poly_rows = cur.fetchall()
        if not poly_rows:
            raise ValueError(f"Нет полиномов для '{name}' в dbo_PolyConstant")

        nasa_coeffs = []
        for Tmin, Tmax, k1, k2, k3, k4, k5, k6, k7 in poly_rows:
            nasa_coeffs.append({
                "coeffs": [k1, k2, k3, k4, k5, k6, k7],
                "Tmin": Tmin,
                "Tmax": Tmax
            })

        # 4) Собираем объект Species
        sp = Species(
            name=name,
            molecular_weight=M,
            composition=composition,
            enthalpy_of_formation=Hf0 or 0.0,
            entropy=0.0,
            extra_param=None,
            nasa_coeffs=nasa_coeffs
        )
        species_list.append(sp)

    conn.close()
    return species_list
