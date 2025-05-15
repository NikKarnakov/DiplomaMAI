# thermo_parser.py
import re
import math

R = 8.314462618
class Species:
    def __init__(self, name, molecular_weight, composition, enthalpy_of_formation, entropy, extra_param, nasa_coeffs):
        self.name = name
        self.M = molecular_weight
        self.composition = composition
        self.Hf0 = enthalpy_of_formation
        self.S0 = entropy
        self.extra = extra_param
        self.nasa = nasa_coeffs
    def cp(self, T):

        for entry in self.nasa:
            if entry['Tmin'] <= T <= entry['Tmax']:
                a = entry['coeffs']
                # Cp/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4
                cp_R = a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4
                return cp_R * R

        entry = self.nasa[-1]
        a = entry['coeffs']
        cp_R = a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4
        return cp_R * R

    def enthalpy(self, T):
        # возвращает H (Дж/моль) при T, включая теплоту образования
        for entry in self.nasa:
            if entry['Tmin'] <= T <= entry['Tmax']:
                a = entry['coeffs']
                # H/RT = a1 + a2*T/2 + a3*T^2/3 + a4*T^3/4 + a5*T^4/5 + a6/T
                H_RT = a[0] + a[1]*T/2 + a[2]*T**2/3 + a[3]*T**3/4 + a[4]*T**4/5 + a[5]/T
                H = H_RT * R * T
                return H
        entry = self.nasa[-1]
        a = entry['coeffs']
        H_RT = a[0] + a[1]*T/2 + a[2]*T**2/3 + a[3]*T**3/4 + a[4]*T**4/5 + a[5]/T
        return H_RT * R * T

    def entropy(self, T):
        # возвращает S (Дж/(моль·К)) при температуре T
        for entry in self.nasa:
            if entry['Tmin'] <= T <= entry['Tmax']:
                a = entry['coeffs']
                # S/R = a1*ln(T) + a2*T + a3*T^2/2 + a4*T^3/3 + a5*T^4/4 + a6
                S_R = a[0]*math.log(T) + a[1]*T + a[2]*T**2/2 + a[3]*T**3/3 + a[4]*T**4/4 + a[6]
                return S_R * R
        entry = self.nasa[-1]
        a = entry['coeffs']
        S_R = a[0]*math.log(T) + a[1]*T + a[2]*T**2/2 + a[3]*T**3/3 + a[4]*T**4/4 + a[6]
        return S_R * R


def parse_trm(file_path: str):
    """
    Читает .trm или «чистый» .txt, возвращает список объектов Species.
    Поддерживает два формата заголовка:
    - оригинальный .trm: имя вида в одинарных кавычках, далее масса
    - .txt: имя и масса — первые два токена строки
    """
    species_list = []
    with open(file_path, 'r', encoding='utf8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        raise ValueError(f"Файл {file_path} пуст или не содержит данных.")
    header = lines[0].split()
    num_species = int(header[0])
    num_elements = int(header[1])
    idx = 1
    for _ in range(num_species):
        line = lines[idx].strip()
        idx += 1
        if "'" in line:
            start = line.find("'")
            end = line.rfind("'")
            name = line[start+1:end]
            head = line[:start].split()
            tail = line[end+1:].split()
            tokens = head + tail
            M = float(tokens[-1])
        else:
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Неправильный формат заголовка вида: '{line}'")
            name = parts[0]
            M = float(parts[1])
        comp_parts = lines[idx].split()
        idx += 1
        if len(comp_parts) < num_elements:
            raise ValueError(f"Ожидалось {num_elements} чисел в составе, найдено {len(comp_parts)}")
        composition = [int(x) for x in comp_parts[:num_elements]]
        thermo_parts = lines[idx].split()
        idx += 1
        if len(thermo_parts) < 2:
            raise ValueError(f"Неполная строка термоданных: '{lines[idx-1]}'")
        Hf0 = float(thermo_parts[0])
        S0 = float(thermo_parts[1])
        extra = float(thermo_parts[2]) if len(thermo_parts) > 2 else None
        range_count = int(lines[idx].strip())
        idx += 1
        nasa_coeffs = []
        for __ in range(range_count):
            parts = lines[idx].split()
            idx += 1
            if len(parts) < 9:
                raise ValueError(f"Ожидалось 9 чисел в коэффициентах, найдено {len(parts)}")
            coeffs = [float(x) for x in parts[:7]]
            Tmin = float(parts[7])
            Tmax = float(parts[8])
            nasa_coeffs.append({
                'coeffs': coeffs,
                'Tmin': Tmin,
                'Tmax': Tmax
            })
        species = Species(name, M, composition, Hf0, S0, extra, nasa_coeffs)
        species_list.append(species)
    return species_list
