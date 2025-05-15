# trm_to_txt.py

import sys
from .thermo_parser import parse_trm


def convert_trm_to_txt(trm_path, txt_path):

    species = parse_trm(trm_path)

    if not species:
        raise RuntimeError("В .trm-файле не нашлось ни одного вещества.")

    num_elems = len(species[0].composition)
    num_sp = len(species)

    with open(txt_path, 'w', encoding='utf8') as f:
        f.write(f"{num_sp} {num_elems}\n")

        for sp in species:
            f.write(f"{sp.name} {sp.M:.6e}\n")
            f.write(" ".join(str(x) for x in sp.composition) + "\n")
            extra = sp.extra if sp.extra is not None else 0.0
            f.write(f"{sp.Hf0:.6e} {sp.S0:.6e} {extra:.6e}\n")
            f.write(f"{len(sp.nasa)}\n")
            for ent in sp.nasa:
                coeffs = " ".join(f"{c:.10e}" for c in ent['coeffs'])
                # Tmin, Tmax
                f.write(f"{coeffs} {ent['Tmin']:.10e} {ent['Tmax']:.10e}\n")
    print(f"Успешно сконвертировано:\n  {trm_path} → {txt_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Использование: python trm_to_txt.py input.trm output.txt")
        sys.exit(1)
    trm_file = sys.argv[1]
    txt_file = sys.argv[2]
    convert_trm_to_txt(trm_file, txt_file)
