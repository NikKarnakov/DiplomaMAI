import io
import os
from pathlib import Path

import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from .services.thermo_db import get_species_from_db
from .services.cj_calc import solve_CJ
from .services.adiabat import calc_adiabat
from .services.trm_to_txt import convert_trm_to_txt
from .services.thermo_parser import parse_trm

# Опционально: выводим отладочную инфу
print("=== LOADING backend/main.py ===")
print("  file:", __file__)
print("  cwd :", os.getcwd())

# Пути
BASE_DIR     = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
DB_PATH      = BASE_DIR / "ThermoBaseR.sqlite"

# Регистрируем шрифт для кириллицы (положите DejaVuSans.ttf рядом с main.py)
font_path = BASE_DIR / "DejaVuSans.ttf"
if font_path.exists():
    pdfmetrics.registerFont(TTFont("DejaVuSans", str(font_path)))

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Compute endpoint ----
@app.post("/api/compute")
async def compute(payload: dict):
    comp = payload["composition"]
    T1   = float(payload["T1"])
    P1   = float(payload["P1"])

    species = get_species_from_db(str(DB_PATH), list(comp.keys()))
    elems   = list(species[0].composition.keys())
    balance = {
        e: sum(
            comp[name] *
            next(sp for sp in species if sp.name==name).composition.get(e,0)
            for name in comp
        )
        for e in elems
    }
    balance["_idx"] = None

    cj        = solve_CJ(species, comp, P1, T1, balance)
    V, P_vals, T_vals = calc_adiabat(species, balance, cj, comp, P1, T1, points=100)

    content = {
        "cj": {
            "P_CJ": round(cj["P_CJ"]),
            "T_CJ": round(cj["T_CJ"],1),
            "D":    round(cj["D"],1),
        },
        "curve": {
            "V": V.tolist(),
            "P": P_vals.tolist(),
            "T": T_vals.tolist(),
        },
    }
    return JSONResponse(jsonable_encoder(content))


# ---- Report endpoint ----
@app.post("/api/report")
async def make_report(payload: dict):
    comp = payload["composition"]
    T1   = float(payload["T1"])
    P1   = float(payload["P1"])

    species = get_species_from_db(str(DB_PATH), list(comp.keys()))
    elems   = list(species[0].composition.keys())
    balance = {
        e: sum(
            comp[name] *
            next(sp for sp in species if sp.name==name).composition.get(e,0)
            for name in comp
        )
        for e in elems
    }
    balance["_idx"] = None

    cj        = solve_CJ(species, comp, P1, T1, balance)
    V, P_vals, T_vals = calc_adiabat(species, balance, cj, comp, P1, T1, points=200)

    # Рисуем два графика в одну фигуру
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))
    ax1.plot(V, P_vals); ax1.set_xscale('linear'); ax1.set_yscale('log')
    ax1.set_xlabel('V, м³/кг'); ax1.set_ylabel('P, Па'); ax1.grid(True)
    ax2.plot(V, T_vals); ax2.set_xlabel('V, м³/кг'); ax2.set_ylabel('T, K'); ax2.grid(True)

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='PNG', dpi=150, bbox_inches='tight')
    plt.close(fig)
    img_buf.seek(0)

    # Генерируем PDF
    pdf_buf = io.BytesIO()
    pdf = rl_canvas.Canvas(pdf_buf, pagesize=A4)
    w,h = A4
    pdf.setFont("DejaVuSans" if font_path.exists() else "Helvetica-Bold", 14)
    pdf.drawString(50, h-50, "Отчёт: Детонационная адиабата")
    pdf.setFont("DejaVuSans" if font_path.exists() else "Helvetica", 12)
    y = h-80
    for line in [
        f"T₁ = {T1:.1f} K, P₁ = {int(P1):,}".replace(",", " "),
        f"P_CJ = {int(cj['P_CJ']):,} Па".replace(",", " "),
        f"T_CJ = {cj['T_CJ']:.1f} K",
        f"D = {cj['D']:.1f} м/с",
    ]:
        pdf.drawString(50, y, line)
        y -= 20

    pdf.drawImage(ImageReader(img_buf), 50, y-300, width=500, height=300)
    pdf.showPage()
    pdf.save()
    pdf_buf.seek(0)

    return StreamingResponse(
        pdf_buf,
        media_type="application/pdf",
        headers={"Content-Disposition":"attachment; filename=report.pdf"},
    )


# ---- Upload & convert .trm/.txt endpoint ----
@app.post("/api/upload", include_in_schema=False)
async def upload_trm(file: UploadFile = File(...)):
    tmp_in  = BASE_DIR / file.filename
    tmp_txt = tmp_in.with_suffix(".txt")
    content = await file.read()
    tmp_in.write_bytes(content)

    # если .trm — конвертим
    if tmp_in.suffix.lower() == ".trm":
        convert_trm_to_txt(str(tmp_in), str(tmp_txt))
    else:
        tmp_txt = tmp_in

    # парсим чистый txt и возвращаем JSON состава
    species_list = parse_trm(str(tmp_txt))
    comp = {sp.name: 1.0 for sp in species_list}

    return JSONResponse({"composition": comp})


# ---- Статика (фронтенд) ----
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
