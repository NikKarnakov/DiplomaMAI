# backend/main.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .reley_hugoniot import solve_reley_hugoniot
from .CJ_reley_hugoniot import solve_cj_reley_hugoniot
from .CJ_Speed_Preassure_Temperature import solve_cj_speed_pressure_temperature

app = FastAPI()


# ==================== Задача 1 ====================
@app.post("/api/reley_hugoniot")
async def api_reley_hugoniot(payload: dict):
    try:
        P1 = float(payload["P1"])
        T1 = float(payload["T1"])
        q = str(payload["q"])
        U1 = float(payload["U1"])
        n_steps = int(payload.get("n_steps", 50))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Неверный формат входных данных: {e}")
    try:
        result = solve_reley_hugoniot(P1, T1, q, U1, n_steps)
        return JSONResponse(content=result)
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=500, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== Задача 2 ====================
@app.post("/api/cj_reley_hugoniot")
async def api_cj_reley_hugoniot(payload: dict):
    try:
        P1 = float(payload["P1"])
        T1 = float(payload["T1"])
        q = str(payload["q"])
        v_steps = int(payload.get("v_steps", 100))
        v_min_factor = float(payload.get("v_min_factor", 0.3))
        v_max_factor = float(payload.get("v_max_factor", 1.7))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Неверный формат входных данных: {e}")
    try:
        result = solve_cj_reley_hugoniot(P1, T1, q, v_steps, v_min_factor, v_max_factor)
        return JSONResponse(content=result)
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=500, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== Задача 3 ====================
@app.post("/api/cj_speed_vs_u")
async def api_cj_speed_vs_u(payload: dict):
    """
    Ожидает JSON:
      {
        "P1": float,
        "T1": float,
        "q_template": str,   # например "H2:1 O2:{u}"
        "u_values": [числа]
      }
    Возвращает JSON:
      {
        "u": [...],
        "speeds": [...],
        "pressures": [...],
        "temperatures": [...]
      }
    """
    try:
        P1 = float(payload["P1"])
        T1 = float(payload["T1"])
        q_template = str(payload["q_template"])
        u_values = payload["u_values"]
        if not isinstance(u_values, list):
            raise ValueError("u_values должен быть списком чисел")
        u_values = [float(u) for u in u_values]
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Неверный формат входных данных: {e}")
    try:
        result = solve_cj_speed_pressure_temperature(P1, T1, q_template, u_values)
        return JSONResponse(content=result)
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=500, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== CORS ====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Статика ====================
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
