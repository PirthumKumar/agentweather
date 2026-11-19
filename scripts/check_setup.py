"""Run this FIRST. It tests every key / service and tells you exactly what is missing.
 
    python scripts/check_setup.py
"""
import sys
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
 
import requests
 
from agentweather import config as C
 
OK, BAD = "\033[92m[ OK ]\033[0m", "\033[91m[FAIL]\033[0m"
 
 
def show(ok: bool, name: str, detail: str = ""):
    print(f"{OK if ok else BAD} {name:<22} {detail}")
    return ok
 
 
def main() -> int:
    print(f"City: {C.CITY_NAME} ({C.LAT}, {C.LON})\n")
    results = []
    # 1. Open-Meteo (no key)
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={"latitude": C.LAT, "longitude": C.LON, "current": "temperature_2m"}, timeout=20)
        results.append(show(r.ok, "Open-Meteo", f"temp now {r.json()['current']['temperature_2m']} C" if r.ok else r.text[:80]))
    except Exception as e:
        results.append(show(False, "Open-Meteo", repr(e)))
    # 2. OpenWeather
    if C.OPENWEATHER_API_KEY:
        r = requests.get("https://api.openweathermap.org/data/2.5/weather", params={"lat": C.LAT, "lon": C.LON, "appid": C.OPENWEATHER_API_KEY, "units": "metric"}, timeout=20)
        results.append(show(r.ok, "OpenWeather", f"temp now {r.json()['main']['temp']} C" if r.ok else f"HTTP {r.status_code} (new keys take up to 2 h to activate)"))
    else:
        results.append(show(False, "OpenWeather", "OPENWEATHER_API_KEY missing in .env"))
    # 3. AQICN
    if C.AQICN_TOKEN:
        target = C.AQICN_STATION or f"geo:{C.LAT};{C.LON}"
        r = requests.get(f"https://api.waqi.info/feed/{target}/", params={"token": C.AQICN_TOKEN}, timeout=20).json()
        results.append(show(r.get("status") == "ok", "AQICN", f"AQI {r['data']['aqi']} at {r['data']['city']['name']}" if r.get("status") == "ok" else str(r)[:100]))
    else:
        results.append(show(False, "AQICN", "AQICN_TOKEN missing in .env"))
    # 4. Groq
    if C.GROQ_API_KEY:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers={"Authorization": f"Bearer {C.GROQ_API_KEY}"},
                          json={"model": C.GROQ_MODEL, "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 5}, timeout=30)
        results.append(show(r.ok, "Groq LLM", C.GROQ_MODEL if r.ok else f"HTTP {r.status_code}: {r.text[:100]}"))
    else:
        print(f"{BAD} {'Groq LLM':<22} GROQ_API_KEY missing (agents will use fallback templates)")
    # 5. Hopsworks
    if C.HOPSWORKS_API_KEY:
        try:
            import hopsworks
            kw = {"api_key_value": C.HOPSWORKS_API_KEY}
            if C.HOPSWORKS_PROJECT:
                kw["project"] = C.HOPSWORKS_PROJECT
            p = hopsworks.login(**kw)
            p.get_feature_store()
            results.append(show(True, "Hopsworks", f"project '{p.name}'"))
        except Exception as e:
            results.append(show(False, "Hopsworks", repr(e)[:150]))
    else:
        print(f"{BAD} {'Hopsworks':<22} HOPSWORKS_API_KEY missing (local parquet store will be used)")
    # 6. Python libs
    for lib in ("pandas", "sklearn", "statsmodels", "torch", "shap", "lime", "langgraph", "streamlit"):
        try:
            __import__(lib)
            show(True, f"import {lib}")
        except Exception as e:
            show(False, f"import {lib}", "pip install -r requirements-pipeline.txt")
    try:
        import torch
        print(f"      PyTorch device: {'GPU - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (fine for this project)'}")
    except Exception:
        pass
    return 0 if all(results) else 1
 
 
if __name__ == "__main__":
    sys.exit(main())
