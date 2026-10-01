"""X1: Open-Meteo (weather, no API key)."""

from __future__ import annotations

import json

from kin.contracts import FactRequest

HOST = "api.open-meteo.com"


def url(req: FactRequest) -> str:
    return (
        f"https://{HOST}/v1/forecast?latitude={req.lat}&longitude={req.lon}"
        f"&hourly=precipitation_probability,temperature_2m&forecast_hours={max(1, req.hours)}&timezone=auto"
    )


def parse(text: str, req: FactRequest) -> dict:
    h = json.loads(text)["hourly"]
    hourly = [
        {"t": t, "rain_pct": int(p or 0), "temp_c": round(float(c), 1)}
        for t, p, c in zip(h["time"], h["precipitation_probability"], h["temperature_2m"], strict=False)
    ][:48]
    return {"lat": req.lat, "lon": req.lon, "hours": req.hours, "hourly": hourly}
