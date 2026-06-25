import argparse
import asyncio
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from pipeline import (
    AudioLevelMonitor,
    WhisperPipeline,
    default_monitor,
    get_translate_urls,
    list_audio_sources,
)

DEFAULT_CONFIG_PATH = BASE_DIR / "config.json"
PRESETS_PATH = BASE_DIR / "presets.json"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

pipeline = None
audio_monitor = None
event_loop = None
subscribers: list[asyncio.Queue] = []
selected_audio_source: str | None = None


def resolve_audio_source():
    if selected_audio_source:
        return selected_audio_source
    return default_monitor()


def set_audio_source(source: str):
    global selected_audio_source
    selected_audio_source = source


@app.on_event("startup")
async def on_startup():
    global event_loop, selected_audio_source
    event_loop = asyncio.get_running_loop()
    sources = list_audio_sources()
    if sources:
        default = next((s["name"] for s in sources if s["is_default"]), sources[0]["name"])
        selected_audio_source = default
    else:
        selected_audio_source = default_monitor()
    restart_audio_monitor(resolve_audio_source())


def load_config(path=DEFAULT_CONFIG_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_config(config, path=DEFAULT_CONFIG_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_presets():
    if not PRESETS_PATH.exists():
        return {}
    with open(PRESETS_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_presets(presets):
    with open(PRESETS_PATH, "w", encoding="utf-8") as f:
        json.dump(presets, f, indent=2, ensure_ascii=False)
        f.write("\n")


def normalize_config(config):
    config.setdefault("translate_tgt_lang", "jpn_Jpan")
    config.setdefault("chunk_flush_chars", 0)
    config.setdefault("max_feed_entries", 20)
    return config


def apply_config_update(config, body: "ConfigUpdate"):
    config.update(body.model_dump())
    save_config(config)
    return config


def emit_event(event):
    if event_loop is None:
        return
    asyncio.run_coroutine_threadsafe(broadcast(event), event_loop)


async def broadcast(event):
    for queue in list(subscribers):
        await queue.put(event)


def restart_audio_monitor(audio_source):
    global audio_monitor
    if pipeline and pipeline.running:
        return
    if audio_monitor:
        audio_monitor.stop()
    audio_monitor = AudioLevelMonitor(audio_source, emit_event)
    audio_monitor.start()


def stop_audio_monitor():
    global audio_monitor
    if audio_monitor:
        audio_monitor.stop()
        audio_monitor = None


class ConfigUpdate(BaseModel):
    lang: str
    translate_tgt_lang: str = "jpn_Jpan"
    chunk_sec: float
    whisper_model: str
    whisper_beam: int
    translate_timeout: float
    min_chars: int
    buffer_chars: int
    chunk_flush_chars: int = 0
    max_feed_entries: int = 20


class AudioSourceUpdate(BaseModel):
    audio_source: str


@app.get("/favicon.ico")
def favicon():
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
def get_config():
    return normalize_config(load_config())


@app.put("/api/config")
def put_config(body: ConfigUpdate):
    if pipeline and pipeline.running:
        raise HTTPException(status_code=409, detail="Cannot change settings while running")
    config = load_config()
    apply_config_update(config, body)
    return {"ok": True}


@app.get("/api/presets")
def list_presets():
    presets = load_presets()
    return {"names": sorted(presets.keys())}


@app.put("/api/presets/{name}")
def put_preset(name: str, body: ConfigUpdate):
    if pipeline and pipeline.running:
        raise HTTPException(status_code=409, detail="Cannot change settings while running")
    if not name.strip():
        raise HTTPException(status_code=400, detail="Preset name is required")
    presets = load_presets()
    presets[name.strip()] = body.model_dump()
    save_presets(presets)
    return {"ok": True, "name": name.strip()}


@app.post("/api/presets/{name}/apply")
def apply_preset(name: str):
    if pipeline and pipeline.running:
        raise HTTPException(status_code=409, detail="Cannot change settings while running")
    presets = load_presets()
    preset = presets.get(name)
    if preset is None:
        raise HTTPException(status_code=404, detail="Preset not found")
    config = load_config()
    preset_data = {key: preset[key] for key in ConfigUpdate.model_fields if key in preset}
    apply_config_update(config, ConfigUpdate(**preset_data))
    return normalize_config(config)


@app.delete("/api/presets/{name}")
def delete_preset(name: str):
    if pipeline and pipeline.running:
        raise HTTPException(status_code=409, detail="Cannot change settings while running")
    presets = load_presets()
    if name not in presets:
        raise HTTPException(status_code=404, detail="Preset not found")
    del presets[name]
    save_presets(presets)
    return {"ok": True}


@app.get("/api/audio-sources")
def get_audio_sources():
    sources = list_audio_sources()
    return {"sources": sources, "selected": resolve_audio_source()}


@app.put("/api/audio-source")
def put_audio_source(body: AudioSourceUpdate):
    if pipeline and pipeline.running:
        raise HTTPException(status_code=409, detail="Cannot change settings while running")
    known = {source["name"] for source in list_audio_sources()}
    if body.audio_source not in known:
        raise HTTPException(status_code=400, detail="Unknown audio source")
    set_audio_source(body.audio_source)
    restart_audio_monitor(body.audio_source)
    return {"ok": True, "audio_source": body.audio_source}


@app.get("/api/status")
def get_status():
    return {"running": pipeline is not None and pipeline.running}


def health_url(translate_url):
    parsed = urlparse(translate_url)
    return f"{parsed.scheme}://{parsed.netloc}/health"


def server_label(url, urls):
    try:
        return f"gpu{urls.index(url)}"
    except ValueError:
        port = urlparse(url).port
        return f":{port}" if port else url


def check_translate_server(url, urls, timeout=3.0):
    label = server_label(url, urls)
    health_endpoint = health_url(url)
    try:
        request = urllib.request.Request(health_endpoint)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
        ok = response.status == 200 and payload.get("status") == "ok" and payload.get("model_loaded")
        return {
            "url": url,
            "label": label,
            "ok": ok,
            "health": payload if ok else None,
            "error": None if ok else "Model not loaded",
        }
    except urllib.error.HTTPError as exc:
        return {"url": url, "label": label, "ok": False, "health": None, "error": f"HTTP {exc.code}"}
    except Exception as exc:
        return {"url": url, "label": label, "ok": False, "health": None, "error": str(exc)}


@app.get("/api/translate-servers")
async def translate_servers():
    config = load_config()
    translate_urls = get_translate_urls(config)
    timeout = min(float(config.get("translate_timeout", 30)), 5.0)
    tasks = [
        asyncio.to_thread(check_translate_server, url, translate_urls, timeout)
        for url in translate_urls
    ]
    servers = await asyncio.gather(*tasks) if tasks else []
    return {"servers": servers}


@app.post("/api/start")
def start_pipeline():
    global pipeline
    if pipeline and pipeline.running:
        return {"ok": True, "running": True}
    stop_audio_monitor()
    config = load_config()
    config = dict(config)
    config["audio_source"] = resolve_audio_source()
    pipeline = WhisperPipeline(config, emit_event)
    pipeline.start()
    return {"ok": True, "running": True}


@app.post("/api/stop")
def stop_pipeline():
    global pipeline
    if pipeline:
        pipeline.stop()
        pipeline = None
    restart_audio_monitor(resolve_audio_source())
    return {"ok": True, "running": False}


async def event_stream():
    queue: asyncio.Queue = asyncio.Queue()
    subscribers.append(queue)
    await queue.put({
        "type": "status",
        "running": pipeline is not None and pipeline.running,
    })
    try:
        while True:
            event = await queue.get()
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    finally:
        if queue in subscribers:
            subscribers.remove(queue)


@app.get("/api/events")
async def events():
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Whisper browser UI server")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    host = args.host or config.get("host", "0.0.0.0")
    port = args.port or config.get("port", 9999)

    print(f"UI: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info", ws="none")


if __name__ == "__main__":
    main()
