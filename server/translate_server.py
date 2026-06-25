import argparse
import json
import os
import sys
import time
from pathlib import Path

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "server_config.json")
RUN_DIR = Path(__file__).resolve().parent / ".run"
SCRIPT_PATH = Path(__file__).resolve()
ROOT_DIR = SCRIPT_PATH.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from process_ctl import ProcessManager, dispatch_command

LANG_ALIASES = {
    "en": "eng_Latn",
    "eng": "eng_Latn",
    "eng_Latn": "eng_Latn",
    "es": "spa_Latn",
    "spa": "spa_Latn",
    "spa_Latn": "spa_Latn",
    "ja": "jpn_Jpan",
    "jpn": "jpn_Jpan",
    "jpn_Jpan": "jpn_Jpan",
    "zh": "zho_Hans",
    "zho_Hans": "zho_Hans",
    "ko": "kor_Hang",
    "kor_Hang": "kor_Hang",
}

MADLAD_TGT = {
    "jpn_Jpan": "ja",
    "eng_Latn": "en",
    "spa_Latn": "es",
    "zho_Hans": "zh",
    "kor_Hang": "ko",
}

app = None
tokenizer = None
translator = None
model_name = None
model_backend = "nllb"
device = "cpu"
device_map = None
load_in_8bit = False
load_in_4bit = False
num_beams = 4
max_new_tokens = 128


def load_config(path, instance=0):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    config = dict(raw)
    instances = config.pop("instances", None)
    if not instances:
        return config

    if instance < 0 or instance >= len(instances):
        raise SystemExit(
            f"--instance {instance} is undefined (use 0–{len(instances) - 1})"
        )

    config.update(instances[instance])
    return config


def instance_count(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    instances = raw.get("instances")
    return len(instances) if instances else 1


def detect_backend(name):
    if "madlad" in name.lower():
        return "madlad"
    return "nllb"


def normalize_lang(lang):
    return LANG_ALIASES.get(lang, lang)


def normalize_src_lang(src_lang):
    return normalize_lang(src_lang)


def normalize_tgt_lang(tgt_lang):
    return normalize_lang(tgt_lang)


def model_input_device():
    if hasattr(translator, "device"):
        return translator.device
    return next(translator.parameters()).device


def uses_multi_gpu():
    import torch

    return device_map == "auto" and torch.cuda.device_count() >= 2


def build_max_memory(margin_gb):
    import torch

    max_memory = {}
    for index in range(torch.cuda.device_count()):
        total_gb = torch.cuda.get_device_properties(index).total_memory / (1024**3)
        usable_gb = max(total_gb - margin_gb, 1.0)
        max_memory[index] = f"{usable_gb:.1f}GiB"
    return max_memory


def normalize_device(device_name):
    if device_name == "cuda":
        return "cuda:0"
    return device_name


def load_model(config):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

    global tokenizer, translator, model_name, model_backend, device, device_map, load_in_8bit, load_in_4bit

    name = config["model"]
    device_name = normalize_device(config.get("device", "cuda:0"))
    map_devices = config.get("device_map")
    load_in_8bit = bool(config.get("load_in_8bit", False))
    load_in_4bit = bool(config.get("load_in_4bit", False))

    model_name = name
    model_backend = detect_backend(name)
    device_map = map_devices
    load_kwargs = {}
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        if gpu_count >= 2 and map_devices == "auto":
            device_map = "auto"
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = device_name
    elif load_in_8bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        use_multi_gpu = map_devices == "auto" or gpu_count >= 2
        if use_multi_gpu and gpu_count >= 2:
            device_map = "auto"
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = device_name
    elif map_devices == "auto":
        load_kwargs["device_map"] = "auto"
        load_kwargs["torch_dtype"] = torch.float16
        margin_gb = config.get("gpu_memory_margin_gb", 0.8)
        load_kwargs["max_memory"] = build_max_memory(margin_gb)
    elif device_name.startswith("cuda"):
        load_kwargs["torch_dtype"] = torch.float16
    else:
        load_kwargs["torch_dtype"] = torch.float32

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(name)
    translator = AutoModelForSeq2SeqLM.from_pretrained(name, **load_kwargs)

    if load_in_4bit or load_in_8bit or device_map == "auto":
        device = str(model_input_device())
    else:
        translator = translator.to(device_name)
        device = device_name

    if model_backend == "nllb":
        translator.generation_config.max_length = None
        translator.generation_config.no_repeat_ngram_size = 3

    if hasattr(translator, "hf_device_map") and translator.hf_device_map:
        offload = {str(v) for v in translator.hf_device_map.values()} & {"cpu", "disk"}
        if offload:
            print(f"WARNING: Part of the model is offloaded to {offload}. OOM may occur during inference.")


def prepare_input(text, src_lang, tgt_lang):
    if model_backend == "madlad":
        code = MADLAD_TGT.get(normalize_tgt_lang(tgt_lang), "ja")
        text = f"<2{code}> {text}"
    else:
        tokenizer.src_lang = normalize_src_lang(src_lang)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    input_device = model_input_device()
    return {key: value.to(input_device) for key, value in inputs.items()}


def translate_text(text, src_lang, tgt_lang="jpn_Jpan"):
    import torch

    inputs = prepare_input(text, src_lang, tgt_lang)
    normalized_tgt = normalize_tgt_lang(tgt_lang)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
    }
    if model_backend == "nllb":
        generate_kwargs["forced_bos_token_id"] = tokenizer.convert_tokens_to_ids(normalized_tgt)
    if uses_multi_gpu():
        generate_kwargs["synced_gpus"] = True

    generated = translator.generate(
        **inputs,
        **generate_kwargs,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    translated = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
    return translated, elapsed


def gpu_memory_status():
    import torch

    if not torch.cuda.is_available():
        return []
    rows = []
    for index in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(index).total_memory
        used = torch.cuda.memory_allocated(index)
        rows.append({
            "gpu": index,
            "used_mb": round(used / (1024**2)),
            "total_mb": round(total / (1024**2)),
        })
    return rows


def create_app():
    global app

    if app is not None:
        return app

    from fastapi import FastAPI
    from pydantic import BaseModel, Field

    class TranslateRequest(BaseModel):
        text: str = Field(min_length=1)
        src_lang: str = "eng_Latn"
        tgt_lang: str = "jpn_Jpan"

    class TranslateResponse(BaseModel):
        text: str
        elapsed: float

    app = FastAPI()

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": model_name,
            "backend": model_backend,
            "device": device,
            "device_map": device_map,
            "load_in_8bit": load_in_8bit,
            "load_in_4bit": load_in_4bit,
            "gpus": gpu_memory_status(),
            "model_loaded": translator is not None,
        }

    @app.post("/translate", response_model=TranslateResponse)
    def translate(request: TranslateRequest):
        translated, elapsed = translate_text(request.text, request.src_lang, request.tgt_lang)
        return TranslateResponse(text=translated, elapsed=elapsed)

    return app


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to config file (default: {DEFAULT_CONFIG_PATH})",
    )

    parser = argparse.ArgumentParser(description="Translation server", add_help=False)
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", parents=[common], help="Run server (foreground)", add_help=False)
    serve.add_argument(
        "--instance",
        type=int,
        default=0,
        help="Instance index in server_config.json (default: 0)",
    )

    subparsers.add_parser("start", parents=[common], help="Start all instances in background", add_help=False)
    subparsers.add_parser("kill", parents=[common], help="Stop all instances", add_help=False)
    subparsers.add_parser("restart", parents=[common], help="Stop all instances, then start", add_help=False)
    return parser


def instance_names(config_path):
    names = [f"instance-{i}" for i in range(instance_count(config_path))]
    manager = ProcessManager(RUN_DIR)
    for name in manager.list_names("instance-"):
        if name not in names:
            names.append(name)
    return names


def cmd_start(args):
    manager = ProcessManager(RUN_DIR)
    for index in range(instance_count(args.config)):
        name = f"instance-{index}"
        config = load_config(args.config, index)
        if manager.is_alive(name):
            pid = manager.read_pid(name)
            print(f"Instance {index} already running (pid {pid}, port {config['port']})")
            continue

        cmd = [
            sys.executable,
            str(SCRIPT_PATH),
            "serve",
            "--config",
            os.path.abspath(args.config),
            "--instance",
            str(index),
        ]
        pid = manager.spawn(name, cmd, cwd=SCRIPT_PATH.parent)
        print(f"Started instance {index} on port {config['port']} (pid {pid})")


def cmd_kill(args):
    manager = ProcessManager(RUN_DIR)
    for name in instance_names(args.config):
        if manager.is_alive(name):
            if manager.kill(name):
                print(f"Stopped {name}")
            else:
                print(f"Failed to stop {name}", file=sys.stderr)
        elif manager.read_pid(name) is not None or manager.pid_path(name).exists():
            manager.pid_path(name).unlink(missing_ok=True)
            print(f"{name} was not running (removed stale pid file)")


def cmd_restart(args):
    cmd_kill(args)
    alive = ProcessManager(RUN_DIR).wait_all_dead(instance_names(args.config))
    if alive:
        print(f"Processes still running: {', '.join(alive)}", file=sys.stderr)
        raise SystemExit(1)
    cmd_start(args)


def run_server(args):
    import torch
    import uvicorn

    global num_beams, max_new_tokens

    config = load_config(args.config, args.instance)
    num_beams = config["beam"]
    max_new_tokens = config["max_tokens"]

    device_name = config.get("device", "cuda:0")
    map_devices = config.get("device_map")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Set device to cpu in server_config.json.")
    if (map_devices == "auto" or config.get("load_in_8bit")) and not torch.cuda.is_available():
        raise SystemExit("GPU mode requires CUDA.")

    config_path = os.path.abspath(args.config)
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    load_8bit = bool(config.get("load_in_8bit", False))
    load_4bit = bool(config.get("load_in_4bit", False))
    will_split_8bit = load_8bit and gpu_count >= 2
    will_split_4bit = load_4bit and map_devices == "auto" and gpu_count >= 2
    if will_split_4bit:
        mode = "4bit+2GPU"
    elif load_4bit:
        mode = "4bit"
    elif will_split_8bit:
        mode = "8bit+2GPU"
    elif load_8bit:
        mode = "8bit"
    else:
        mode = f"fp16 device_map={map_devices}"

    print(f"Config: {config_path} (instance {args.instance})")
    print(f"Device: {device_name}, Port: {config['port']}")
    print(f"GPUs on host: {gpu_count}")
    print(f"Loading {config['model']} ({mode}) ...")
    load_model(config)
    print(f"Ready on http://{config['host']}:{config['port']} ({model_name})")

    uvicorn.run(create_app(), host=config["host"], port=config["port"], log_level="info")


def main():
    dispatch_command(build_parser(), {
        "serve": run_server,
        "start": cmd_start,
        "kill": cmd_kill,
        "restart": cmd_restart,
    })


if __name__ == "__main__":
    main()
