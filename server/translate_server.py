import argparse
import json
import os
import time

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "server_config.json")
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


class TranslateRequest(BaseModel):
    text: str = Field(min_length=1)
    src_lang: str = "eng_Latn"
    tgt_lang: str = "jpn_Jpan"


class TranslateResponse(BaseModel):
    text: str
    elapsed: float


app = FastAPI()


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
    return device_map == "auto" and torch.cuda.device_count() >= 2


def build_max_memory(margin_gb):
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


def parse_args():
    parser = argparse.ArgumentParser(description="Translation server")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to config file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--instance",
        type=int,
        default=0,
        help="Instance index in server_config.json (default: 0)",
    )
    return parser.parse_args()


def main():
    global num_beams, max_new_tokens

    args = parse_args()
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

    uvicorn.run(app, host=config["host"], port=config["port"], log_level="info")


if __name__ == "__main__":
    main()
