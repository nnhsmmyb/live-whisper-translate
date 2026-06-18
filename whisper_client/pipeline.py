import json
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from itertools import count
from urllib.parse import urlparse

import numpy as np
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
SENTENCE_PATTERN = re.compile(r".+?[.!?….。！？]+")

WHISPER_TO_NLLB = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "ja": "jpn_Jpan",
}

DEFAULT_TRANSLATE_URLS = [
    "http://localhost:8765/translate",
    "http://localhost:8766/translate",
]


def get_translate_urls(config):
    return config.get("translate_urls") or DEFAULT_TRANSLATE_URLS


def list_audio_sources():
    default = default_monitor()
    output = subprocess.check_output(["pactl", "list", "sources", "short"], text=True)
    sources = []
    for line in output.strip().splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        name = parts[1]
        sources.append({
            "name": name,
            "description": parts[2] if len(parts) > 2 else name,
            "is_default": name == default,
        })
    return sources


def default_monitor():
    sink = subprocess.check_output(["pactl", "get-default-sink"], text=True).strip()
    return f"{sink}.monitor"


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


def merge_text(previous, new):
    if not previous:
        return new
    if not new:
        return previous
    if new in previous:
        return previous
    if previous in new:
        return new

    max_overlap = min(len(previous), len(new))
    for size in range(max_overlap, 0, -1):
        if previous[-size:] == new[:size]:
            return previous + new[size:]

    return f"{previous} {new}"


def split_sentences(text):
    sentences = []
    last_end = 0
    for match in SENTENCE_PATTERN.finditer(text):
        sentences.append(normalize_text(match.group(0)))
        last_end = match.end()
    remainder = normalize_text(text[last_end:])
    return sentences, remainder


def merge_remainder(sentences, remainder, min_chars):
    if not sentences:
        return [], remainder
    if len(sentences[-1]) >= min_chars:
        return sentences, remainder
    short = sentences.pop()
    remainder = normalize_text(f"{short} {remainder}") if remainder else short
    return sentences, remainder


def server_label(url, urls):
    try:
        return f"gpu{urls.index(url)}"
    except ValueError:
        port = urlparse(url).port
        return f":{port}" if port else url


def translate_via_server(url, text, src_lang, tgt_lang, timeout):
    body = json.dumps({
        "text": text,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
    }).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{url}: {exc.reason}") from exc
    elapsed = time.perf_counter() - started
    return payload["text"], elapsed


def translate_with_failover(urls, start_index, text, src_lang, tgt_lang, timeout):
    errors = []
    for offset in range(len(urls)):
        url = urls[(start_index + offset) % len(urls)]
        try:
            translated, elapsed = translate_via_server(url, text, src_lang, tgt_lang, timeout)
            return translated, elapsed, url
        except (urllib.error.URLError, RuntimeError) as exc:
            errors.append(str(exc))
    raise RuntimeError(" / ".join(errors))


class AudioLevelMonitor:
    def __init__(self, audio_source, emit):
        self.audio_source = audio_source
        self.emit = emit
        self._stop = threading.Event()
        self._thread = None
        self._proc = None

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._proc:
            self._proc.terminate()
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        self._proc = None

    def _run(self):
        cmd = [
            "parec",
            "-d", self.audio_source,
            "--format=s16le",
            f"--rate={SAMPLE_RATE}",
            "--channels=1",
        ]
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        except Exception as exc:
            self.emit({"type": "error", "message": f"Failed to start audio monitor: {exc}"})
            return

        bytes_per_chunk = int(SAMPLE_RATE * 0.1 * 2)
        try:
            while not self._stop.is_set():
                raw = self._proc.stdout.read(bytes_per_chunk)
                if not raw:
                    continue
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                level = float(np.sqrt(np.mean(audio ** 2)))
                self.emit({"type": "audio_level", "level": min(level * 8, 1.0)})
        finally:
            if self._proc:
                self._proc.terminate()
                self._proc = None


class WhisperPipeline:
    def __init__(self, config, emit):
        self.config = config
        self.emit = emit
        self._stop = threading.Event()
        self._thread = None
        self._proc = None
        self._executor = None

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._proc:
            self._proc.terminate()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        self.emit({"type": "status", "running": False})

    def _run(self):
        config = self.config
        translate_urls = get_translate_urls(config)
        audio_source = config.get("audio_source") or default_monitor()
        whisper_lang = config["lang"]
        source_lang = WHISPER_TO_NLLB.get(whisper_lang, whisper_lang)
        tgt_lang = config.get("translate_tgt_lang", "jpn_Jpan")

        try:
            whisper = WhisperModel(config["whisper_model"], device="cuda", compute_type="float16")
        except Exception as exc:
            self.emit({"type": "error", "message": f"Failed to load Whisper model: {exc}"})
            return

        cmd = [
            "parec",
            "-d", audio_source,
            "--format=s16le",
            f"--rate={SAMPLE_RATE}",
            "--channels=1",
        ]
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        except Exception as exc:
            self.emit({"type": "error", "message": f"Failed to start audio capture: {exc}"})
            return

        bytes_per_chunk = int(SAMPLE_RATE * config["chunk_sec"] * 2)
        text_buffer = ""
        last_translated = ""
        url_cycle = count()
        skip_translate = source_lang == tgt_lang
        self._executor = None if skip_translate else ThreadPoolExecutor(max_workers=len(translate_urls))

        self.emit({
            "type": "status",
            "running": True,
            "audio_source": audio_source,
            "lang": whisper_lang,
        })

        def submit_translation(sentence):
            self.emit({"type": "transcription", "text": sentence})

            if skip_translate:
                self.emit({
                    "type": "translation",
                    "source": sentence,
                    "text": sentence,
                    "elapsed": 0,
                    "gpu": "-",
                })
                return

            url_index = next(url_cycle)

            def on_done(future):
                try:
                    translated, elapsed, server_url = future.result()
                except Exception as exc:
                    self.emit({"type": "error", "message": f"Translation error: {exc}"})
                    return
                self.emit({
                    "type": "translation",
                    "source": sentence,
                    "text": translated,
                    "elapsed": round(elapsed, 2),
                    "gpu": server_label(server_url, translate_urls),
                })

            future = self._executor.submit(
                translate_with_failover,
                translate_urls,
                url_index,
                sentence,
                source_lang,
                tgt_lang,
                config["translate_timeout"],
            )
            future.add_done_callback(on_done)

        chunk_flush_chars = config.get("chunk_flush_chars") or 0

        def flush_buffer(force=False, on_chunk=False):
            nonlocal text_buffer, last_translated
            text_buffer = normalize_text(text_buffer)
            if not text_buffer:
                return

            sentences, remainder = split_sentences(text_buffer)
            chunk_flushed = False
            if not sentences and len(text_buffer) >= config["buffer_chars"]:
                sentences, remainder = [text_buffer], ""
            elif on_chunk and chunk_flush_chars > 0 and not sentences:
                if len(text_buffer) >= chunk_flush_chars:
                    sentences, remainder = [text_buffer], ""
                    chunk_flushed = True

            if not chunk_flushed:
                sentences, remainder = merge_remainder(sentences, remainder, config["min_chars"])

            for sentence in sentences:
                if sentence == last_translated:
                    continue
                submit_translation(sentence)
                last_translated = sentence

            if force and remainder and len(remainder) >= config["min_chars"]:
                if remainder != last_translated:
                    submit_translation(remainder)
                    last_translated = remainder
                remainder = ""

            text_buffer = remainder

        try:
            while not self._stop.is_set():
                raw = self._proc.stdout.read(bytes_per_chunk)
                if not raw:
                    continue

                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                level = float(np.sqrt(np.mean(audio ** 2)))
                self.emit({"type": "audio_level", "level": min(level * 8, 1.0)})

                segments, _info = whisper.transcribe(
                    audio,
                    language=whisper_lang,
                    task="transcribe",
                    beam_size=config["whisper_beam"],
                    vad_filter=True,
                    condition_on_previous_text=False,
                    without_timestamps=True,
                )

                chunk_text = normalize_text(" ".join(seg.text.strip() for seg in segments))
                if not chunk_text:
                    continue

                text_buffer = normalize_text(merge_text(text_buffer, chunk_text))
                flush_buffer(on_chunk=True)
        except Exception as exc:
            self.emit({"type": "error", "message": str(exc)})
        finally:
            flush_buffer(force=True)
            if self._proc:
                self._proc.terminate()
            if self._executor:
                self._executor.shutdown(wait=True)
                self._executor = None
            self.emit({"type": "status", "running": False})
