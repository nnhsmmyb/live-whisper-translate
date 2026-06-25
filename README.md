# live-whisper-translate

Whisper によるリアルタイム音声文字起こしと、NLLB / MADLAD 翻訳サーバを使った翻訳ツールです。

## 構成

| ディレクトリ | 役割 |
|-------------|------|
| `server/` | 翻訳 API（FastAPI + Hugging Face transformers） |
| `whisper_client/` | ブラウザ UI とパイプライン制御（FastAPI + faster-whisper） |

## 必要環境

- Linux（PulseAudio または PipeWire、`pactl` / `parec` が使えること）
- NVIDIA GPU + CUDA（Whisper と翻訳モデル用。推奨）
- Python 3.10 以上

## セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 使い方

### 1. 翻訳サーバを起動

```bash
cd server
./start_translate_servers.sh
```

`server_config.json` の `instances` に登録された数だけサーバが起動します（デフォルトはポート 8765 / 8766）。

### 2. Web UI を起動

```bash
cd whisper_client
python client_server.py
```

ブラウザで `http://localhost:9999` を開きます。

## 環境に合わせた設定

リポジトリのデフォルトは `localhost` と自動検出の音声ソースです。別環境向けに変更する場合は、次のファイルを編集します。

### 翻訳サーバの URL（別マシンや LAN 内の IP を使う場合）

`whisper_client/config.json` の `translate_urls` を変更します。

```json
"translate_urls": [
  "http://localhost:8765/translate",
  "http://localhost:8766/translate"
]
```

別マシンや LAN 内の IP を使う場合は、上記の `localhost` をそのマシンの IP に置き換えます。

### 音声ソース

起動時に `pactl` で PulseAudio のソースをすべて検出し、Web UI の「音声ソース」に一覧表示します。デフォルトのスピーカー出力（`*.monitor`）が自動選択されます。

別のデバイスを使う場合は UI から選ぶだけで構いません。`config.json` への記述は不要です（サーバ再起動後は再びデフォルトが選ばれます）。

### 翻訳サーバの GPU・ポート

`server/server_config.json` の `instances` で、ポートと GPU を指定します。

```json
"instances": [
  {"port": 8765, "device": "cuda:0"},
  {"port": 8766, "device": "cuda:1"}
]
```

### Web UI の待ち受けアドレス

`whisper_client/config.json` の `host` / `port` を変更します（デフォルト: `0.0.0.0:9999`）。

## 設定項目の一覧

### 翻訳サーバ（`server/server_config.json`）

- `model` — Hugging Face のモデル ID（例: `facebook/nllb-200-3.3B`）
- `instances` — `{port, device}` のリスト。マルチ GPU 構成向け
- `load_in_8bit` / `load_in_4bit` — 量子化オプション

### Web クライアント（`whisper_client/config.json`）

- `translate_urls` — 翻訳サーバのエンドポイント一覧
- `lang` — Whisper の入力言語（`en`, `es`, `ja`）
- `translate_tgt_lang` — 翻訳先の NLLB 言語コード（例: `jpn_Jpan`）
- `chunk_sec`, `min_chars`, `buffer_chars` — バッファリングと文の区切り

UI から保存したプリセットは `whisper_client/presets.json` に記録されます。

## ライセンス

MIT
