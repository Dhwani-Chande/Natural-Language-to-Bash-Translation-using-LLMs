---
title: NL To Bash
emoji: 💻
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
---

# NL to Bash Translator API

Fine-tuned Llama-3.2-1B & Qwen2.5-Coder on 40K NL→Bash pairs.

## API Endpoints

- `GET /health` — model status
- `POST /translate` — translate natural language to bash
- `POST /batch` — batch translate up to 10 instructions
