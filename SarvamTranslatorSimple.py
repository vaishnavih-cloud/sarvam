

import os
import sys
import json
import requests
from typing import Optional

API_CHAT_URL = "https://api.sarvam.ai/v1/chat/completions"  # OpenAI-like chat endpoint

# Supported languages (common names). You can add or remove as needed.
SUPPORTED_LANGS = [
    "Hindi","Marathi","Gujarati","Bengali","Tamil","Telugu","Kannada","Malayalam",
    "Punjabi","Odia","Assamese","Urdu","Konkani","Sanskrit","Nepali","Manipuri",
    "Bodo","Dogri","Santali","Maithili","Sindhi","Kashmiri","English"
]

def print_lang_menu():
    print("\nChoose target language (enter number or language name):")
    for i, lang in enumerate(SUPPORTED_LANGS, start=1):
        print(f"  {i:02d}. {lang}")
    print()

def pick_language(choice: str) -> Optional[str]:
    choice = choice.strip()
    if not choice:
        return None
    # numeric selection
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(SUPPORTED_LANGS):
            return SUPPORTED_LANGS[idx]
        return None
    # accept name (case-insensitive)
    for lang in SUPPORTED_LANGS:
        if choice.lower() == lang.lower():
            return lang
    return None

def translate_via_api(text: str, target_lang: str, api_key: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "sarvam-translate",
        "messages": [
            {"role": "system", "content": f"Detect the language and translate the following text to {target_lang}."},
            {"role": "user", "content": text}
        ],
        "max_tokens": 512,
        "temperature": 0.0
    }
    resp = requests.post(API_CHAT_URL, headers=headers, json=payload, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")
    data = resp.json()
    # handle common response shapes
    if "choices" in data and len(data["choices"]) > 0:
        return data["choices"][0]["message"]["content"].strip()
    if "output_text" in data:
        return data["output_text"].strip()
    if "output" in data and isinstance(data["output"], list):
        parts = []
        for item in data["output"]:
            if isinstance(item, dict) and "content" in item:
                parts.append(item["content"])
            elif isinstance(item, str):
                parts.append(item)
        if parts:
            return " ".join(parts).strip()
    return json.dumps(data)

def translate_via_local_model(text: str, target_lang: str) -> str:
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except Exception as e:
        raise RuntimeError("Local mode requires 'transformers' package. Install with: pip install transformers torch accelerate") from e

    model_name = "sarvamai/sarvam-translate"
    print(f"[Local] Loading model {model_name}. This may take a long time and consume many GB of disk/RAM.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    messages = [
        {"role": "system", "content": f"Detect the language and translate the following text to {target_lang}."},
        {"role": "user", "content": text}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt")
    try:
        # move inputs to model device if supported
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except Exception:
        pass
    out = model.generate(**inputs, max_new_tokens=256, temperature=0.01)
    translated = tokenizer.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return translated.strip()

def main():
    print("\n=== Sarvam Translator CLI ===\n")
    # get the input text
    if len(sys.argv) > 1:
        # allow passing sentence in command line: python translate_cli.py "Hello world"
        input_text = " ".join(sys.argv[1:])
    else:
        input_text = input("Enter text to translate: ").strip()
        while not input_text:
            input_text = input("Please enter non-empty text: ").strip()

    print_lang_menu()
    lang_choice = input("Target language (number or name): ").strip()
    target_lang = pick_language(lang_choice)
    while target_lang is None:
        lang_choice = input("Invalid choice. Enter number or language name: ").strip()
        target_lang = pick_language(lang_choice)

    os.environ["SARVAM_API_KEY"] = "sk_3joxmkb4_PmejVzy3DWeEUtyUc045drct"
    api_key = os.environ.get("SARVAM_API_KEY")
    if api_key:
        print("\n[Mode] Using Sarvam cloud API (fast). Translating...\n")
        try:
            translated = translate_via_api(input_text, target_lang, api_key)
            print(f"Translated ({target_lang}):\n{translated}\n")
            return
        except Exception as e:
            print("API translation failed:", str(e))
            # fall through to ask about local
    else:
        print("\n[Info] No SARVAM_API_KEY found in environment. API mode unavailable.")

    # ask user whether to use local model
    use_local = input("Do you want to use local model instead? (y/N): ").strip().lower()
    if use_local not in ("y", "yes"):
        print("Exiting. To use cloud API set SARVAM_API_KEY environment variable and re-run.")
        return

    # confirm again (warn)
    confirm = input("Local model is large (~several GB). Continue and download model? (y/N): ").strip().lower()
    if confirm not in ("y", "yes"):
        print("Cancelled local download. Exiting.")
        return

    # perform local translation
    try:
        print("\n[Local] Translating (this may take time)...\n")
        translated = translate_via_local_model(input_text, target_lang)
        print(f"Translated ({target_lang}):\n{translated}\n")
    except Exception as e:
        print("Local translation failed:", str(e))
        print("Suggestion: set SARVAM_API_KEY for API mode or run on a GPU machine / Colab.")

if __name__ == "__main__":
    main()
