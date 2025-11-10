
# Sarvam Translator CLI

A simple command-line interface (CLI) tool for translating text between Indian languages using the Sarvam AI Translation API or an optional local model.

This project supports translation across 20+ Indian languages and English, making it ideal for multilingual communication, research, or quick translations directly from your terminal.

------------------------------------------------------------

## Features

- Translate between 20+ Indian languages and English.
- Uses Sarvam AI API (fast and accurate).
- Optionally runs locally using Hugging Face `sarvamai/sarvam-translate` model.
- Automatically detects source language.
- Command-line interface — no GUI needed.

------------------------------------------------------------

## Installation

1. Clone this repository
   ```
   git clone https://github.com/<your-username>/SarvamTranslatorCLI.git
   cd SarvamTranslatorCLI
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

------------------------------------------------------------

## Setup

### Option 1 — Use Cloud API (Recommended)
1. Get your Sarvam API key from https://sarvam.ai
2. Set it as an environment variable:
   ```
   export SARVAM_API_KEY="your_api_key_here"
   ```
3. Run the program:
   ```
   python SarvamTranslatorSimple.py
   ```

### Option 2 — Use Local Model
If you don't have an API key, the program can download and run a local translation model using Hugging Face Transformers.

Note: Local mode requires several GB of RAM and storage.

------------------------------------------------------------

## Usage

### Example 1 — Interactive Mode
```
python SarvamTranslatorSimple.py
```
You’ll be prompted to:
1. Enter text to translate.
2. Choose a target language by number or name.

### Example 2 — Command-Line Mode
You can pass text directly as an argument:
```
python SarvamTranslatorSimple.py "Good morning! How are you?"
```

------------------------------------------------------------

## Supported Languages

Hindi, Marathi, Gujarati, Bengali, Tamil, Telugu, Kannada, Malayalam, Punjabi, Odia, Assamese, Urdu, Konkani, Sanskrit, Nepali, Manipuri, Bodo, Dogri, Santali, Maithili, Sindhi, Kashmiri, English

------------------------------------------------------------

## Example Output

```
=== Sarvam Translator CLI ===

Enter text to translate: Hello, how are you?

Choose target language:
  01. Hindi
  02. Marathi
  03. Gujarati
  ...
Target language (number or name): 2

[Mode] Using Sarvam cloud API (fast). Translating...

Translated (Marathi):
नमस्कार, तुम्ही कसे आहात?
```

------------------------------------------------------------

## Project Structure

```
SarvamTranslatorCLI/
├── SarvamTranslatorSimple.py
├── requirements.txt
└── README.md
```

------------------------------------------------------------

## Dependencies

- Python 3.8+
- requests
- transformers
- torch
- accelerate

(see `requirements.txt` for details)

------------------------------------------------------------

## requirements.txt

```
requests
transformers
torch
accelerate
```


