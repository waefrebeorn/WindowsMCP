# WuBu: Your Desktop AI Assistant

**WuBu** is a Python-based AI assistant designed to understand your commands and interact with your Windows desktop environment. It leverages Large Language Models (LLMs like Google's Gemini or local models via Ollama) for natural language understanding and task execution. WuBu uses dedicated libraries for screen interaction, OCR, vision analysis, voice control, and more.

## Features

*   **Desktop Control**: Programmatic control over mouse and keyboard actions.
*   **Screen Interaction**:
    *   Capture the full screen or specific regions.
    *   OCR (Optical Character Recognition) using Tesseract to find text and its coordinates on screen.
    *   Click on text found via OCR.
*   **Vision Analysis**: Utilizes Moondream v2 (or a similar vision model if configured) for general image description tasks.
*   **Window Management**: List open windows, get active window title, focus windows, get window geometry.
*   **File System (Read-Only Tools)**: Includes tools to list directory contents and read text files.
*   **Contextual Codebase Understanding**:
    *   **Codebase Indexing**: When WuBu starts, it indexes the files in the current working directory (where you run `python main.py`). It builds a Merkle tree of file hashes to efficiently track the state of your project. This helps WuBu understand the broader context of your work.
    *   **Ignore Files**: Respects `.gitignore` and an additional `.wubuignore` file in the project root to exclude certain files/directories from indexing.
    *   **@-Mentions for Files**: You can refer to specific files in your commands using `@` notation (e.g., "summarize @src/main.py"). WuBu will load the content of the mentioned file to provide more relevant responses.
*   **Voice Interaction**:
    *   **Activation Phrases**: Activate WuBu by saying "WuBu", "Hey WuBu", "Yo WuBu", "WooBoo", or "WuhBoo" followed by your command when using voice input mode (`--voice`).
    *   Speech-to-Text (STT): Uses local Whisper models for accurate transcription.
    *   Text-to-Speech (TTS): Provides spoken responses using pyttsx3.
*   **LLM Integration**: Supports Google Gemini (via API) and local LLMs (e.g., Llama, Phi, Qwen) through Ollama for natural language understanding and tool orchestration.
*   **Command-Line Interface**: Allows users to type commands or use voice to interact with their desktop.
*   **Configurable**: Settings for API keys, model names, and behavior can be managed through `config.json` and `.env` files.

## Prerequisites

*   **Python**: Version 3.9 or higher is recommended. Ensure Python is added to your system's PATH.
*   **Git**: For cloning the repository.
*   **Tesseract OCR**: Required for the `find_text_on_screen_and_click` tool and other precise text-location tasks.
    *   Installation instructions: [Tesseract OCR Documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html)
    *   Ensure Tesseract is added to your system's PATH. Alternatively, you can set the `TESSERACT_CMD_PATH` in your `config.json` to the full path of the Tesseract executable if it's not in your PATH or you need to specify a particular installation.
*   **Ollama**: (Optional, if using local LLMs/Moondream) Install from [ollama.com](https://ollama.com).
*   **ffmpeg**: (Required by `openai-whisper`) A cross-platform solution to record, convert and stream audio and video.
    *   Linux: `sudo apt update && sudo apt install ffmpeg`
    *   macOS: `brew install ffmpeg`
    *   Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH, or via Chocolatey: `choco install ffmpeg`

## Setup Instructions

1.  **Clone the Repository**:
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd <repository_directory>   # Replace <repository_directory>
    ```

2.  **Create and Activate Python Virtual Environment**:
    ```bash
    python -m venv venv
    ```
    Activate:
    *   Windows: `.\venv\Scripts\activate`
    *   macOS/Linux: `source venv/bin/activate`

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    This installs all necessary Python packages including `pyautogui`, `Pillow`, `requests`, `pytesseract`, `pandas`, `openai-whisper`, `sounddevice`, `scipy`, `pyttsx3`, `PyGetWindow`, and `PyWinCtl`.

4.  **Configuration**:
    *   **Copy Example Configuration**:
        ```bash
        cp config.example.json config.json
        ```
    *   **Edit `config.json`**: Review and update the settings:
        *   `LLM_PROVIDER`: `"gemini"` or `"ollama"`.
        *   `OLLAMA_API_URL`: (If Ollama) Defaults to `"http://localhost:11434"`.
        *   `OLLAMA_DEFAULT_MODEL`: (If Ollama) Default text LLM. Recommended: `"qwen2.5-coder:7b-instruct-q4_K_M"` for good tool calling capability.
        *   `MOONDREAM_API_URL`: Endpoint for Moondream v2 (if used, e.g., via Ollama: `"http://localhost:11434/api/generate"`).
        *   `OLLAMA_MOONDREAM_MODEL`: Name of Moondream model in Ollama (e.g., `"moondream"`).
        *   `SCREENSHOT_SAVE_PATH`: Optional path to save screenshots (e.g., `"./screenshots"`).
        *   `GEMINI_MODEL_NAME`: (If Gemini) e.g., `"gemini-1.5-flash-latest"`.
        *   `PYAUTOGUI_FAILSAFE_ENABLED`: `true` or `false`.
        *   `PYAUTOGUI_PAUSE_PER_ACTION`: e.g., `0.1`.
        *   `VOICE_RECORDING_DURATION`: Default recording time in seconds for voice input (e.g., `5`).
        *   `WHISPER_MODEL_NAME`: Whisper model for STT (e.g., `"base"`, `"tiny"`).
        *   `ENABLE_TTS`: `true` or `false` to enable/disable spoken responses.
    *   **Gemini API Key (if using Gemini)**:
        Create a `.env` file in the project root:
        ```env
        GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
        ```
    *   **Ollama Setup (if using Ollama)**:
        *   Ensure Ollama is installed and running.
        *   Pull your chosen LLM. For tool calling, it's recommended to use a model like `qwen2.5-coder:7b-instruct-q4_K_M`:
            ```bash
            ollama pull qwen2.5-coder:7b-instruct-q4_K_M
            ```
        *   Pull Moondream (if using for vision):
            ```bash
            ollama pull moondream
            ```
            (The `ollama_setup.bat` script can help automate pulling these recommended models.)
    *   **Microphone Access**: Ensure the application has permission to access your microphone for voice input.

## Running the Assistant

1.  **Activate your virtual environment**.
2.  **Run `main.py`**:
    ```bash
    python main.py
    ```
    **Command-Line Arguments**:
    *   `--voice`: Enable voice input mode (uses Whisper).
    *   `--llm_provider {gemini,ollama}`: Override `LLM_PROVIDER` from `config.json`.
    *   `--ollama_model <model_name>`: (If Ollama) Override `OLLAMA_DEFAULT_MODEL`.
    *   `--test_command "<command>"`: Execute a single command.
    *   `--test_file <filepath>`: Execute commands from a file.

### Interactive Mode
Type commands or use voice (if `--voice` flag is used).

**Voice Interaction:**
When using the `--voice` flag, activate WuBu by saying one of the activation phrases:
*   "WuBu"
*   "Hey WuBu"
*   "Yo WuBu"
*   "WooBoo"
*   "WuhBoo"

Follow the activation phrase with your command. For example: *"Hey WuBu, list all open windows."*
If you only say the activation phrase (e.g., "WuBu"), it will prompt you to type your command.

**Contextual Understanding with @-Mentions:**
WuBu indexes the files in the directory where it's started (your current working directory when you run `python main.py`). You can refer to files in this indexed project using `@` notation in your commands. This allows WuBu to load the content of that file and use it as context for your query. The file mentioned with `@` will also become the "current file" for subsequent commands until another file is mentioned.

Examples:
*   "List all open windows."
*   "What is the title of the active window?"
*   "Focus the window titled 'Notepad'." (Ensure Notepad is open)
*   (In a project with `requirements.txt`) *"Hey WuBu, what are the main dependencies in @requirements.txt?"*
*   (In a Python project) *"WuBu, can you explain the purpose of the main function in @src/app.py?"*
*   After the previous command: *"WuBu, what other functions are in the current file?"* (WuBu will know "current file" is `src/app.py`)
*   "Capture the screen and tell me what text you see near the top." (Uses screenshot + Moondream)
*   "Find the text 'File' on screen and click it." (Uses screenshot + Tesseract OCR)

**Note on Project Context:** The project context (indexed files) is determined by the directory you are in when you run `python main.py`. For best results, run WuBu from the root of the project you want it to be aware of. A `.wubuignore` file can be created in the project root, similar to `.gitignore`, to specify files or directories that WuBu should ignore during indexing.

## Available Tools (Examples)

*   **Screen & Vision**: `capture_screen_region`, `capture_full_screen`, `get_screen_resolution`, `analyze_image_with_vision_model` (Moondream), `find_text_on_screen_and_click` (Tesseract).
*   **Mouse & Keyboard**: `mouse_move`, `mouse_click`, `mouse_drag`, `mouse_scroll`, `keyboard_type`, `keyboard_press_key`, `keyboard_hotkey`.
*   **Window Management**: `list_windows`, `get_active_window_title`, `focus_window`, `get_window_geometry`.
*   **File System (Read-Only)**: `list_directory`, `read_text_file`. Note: For code understanding, prefer using `@filename` syntax which provides richer context to WuBu.

## Example Workflow

1.  **User (Voice/Text)**: *"Hey WuBu, what functions are defined in @myproject/utils.py?"*
2.  **WuBu (LLM Decision with Context)**: WuBu loads `myproject/utils.py` content. The LLM analyzes the file content provided in the context and identifies function definitions.
3.  **WuBu (Response to User - Text/TTS)**: "In `@myproject/utils.py`, I found the following functions: `helper_function1`, `util_func2`."
4.  **User**: *"WuBu, now focus the window 'Visual Studio Code'."*
5.  **WuBu**: Calls `focus_window` tool.
6.  **WuBu**: "Okay, I've attempted to focus 'Visual Studio Code'."
7.  **User**: *"In the current file, refactor `helper_function1` to be more efficient."* (Assuming `myproject/utils.py` is still considered the "current file" due to the previous @-mention).
8.  **WuBu (LLM Decision with Context)**: WuBu uses its knowledge of `helper_function1` from `myproject/utils.py` (which is the current file context) and suggests a refactoring, potentially by invoking a `replace_text_in_file` tool (if such a tool were implemented and available).
9.  **WuBu**: "Okay, I can try to refactor it. Here's a suggestion..."

---
*Several batch scripts (`.bat` files) are provided in the repository as convenient shortcuts for common tasks like initial setup (`ollama_setup.bat`, `setup_venv.bat`) and running WuBu. (Consider renaming `run_agent.bat` to `run_wubu.bat` and `run_ollama_agent.bat` to `run_ollama_wubu.bat`). While `python main.py` with appropriate arguments is the primary way to run the assistant, these scripts can simplify the process.*
