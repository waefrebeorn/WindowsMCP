# WuBu: Your Desktop AI Assistant


**WuBu** is a Python-based AI assistant designed to understand your commands and interact with your Windows desktop environment. It leverages Large Language Models (LLMs like Google's Gemini or local models via Ollama) for natural language understanding and task execution. WuBu uses dedicated libraries for screen interaction, OCR, vision analysis, voice control, system interaction, and more.

## Features

*   **Desktop Control**: Programmatic control over mouse and keyboard actions.
*   **Screen Interaction**:
    *   Capture the full screen or specific regions.
    *   OCR (Optical Character Recognition) using Tesseract to find text and its coordinates on screen.
    *   Click on text found via OCR.
*   **Vision Analysis**: Utilizes Moondream v2 (or a similar vision model if configured) for general image description tasks.

*   **Window Management**: List open windows, get active window title, focus windows, get window geometry, and control active window (minimize, maximize, restore, close).
*   **Application Management**: Launch and close applications.
*   **System Monitoring**: Query system information like CPU usage, memory, disk space, and battery status.
*   **Web Interaction (Browser)**: Open URLs or perform web searches in the default browser.
*   **File System Tools**: Includes tools to list directory contents, read text files, and **perform write operations like creating files/folders, renaming, copying, and deleting. Exercise extreme caution with write operations.**
*   **Contextual Codebase Understanding**:
    *   **Codebase Indexing**: When WuBu starts, it indexes the files in the current working directory (Windows Search Index is used if available, falling back to manual scan). It builds a Merkle tree of file hashes to efficiently track the state of your project. This helps WuBu understand the broader context of your work.
    *   **Ignore Files**: Respects `.gitignore` and an additional `.wubuignore` file in the project root to exclude certain files/directories from indexing.
    *   **@-Mentions for Files**: You can refer to specific files in your commands using `@` notation (e.g., "summarize @src/main.py"). WuBu will load the content of the mentioned file to provide more relevant responses.
*   **Voice Interaction**:
    *   **Activation Phrases**: Activate WuBu by saying "WuBu", "Hey WuBu", "Yo WuBu", "WooBoo", or "WuhBoo" followed by your command when using voice input mode (`--voice`).
    *   Speech-to-Text (STT): Uses local Whisper models for accurate transcription.
    *   Text-to-Speech (TTS): Provides spoken responses using various engines:
        *   Default: `pyttsx3` (basic, system-dependent voices).
        *   Advanced: **Zonos TTS** (via Docker) for high-quality, clonable voices.
        *   Other engines like GLaDOS-style and Kokoro are also available.
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
*   **psutil**: (Required for System Monitoring tools) `pip install psutil` (this will be handled by `requirements.txt`).
*   **Docker Desktop**: (Required for Zonos TTS)
    *   Download and install Docker Desktop for Windows from [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop).
    *   Ensure Docker Desktop is running after installation. **This is crucial before running `setup_venv.bat` for Zonos setup.**
    *   The `setup_venv.bat` script will check for Docker. If found and running, it will attempt to clone the Zonos repository (if not already present in a `Zonos_src` subdirectory) and then build a local Docker image named `wubu_zonos_image` using Zonos's own Dockerfile. This image is then used by WuBu for Zonos TTS.

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
    This installs all necessary Python packages for WuBu's core functionality and various features.
    The `setup_venv.bat` script also includes checks for Docker. If Docker is running, it will proceed to clone the Zonos source code (into `Zonos_src/`) and build the `wubu_zonos_image` Docker image locally. This process might take some time.

4.  **Configuration**:
    *   **Copy Example Configuration**:
        ```bash
        cp config.example.json config.json
        ```
    *   **Edit `config.json`**: Review and update the settings:
        *   `LLM_PROVIDER`: `"gemini"` or `"ollama"`.
        *   `USE_WINDOWS_SEARCH_INDEX`: (Windows only) `true` or `false`. Defaults to `true`. If true, WuBu will try to use the Windows Search Index for faster file discovery. Falls back to manual scan if disabled or fails.
        *   (Other settings as before...)
    *   **Gemini API Key (if using Gemini)**:
        Create a `.env` file in the project root:
        ```env
        GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
        ```
    *   **Ollama Setup (if using Ollama)**:
        (As before...)
    *   **Text-to-Speech (TTS) Configuration (including Zonos via Docker)**:
        WuBu supports multiple TTS engines. You can configure them in the `tts` section of your `config.json` (or `wubu_config.yaml`).
        To enable Zonos TTS (which runs via Docker):
        ```json
        {
          // ... other configurations ...
          "tts": {
            "default_voice": "zonos_engine_cloning_service", // Example: Make Zonos the default TTS engine
            "zonos_voice_engine": {
              "enabled": true,
              "language": "en", // Default language for Zonos (e.g., "en", "ja", "zh", "fr", "de")
              "zonos_docker_image": "wubu_zonos_image", // Docker image built locally by setup_venv.bat
              "zonos_model_name_in_container": "Zyphra/Zonos-v0.1-transformer", // Model Zonos library loads inside container
              "device_in_container": "cpu", // "cpu" or "cuda" (for GPU use inside Docker)
              "default_reference_audio_path": "path/to_your/host_reference_speaker.wav"
              // Optional: Host path to a .wav file for default voice cloning.
              // Use forward slashes (/) or escaped backslashes (\\\\) in JSON for paths.
            },
            // ... other TTS engine configurations ...
            "wubu_glados_style_voice": { "enabled": false },
            "wubu_kokoro_voice": { "enabled": false }
          }
          // ... other configurations ...
        }
        ```
        **Zonos (Docker-based) Configuration Details**:
        *   `enabled` (boolean): Set to `true` to enable Zonos.
        *   `language` (string): Default language for Zonos (e.g., "en" maps to "en-us").
        *   `zonos_docker_image` (string): Should be `"wubu_zonos_image"`. This is the name of the Docker image built locally by the `setup_venv.bat` script from the Zonos repository source.
        *   `zonos_model_name_in_container` (string): The model name the script inside the Docker container will load using the Zonos library (e.g., `Zyphra/Zonos-v0.1-transformer`). This usually corresponds to a model available within the Zonos ecosystem.
        *   `device_in_container` (string): Instructs the script inside Docker to use "cpu" or "cuda". If "cuda", WuBu will attempt to pass GPU access to the container. Your Docker setup must support GPU passthrough.
        *   `default_reference_audio_path` (string, optional): Path on your *host machine* to a `.wav` audio file for the default speaker voice. This file will be mounted into the Docker container during synthesis.
        *   **Important**: Ensure Docker Desktop is installed and running *before* running `setup_venv.bat` to allow the Zonos image to be built.
    *   **Microphone Access**: Ensure the application has permission to access your microphone for voice input.

## Running the Assistant

1.  **Activate your virtual environment**.
2.  **Run `main.py` using one of the provided batch scripts (e.g., `run_wubu.bat`) or directly**:
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
When using the `--voice` flag, activate WuBu by saying one of the activation phrases: "WuBu", "Hey WuBu", "Yo WuBu", "WooBoo", or "WuhBoo". Follow with your command. If you only say the activation phrase, WuBu will prompt you to type your command.

**Contextual Understanding with @-Mentions:**
WuBu indexes files in its startup directory (current working directory). Use `@` to refer to files (e.g., "summarize @src/app.py"). The mentioned file becomes the "current file" for subsequent related commands.

**Note on Project Context:** Run WuBu from the root of the project you want it to be aware of. A `.wubuignore` file (similar to `.gitignore`) can exclude files/directories from indexing.

### Available Tools (Examples)

WuBu has access to a variety of tools to interact with your system:

*   **Screen & Vision**:
    *   `capture_screen_region`, `capture_full_screen`, `get_screen_resolution`
    *   `analyze_image_with_vision_model` (e.g., with Moondream for OCR or image description)
    *   `find_text_on_screen_and_click` (combines OCR and mouse click)
*   **Mouse & Keyboard**:
    *   `mouse_move`, `mouse_click`, `mouse_drag`, `mouse_scroll`
    *   `keyboard_type`, `keyboard_press_key`, `keyboard_hotkey`
*   **Window Management**:
    *   `list_windows`: Lists titles of open windows (can be filtered).
    *   `control_active_window`: Actions like "minimize", "maximize", "restore", "close", "get_title", "get_geometry" on the active window.
    *   `focus_window`: Brings a window to the foreground by its title.
*   **Application Management**:
    *   `start_application`: Launches an application by name or path (e.g., "notepad", "chrome.exe").
    *   `get_running_processes`: Lists currently running processes.
    *   `close_application_by_pid`, `close_application_by_title`: Terminates applications.
*   **System Information & Control**:
    *   `get_system_information`: Queries "cpu_usage", "memory_usage", "disk_usage" (optional path), "battery_status".
    *   `get_clipboard_text`, `set_clipboard_text`
    *   `get_system_volume`, `set_system_volume` (Windows-only)
    *   `lock_windows_session` (Windows-only)
    *   `shutdown_windows_system` (mode: "shutdown", "restart", "logoff")
*   **Web Interaction**:
    *   `open_url_or_search_web`: Opens a URL directly or performs a web search (e.g., for "Python tutorials") in the default browser.
*   **File System Tools**:
    *   `list_directory`: Lists contents of a directory.
    *   `read_text_file`: Reads content from a text file.
    *   `get_file_properties`: Gets metadata about a file or folder.
    *   **WARNING: The following file system tools modify files and directories. Use them with EXTREME CAUTION and ensure WuBu correctly understands your intent, especially with paths and destructive operations.**
        *   `create_folder`: Creates a new folder.
        *   `write_text_file`: Writes text to a file (can overwrite).
        *   `append_text_to_file`: Appends text to a file.
        *   `move_or_rename_item`: Moves/renames files or folders.
        *   `copy_item`: Copies files or folders.
        *   `delete_item`: Deletes files or folders. **This is highly destructive, especially with non-empty folders if forced.**

### Example Workflow

1.  **User (Voice)**: *"Hey WuBu, launch Notepad."*
2.  **WuBu**: Calls `start_application` tool with `application_name_or_path="notepad.exe"`. (Notepad opens)
3.  **User (Text)**: *"What's the CPU usage?"*
4.  **WuBu**: Calls `get_system_information` with `query="cpu_usage"`.
5.  **WuBu (Response)**: "Current CPU usage is 15.7%."
6.  **User (Text)**: *"Summarize the main points in @project_docs/feature_spec.md"*
7.  **WuBu (LLM Decision with Context)**: WuBu loads `project_docs/feature_spec.md`. The LLM analyzes the file content.
8.  **WuBu (Response)**: "The document outlines features X, Y, and Z..."
9.  **User (Voice)**: *"WuBu, create a folder named 'old_specs' in @project_docs."*
10. **WuBu (after confirming intent, if it were that cautious)**: Calls `create_folder` with `path="project_docs/old_specs"`.
11. **WuBu (Response)**: "Folder 'project_docs/old_specs' created."
12. **User (Text)**: *"Minimize this window."* (referring to the terminal/console WuBu is in)
13. **WuBu**: Calls `control_active_window` with `action="minimize"`.

---
*Several batch scripts (`.bat`) are provided in the repository for common tasks like initial setup (`ollama_setup.bat`, `setup_venv.bat`) and running WuBu (`run_wubu.bat`, `run_ollama_wubu.bat`). While `python main.py` with arguments is the primary way to run WuBu, these scripts can simplify the process.*
*Please review the batch scripts for any warnings, especially if you intend for WuBu to perform file system modifications.*

