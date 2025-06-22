from google.genai import types
from typing import List, Dict, Any

# --- Desktop Tool Definitions for LLMs (Gemini and Ollama Schema Generation) ---

DESKTOP_TOOLS_INSTANCE = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="capture_screen_region",
            description="Captures a specified rectangular region of the primary screen and returns it as an image. The image can then be used as input for other tools that analyze images.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "x": types.Schema(
                        type=types.Type.INTEGER,
                        description="The x-coordinate of the top-left corner of the region.",
                    ),
                    "y": types.Schema(
                        type=types.Type.INTEGER,
                        description="The y-coordinate of the top-left corner of the region.",
                    ),
                    "width": types.Schema(
                        type=types.Type.INTEGER,
                        description="The width of the region in pixels.",
                    ),
                    "height": types.Schema(
                        type=types.Type.INTEGER,
                        description="The height of the region in pixels.",
                    ),
                    "save_path": types.Schema(
                        type=types.Type.STRING,
                        nullable=True,
                        description="Optional file path to save the captured image for debugging. If None, image is not saved to disk by this tool directly but returned for further processing.",
                    ),
                },
                required=["x", "y", "width", "height"],
            ),
        ),
        types.FunctionDeclaration(
            name="capture_full_screen",
            description="Captures the entire primary screen and returns it as an image. The image can then be used as input for other tools that analyze images.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "save_path": types.Schema(
                        type=types.Type.STRING,
                        nullable=True,
                        description="Optional file path to save the captured image for debugging. If None, image is not saved to disk by this tool directly but returned for further processing.",
                    )
                },
            ),
        ),
        types.FunctionDeclaration(
            name="get_screen_resolution",
            description="Returns the width and height of the primary screen in pixels.",
            parameters=types.Schema(type=types.Type.OBJECT, properties={}),
        ),
        types.FunctionDeclaration(
            name="mouse_move",
            description="Moves the mouse cursor to the specified X, Y coordinates on the screen.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "x": types.Schema(
                        type=types.Type.INTEGER, description="The target x-coordinate."
                    ),
                    "y": types.Schema(
                        type=types.Type.INTEGER, description="The target y-coordinate."
                    ),
                    "duration": types.Schema(
                        type=types.Type.NUMBER,
                        nullable=True,
                        description="Optional. Time in seconds to spend moving the mouse. Defaults to a short duration (e.g., 0.25s).",
                    ),
                },
                required=["x", "y"],
            ),
        ),
        types.FunctionDeclaration(
            name="mouse_click",
            description="Performs a mouse click at the specified X, Y coordinates, or at the current mouse position if coordinates are not provided.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "x": types.Schema(
                        type=types.Type.INTEGER,
                        nullable=True,
                        description="Optional. The target x-coordinate. If None, clicks at current mouse position.",
                    ),
                    "y": types.Schema(
                        type=types.Type.INTEGER,
                        nullable=True,
                        description="Optional. The target y-coordinate. If None, clicks at current mouse position.",
                    ),
                    "button": types.Schema(
                        type=types.Type.STRING,
                        nullable=True,
                        enum=["left", "middle", "right"],
                        description="Optional. Mouse button to click ('left', 'middle', 'right'). Defaults to 'left'.",
                    ),
                    "clicks": types.Schema(
                        type=types.Type.INTEGER,
                        nullable=True,
                        description="Optional. Number of times to click. Defaults to 1.",
                    ),
                    "interval": types.Schema(
                        type=types.Type.NUMBER,
                        nullable=True,
                        description="Optional. Time in seconds between clicks if clicks > 1. Defaults to 0.1s.",
                    ),
                },
                # No required fields, as clicking at current position with defaults is valid.
            ),
        ),
        types.FunctionDeclaration(
            name="mouse_drag",
            description="Drags the mouse from a starting X,Y position to an ending X,Y position while holding a mouse button.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "start_x": types.Schema(
                        type=types.Type.INTEGER,
                        description="The x-coordinate of the drag start position.",
                    ),
                    "start_y": types.Schema(
                        type=types.Type.INTEGER,
                        description="The y-coordinate of the drag start position.",
                    ),
                    "end_x": types.Schema(
                        type=types.Type.INTEGER,
                        description="The x-coordinate of the drag end position.",
                    ),
                    "end_y": types.Schema(
                        type=types.Type.INTEGER,
                        description="The y-coordinate of the drag end position.",
                    ),
                    "duration": types.Schema(
                        type=types.Type.NUMBER,
                        nullable=True,
                        description="Optional. Time in seconds to spend dragging. Defaults to 0.5s.",
                    ),
                    "button": types.Schema(
                        type=types.Type.STRING,
                        nullable=True,
                        enum=["left", "middle", "right"],
                        description="Optional. Mouse button to hold during drag. Defaults to 'left'.",
                    ),
                },
                required=["start_x", "start_y", "end_x", "end_y"],
            ),
        ),
        types.FunctionDeclaration(
            name="mouse_scroll",
            description="Scrolls the mouse wheel up or down. Can specify scroll location or use current mouse position.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "amount": types.Schema(
                        type=types.Type.INTEGER,
                        description="Number of units to scroll. Positive for up, negative for down.",
                    ),
                    "x": types.Schema(
                        type=types.Type.INTEGER,
                        nullable=True,
                        description="Optional. X-coordinate for scroll. Defaults to current mouse position.",
                    ),
                    "y": types.Schema(
                        type=types.Type.INTEGER,
                        nullable=True,
                        description="Optional. Y-coordinate for scroll. Defaults to current mouse position.",
                    ),
                },
                required=["amount"],
            ),
        ),
        types.FunctionDeclaration(
            name="keyboard_type",
            description="Types the provided text using the keyboard, as if typed by a user.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "text": types.Schema(
                        type=types.Type.STRING, description="The text string to type."
                    ),
                    "interval": types.Schema(
                        type=types.Type.NUMBER,
                        nullable=True,
                        description="Optional. Time in seconds between pressing each key. Defaults to a small interval (e.g., 0.01s).",
                    ),
                },
                required=["text"],
            ),
        ),
        types.FunctionDeclaration(
            name="keyboard_press_key",
            description="Presses a single special key (e.g., 'enter', 'esc', 'ctrl', 'f1') or a sequence of regular keys.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "key_name": types.Schema(
                        type=types.Type.STRING,
                        description="Name of the key to press (e.g., 'enter', 'a', 'ctrl', 'shift', 'alt', 'left', 'f5'). Can also be a list of characters to press sequentially e.g. 'abc'.",
                    )
                },
                required=["key_name"],
            ),
        ),
        types.FunctionDeclaration(
            name="keyboard_hotkey",
            description="Presses a combination of keys simultaneously (e.g., Ctrl+C, Alt+F4).",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "keys": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.STRING),
                        description="A list of key names to press together. Example: ['ctrl', 'c'] for Ctrl+C.",
                    )
                },
                required=["keys"],
            ),
        ),
        types.FunctionDeclaration(
            name="analyze_image_with_vision_model",
            description="Sends a previously captured image (identified by a reference ID or using the last captured image) along with a text prompt to a vision model (e.g., Moondream) for analysis, such as OCR or description. The image capture should typically happen in a prior step.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "prompt_text": types.Schema(
                        type=types.Type.STRING,
                        description="The text prompt to guide the vision model's analysis (e.g., 'What text is in this image?', 'Describe this UI element.').",
                    ),
                    "image_reference_id": types.Schema(
                        type=types.Type.STRING,
                        nullable=True,
                        description="Optional. A reference ID of a previously captured image to analyze. If not provided, the system may use the last captured image if available.",
                    ),
                },
                required=["prompt_text"],
            ),
        ),
        # Example of a more composite tool that might combine actions:
        types.FunctionDeclaration(
            name="find_text_on_screen_and_click",
            description="Captures the full screen, uses a vision model (OCR) to find the specified text, and then clicks on the center of the found text's bounding box. Returns coordinates if successful, or an error.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "text_to_find": types.Schema(
                        type=types.Type.STRING,
                        description="The text string to search for on the screen.",
                    ),
                    "click_button": types.Schema(
                        type=types.Type.STRING,
                        nullable=True,
                        enum=["left", "middle", "right"],
                        description="Optional. Mouse button to click if text is found. Defaults to 'left'.",
                    ),
                    "occurrence": types.Schema(
                        type=types.Type.INTEGER,
                        nullable=True,
                        description="Optional. Which occurrence of the text to click if multiple are found (1-based index). Defaults to 1 (the first one).",
                    ),
                },
                required=["text_to_find"],
            ),
        ),
        # Window Management Tools
        types.FunctionDeclaration(
            name="list_windows",
            description="Lists the titles of all currently open and visible windows. Can be filtered by a search string.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "title_filter": types.Schema(
                        type=types.Type.STRING,
                        nullable=True,
                        description="Optional. If provided, only windows whose titles contain this string (case-insensitive) will be returned.",
                    )
                },
            ),
        ),
        types.FunctionDeclaration(
            name="get_active_window_title",
            description="Returns the title of the currently active (focused) window.",
            parameters=types.Schema(type=types.Type.OBJECT, properties={}),
        ),
        types.FunctionDeclaration(
            name="focus_window",
            description="Attempts to focus (activate or bring to foreground) a window identified by its title. Matches exact title first, then substring.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "title": types.Schema(
                        type=types.Type.STRING,
                        description="The title of the window to focus (exact or substring).",
                    )
                },
                required=["title"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_window_geometry",
            description="Returns the position (x, y) and size (width, height) of a window identified by its title. Matches exact title first, then substring.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "title": types.Schema(
                        type=types.Type.STRING,
                        description="The title of the window to get geometry for (exact or substring).",
                    )
                },
                required=["title"],
            ),
        ),
        # File System Tools
        types.FunctionDeclaration(
            name="list_directory",
            description="Lists the contents (files and subdirectories) of a specified directory path.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "path": types.Schema(
                        type=types.Type.STRING,
                        description="The path to the directory to list.",
                    )
                },
                required=["path"],
            ),
        ),
        types.FunctionDeclaration(
            name="read_text_file",
            description="Reads the content of a specified text file. Content may be truncated if very large or if max_chars is specified.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "path": types.Schema(
                        type=types.Type.STRING,
                        description="The path to the text file to read.",
                    ),
                    "max_chars": types.Schema(
                        type=types.Type.INTEGER,
                        nullable=True,
                        description="Optional. Maximum number of characters to read from the beginning of the file.",
                    ),
                },
                required=["path"],
            ),
        ),
        types.FunctionDeclaration(
            name="write_text_file",
            description="Writes text content to a specified file. Can optionally overwrite if the file already exists.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "path": types.Schema(type=types.Type.STRING, description="The full path to the file to be written."),
                    "content": types.Schema(type=types.Type.STRING, description="The text content to write into the file."),
                    "overwrite": types.Schema(type=types.Type.BOOLEAN, nullable=True, description="If True, overwrite the file if it exists. Defaults to False."),
                },
                required=["path", "content"],
            ),
        ),
        types.FunctionDeclaration(
            name="append_text_to_file",
            description="Appends text content to an existing file. Creates the file if it does not exist.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "path": types.Schema(type=types.Type.STRING, description="The full path to the file to append to."),
                    "content": types.Schema(type=types.Type.STRING, description="The text content to append to the file."),
                },
                required=["path", "content"],
            ),
        ),
        types.FunctionDeclaration(
            name="create_directory",
            description="Creates a new directory at the specified path. Parent directories will be created if they don't exist.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "path": types.Schema(type=types.Type.STRING, description="The full path where the new directory should be created."),
                },
                required=["path"],
            ),
        ),
        types.FunctionDeclaration(
            name="delete_file_or_directory",
            description="Deletes a specified file or an entire directory (recursively). Use with caution.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "path": types.Schema(type=types.Type.STRING, description="The full path of the file or directory to delete."),
                },
                required=["path"],
            ),
        ),
        types.FunctionDeclaration(
            name="move_file_or_directory",
            description="Moves a file or directory from a source path to a destination path.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "source_path": types.Schema(type=types.Type.STRING, description="The full path of the source file or directory."),
                    "destination_path": types.Schema(type=types.Type.STRING, description="The full path for the destination. If it's a directory, the source will be moved into it."),
                },
                required=["source_path", "destination_path"],
            ),
        ),
        types.FunctionDeclaration(
            name="copy_file_or_directory",
            description="Copies a file or directory from a source path to a destination path. Overwrites destination file if it exists.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "source_path": types.Schema(type=types.Type.STRING, description="The full path of the source file or directory."),
                    "destination_path": types.Schema(type=types.Type.STRING, description="The full path for the destination. If it's a directory, the source will be copied into it."),
                },
                required=["source_path", "destination_path"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_file_properties",
            description="Retrieves properties of a specified file or directory, such as size, type, and modification times.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "path": types.Schema(type=types.Type.STRING, description="The full path to the file or directory."),
                },
                required=["path"],
            ),
        ),
        # --- Application Management Tools ---
        types.FunctionDeclaration(
            name="start_application",
            description="Starts or launches an application given its name or full path. Optional arguments can be provided.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "application_path_or_name": types.Schema(type=types.Type.STRING, description="The name of the application (e.g., 'notepad.exe', 'firefox') or its full path."),
                    "arguments": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING), nullable=True, description="Optional list of command-line arguments to pass to the application."),
                },
                required=["application_path_or_name"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_running_processes",
            description="Lists currently running processes, providing details like PID, name, username, status, CPU and memory usage.",
            parameters=types.Schema(type=types.Type.OBJECT, properties={}), # No parameters needed
        ),
        types.FunctionDeclaration(
            name="close_application_by_pid",
            description="Closes or terminates an application using its Process ID (PID). Can attempt a graceful termination or a forceful kill.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "pid": types.Schema(type=types.Type.INTEGER, description="The Process ID (PID) of the application to close."),
                    "force": types.Schema(type=types.Type.BOOLEAN, nullable=True, description="If True, forcefully kill the process. If False (default), attempt graceful termination first."),
                },
                required=["pid"],
            ),
        ),
        types.FunctionDeclaration(
            name="close_application_by_title",
            description="Closes or terminates an application by matching a substring in its main window title. This may affect multiple processes if titles are similar. Uses case-insensitive matching.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "window_title_substring": types.Schema(type=types.Type.STRING, description="A substring of the window title to search for and close."),
                    "force": types.Schema(type=types.Type.BOOLEAN, nullable=True, description="If True, forcefully kill the process(es). If False (default), attempt graceful termination first."),
                },
                required=["window_title_substring"],
            ),
        ),
        # --- System Information & Control Tools ---
        types.FunctionDeclaration(
            name="get_clipboard_text",
            description="Retrieves the current text content from the system clipboard.",
            parameters=types.Schema(type=types.Type.OBJECT, properties={}),
        ),
        types.FunctionDeclaration(
            name="set_clipboard_text",
            description="Sets the system clipboard to the provided text content.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "text": types.Schema(type=types.Type.STRING, description="The text to place onto the clipboard."),
                },
                required=["text"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_system_volume",
            description="Gets the current master system volume level (0-100). Currently Windows-only.",
            parameters=types.Schema(type=types.Type.OBJECT, properties={}),
        ),
        types.FunctionDeclaration(
            name="set_system_volume",
            description="Sets the master system volume level (0-100). Currently Windows-only.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "level": types.Schema(type=types.Type.INTEGER, description="The desired volume level (0-100)."),
                },
                required=["level"],
            ),
        ),
        types.FunctionDeclaration(
            name="lock_windows_session",
            description="Locks the current Windows user session. This tool is specific to Windows.",
            parameters=types.Schema(type=types.Type.OBJECT, properties={}),
        ),
        types.FunctionDeclaration(
            name="shutdown_windows_system",
            description="Initiates system shutdown, restart, or logoff. Primarily for Windows, with basic support for Linux/macOS via 'shutdown' command.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "mode": types.Schema(type=types.Type.STRING, enum=["shutdown", "restart", "logoff"], description="The desired power mode: 'shutdown', 'restart', or 'logoff'. Defaults to 'shutdown' if not specified by LLM but it should always specify one.", default="shutdown"),
                    "force": types.Schema(type=types.Type.BOOLEAN, nullable=True, description="If True, forcefully performs the operation (e.g., closes applications without saving). Defaults to False."),
                    "delay_seconds": types.Schema(type=types.Type.INTEGER, nullable=True, description="Delay in seconds before the operation. Defaults to 0."),
                },
                required=["mode"], # Mode is essential
            ),
        ),
        # --- Basic Web Interaction Tool ---
        types.FunctionDeclaration(
            name="open_url_in_default_browser",
            description="Opens a given URL in the system's default web browser.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "url": types.Schema(type=types.Type.STRING, description="The full URL to open (e.g., 'https://www.google.com'). Must include http:// or https:// scheme."),
                },
                required=["url"],
            ),
        ),

        # --- Code Editor Tool (Phase 2) ---
        types.FunctionDeclaration(
            name="show_code_in_editor",
            description="Displays given code content in a simple pop-up, read-only code viewer window. Useful for showing code snippets to the user.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "code_content": types.Schema(type=types.Type.STRING, description="The string of code or text to display in the editor window."),
                    "language": types.Schema(type=types.Type.STRING, nullable=True, description="Optional. The programming language of the code (e.g., 'python', 'javascript', 'text'). Currently used for window title, future for syntax highlighting."),
                    "window_title": types.Schema(type=types.Type.STRING, nullable=True, description="Optional. A specific title for the code viewer window."),
                },
                required=["code_content"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_contextual_code_info",
            description="Parses provided Python code and identifies the function or class definition that encloses a given line number. Returns information about this specific code structure.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "code_text": types.Schema(type=types.Type.STRING, description="The Python code content as a string."),
                    "line_number": types.Schema(type=types.Type.INTEGER, description="The 1-based line number within the code to find context for."),
                    "column_number": types.Schema(type=types.Type.INTEGER, nullable=True, description="Optional. The 0-based column number for more precise context (currently not heavily used by the backend but can be provided)."),
                    "language": types.Schema(type=types.Type.STRING, nullable=True, default="python", description="The programming language of the code. Currently only 'python' is supported."),
                },
                required=["code_text", "line_number"],
            ),
        ),

    ]
)


def get_ollama_tools_json_schema() -> List[Dict[str, Any]]:
    """
    Converts Gemini tool declarations from DESKTOP_TOOLS_INSTANCE
    to a JSON schema list compatible with Ollama.
    """
    ollama_tools = []

    gemini_type_to_json_type = {
        types.Type.STRING: "string",
        types.Type.OBJECT: "object",
        types.Type.ARRAY: "array",
        types.Type.NUMBER: "number",
        types.Type.INTEGER: "integer",
        types.Type.BOOLEAN: "boolean",
    }

    def convert_schema(gemini_schema: types.Schema) -> Dict[str, Any]:
        if not gemini_schema:
            return {}

        json_schema = {}
        gemini_type = gemini_schema.type

        if gemini_type in gemini_type_to_json_type:
            json_schema["type"] = gemini_type_to_json_type[gemini_type]
        elif gemini_schema.properties:
            json_schema["type"] = "object"
        else:
            json_schema["type"] = "string"  # Default/fallback

        if gemini_schema.description:
            json_schema["description"] = gemini_schema.description

        if gemini_schema.nullable:
            # For JSON schema, optionality is often handled by not being in 'required'.
            # Some systems support "nullable": true, or type: ["type", "null"]
            # We'll add "nullable" for clarity if the target system (Ollama) might use it.
            # Or, adjust based on Ollama's specific schema expectations.
            # For now, let's assume Ollama might handle it by type union or just optionality.
            # To be safe, if nullable is true, make it a union type with "null"
            if (
                json_schema.get("type") and json_schema.get("type") != "object"
            ):  # Avoid for objects with properties
                json_schema["type"] = [json_schema["type"], "null"]

        if gemini_schema.enum:
            json_schema["enum"] = list(gemini_schema.enum)

        if gemini_type == types.Type.OBJECT and gemini_schema.properties:
            json_schema["properties"] = {
                name: convert_schema(prop_schema)
                for name, prop_schema in gemini_schema.properties.items()
            }
            if gemini_schema.required:
                json_schema["required"] = list(gemini_schema.required)

        elif gemini_type == types.Type.ARRAY and gemini_schema.items:
            json_schema["items"] = convert_schema(gemini_schema.items)

        return json_schema

    if DESKTOP_TOOLS_INSTANCE and DESKTOP_TOOLS_INSTANCE.function_declarations:
        for declaration in DESKTOP_TOOLS_INSTANCE.function_declarations:
            tool_schema = {
                "name": declaration.name,
                "description": declaration.description,
                "parameters": (
                    convert_schema(declaration.parameters)
                    if declaration.parameters
                    else {"type": "object", "properties": {}}
                ),
            }
            ollama_tools.append({"type": "function", "function": tool_schema})

    return ollama_tools


if __name__ == "__main__":
    # Print the Gemini tool declarations (for inspection)
    # print("--- Gemini Tool Declarations ---")
    # if DESKTOP_TOOLS_INSTANCE and DESKTOP_TOOLS_INSTANCE.function_declarations:
    #     for func_decl in DESKTOP_TOOLS_INSTANCE.function_declarations:
    #         print(f"Name: {func_decl.name}")
    #         print(f"  Description: {func_decl.description}")
    #         if func_decl.parameters:
    #             print(f"  Parameters:")
    #             for param_name, param_schema in func_decl.parameters.properties.items():
    #                 print(f"    {param_name}:")
    #                 print(f"      Type: {param_schema.type}")
    #                 if param_schema.description:
    #                     print(f"      Description: {param_schema.description}")
    #                 if param_schema.nullable:
    #                     print(f"      Nullable: {param_schema.nullable}")
    #                 if param_schema.enum:
    #                     print(f"      Enum: {list(param_schema.enum)}")
    #         print("-" * 20)

    # Print the Ollama JSON schema (for inspection)
    print("\n--- Ollama JSON Schema ---")
    ollama_schema = get_ollama_tools_json_schema()
    import json

    print(json.dumps(ollama_schema, indent=2))

    # Verify a specific tool's schema for nullable properties
    print("\n--- Specific Tool Schema Check (mouse_click) ---")
    for tool in ollama_schema:
        if tool["function"]["name"] == "mouse_click":
            print(json.dumps(tool, indent=2))
            break
