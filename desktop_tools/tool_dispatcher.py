import json
import logging
import asyncio
import uuid  # For generating IDs if needed for image references
from typing import Any, Dict, NamedTuple
from PIL import Image

# Local module imports for desktop actions and vision model
from . import screen
from . import mouse
from . import keyboard
from . import moondream_interaction
from . import ocr_service  # Import the new OCR service
from . import window_manager  # Import the new window manager
from . import file_system  # Import the new file system manager

# Attempt to import configuration for screenshot path
try:
    from config_manager import config as global_config
except ImportError:
    global_config = {}

logger = logging.getLogger(__name__)


# Generic FunctionCall Representation
class FunctionCall(NamedTuple):
    name: str
    args: Dict[str, Any]
    id: str = None


class DesktopToolDispatcher:
    def __init__(self):
        self.captured_images_store: Dict[str, Image.Image] = {}

    def _get_screenshot_save_path(
        self, tool_name: str, image_id: str = None
    ) -> str | None:
        base_path_str = global_config.get("SCREENSHOT_SAVE_PATH", "")
        if not base_path_str:
            return None
        from pathlib import Path

        base_path = Path(base_path_str)
        base_path.mkdir(parents=True, exist_ok=True)
        filename = f"{tool_name.replace('_', '-')}"
        if image_id:
            filename += f"_{image_id}"
        else:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename += f"_{timestamp}"
        return str(base_path / f"{filename}.png")

    async def execute_tool_call(self, function_call: FunctionCall) -> Dict[str, Any]:
        tool_name = function_call.name
        tool_args = function_call.args if function_call.args is not None else {}
        tool_call_id_from_llm = function_call.id

        logger.info(
            f"Executing tool: {tool_name} with args: {tool_args} (LLM Call ID: {tool_call_id_from_llm})"
        )

        output_data: Dict[str, Any] = {}  # Holds specific results or error details
        status_code = "success"

        try:
            if tool_name == "capture_screen_region":
                x, y, width, height = (
                    tool_args.get("x"),
                    tool_args.get("y"),
                    tool_args.get("width"),
                    tool_args.get("height"),
                )
                if not all(isinstance(arg, int) for arg in [x, y, width, height]):
                    raise ValueError("x, y, width, and height must be integers.")
                if width <= 0 or height <= 0:
                    raise ValueError("width and height must be positive.")

                image_id = str(uuid.uuid4())
                actual_save_path = tool_args.get(
                    "save_path"
                ) or self._get_screenshot_save_path(tool_name, image_id)
                captured_image = screen.capture_screen_region(
                    x, y, width, height, filename=actual_save_path
                )
                self.captured_images_store[image_id] = captured_image
                output_data = {
                    "message": f"Screen region ({x},{y},{width},{height}) captured.",
                    "image_reference_id": image_id,
                    "saved_to": actual_save_path or "Not saved to disk by tool.",
                }
                logger.info(
                    f"Captured screen region. ID: {image_id}. Saved to: {actual_save_path or 'N/A'}"
                )

            elif tool_name == "capture_full_screen":
                image_id = str(uuid.uuid4())
                actual_save_path = tool_args.get(
                    "save_path"
                ) or self._get_screenshot_save_path(tool_name, image_id)
                captured_image = screen.capture_full_screen(filename=actual_save_path)
                self.captured_images_store[image_id] = captured_image
                output_data = {
                    "message": "Full screen captured.",
                    "image_reference_id": image_id,
                    "saved_to": actual_save_path or "Not saved to disk by tool.",
                }
                logger.info(
                    f"Captured full screen. ID: {image_id}. Saved to: {actual_save_path or 'N/A'}"
                )

            elif tool_name == "get_screen_resolution":
                res_width, res_height = screen.get_screen_resolution()
                output_data = {"width": res_width, "height": res_height}
                logger.info(f"Got screen resolution: {res_width}x{res_height}")

            elif tool_name == "mouse_move":
                mouse.mouse_move(
                    tool_args.get("x"),
                    tool_args.get("y"),
                    duration=tool_args.get("duration", 0.25),
                )
                output_data = {
                    "message": f"Mouse moved to ({tool_args.get('x')}, {tool_args.get('y')})."
                }

            elif tool_name == "mouse_click":
                mouse.mouse_click(
                    x=tool_args.get("x"),
                    y=tool_args.get("y"),
                    button=tool_args.get("button", "left"),
                    clicks=tool_args.get("clicks", 1),
                    interval=tool_args.get("interval", 0.1),
                )
                output_data = {"message": "Mouse click performed."}

            elif tool_name == "mouse_drag":
                mouse.mouse_drag(
                    start_x=tool_args.get("start_x"),
                    start_y=tool_args.get("start_y"),
                    end_x=tool_args.get("end_x"),
                    end_y=tool_args.get("end_y"),
                    duration=tool_args.get("duration", 0.5),
                    button=tool_args.get("button", "left"),
                )
                output_data = {"message": "Mouse drag performed."}

            elif tool_name == "mouse_scroll":
                mouse.mouse_scroll(
                    amount=tool_args.get("amount"),
                    x=tool_args.get("x"),
                    y=tool_args.get("y"),
                )
                output_data = {
                    "message": f"Mouse scrolled by {tool_args.get('amount')} units."
                }

            elif tool_name == "keyboard_type":
                keyboard.keyboard_type(
                    tool_args.get("text"), interval=tool_args.get("interval", 0.01)
                )
                output_data = {"message": f"Typed text: '{tool_args.get('text')}'."}

            elif tool_name == "keyboard_press_key":
                keyboard.keyboard_press_key(tool_args.get("key_name"))
                output_data = {
                    "message": f"Key(s) '{tool_args.get('key_name')}' pressed."
                }

            elif tool_name == "keyboard_hotkey":
                keys_arg = tool_args.get("keys")
                if not isinstance(keys_arg, list) or not keys_arg:
                    raise ValueError("'keys' must be a non-empty list.")
                keyboard.keyboard_hotkey(keys_arg)
                output_data = {"message": f"Hotkey '{'+'.join(keys_arg)}' pressed."}

            elif tool_name == "analyze_image_with_vision_model":
                image_ref_id = tool_args.get("image_reference_id")
                if not image_ref_id:
                    raise ValueError("image_reference_id is required.")
                image_to_analyze = self.captured_images_store.get(image_ref_id)
                if not image_to_analyze:
                    output_data = {
                        "error_type": "ImageNotFoundError",
                        "details": f"Image ID '{image_ref_id}' not found.",
                    }
                    status_code = "error"
                else:
                    analysis_result = await asyncio.to_thread(
                        moondream_interaction.analyze_image_with_moondream,
                        image_to_analyze,
                        tool_args.get("prompt_text"),
                    )
                    if "error" in analysis_result:
                        output_data = {
                            "error_type": "VisionModelError",
                            "details": analysis_result["error"],
                        }
                        status_code = "error"
                    else:
                        output_data = analysis_result.get(
                            "data", {"message": "Analysis complete."}
                        )
                        if not isinstance(output_data, dict):
                            output_data = {"text_response": str(output_data)}

            elif tool_name == "find_text_on_screen_and_click":
                text_to_find, click_button, occurrence = (
                    tool_args.get("text_to_find"),
                    tool_args.get("click_button", "left"),
                    tool_args.get("occurrence", 1),
                )
                if not text_to_find:
                    raise ValueError("text_to_find is required.")

                full_screen_image = screen.capture_full_screen()
                if not full_screen_image:
                    raise RuntimeError("Failed to capture screen.")

                ocr_data_list = await asyncio.to_thread(
                    ocr_service.get_text_and_bounding_boxes, full_screen_image
                )
                if not ocr_data_list:
                    output_data = {
                        "error_type": "OCRError",
                        "details": "OCR found no text or service failed.",
                    }
                    status_code = "error"
                else:
                    found = [
                        item
                        for item in ocr_data_list
                        if text_to_find.lower() in item.get("text", "").lower()
                    ]
                    if not found or occurrence <= 0 or occurrence > len(found):
                        err_msg = f"Text '{text_to_find}' (occurrence {occurrence}) not found. Found {len(found)} matches."
                        output_data = {
                            "error_type": "TextNotFoundError",
                            "details": err_msg,
                        }
                        status_code = "error"
                    else:
                        item = found[occurrence - 1]
                        x, y, w, h = (
                            item["left"],
                            item["top"],
                            item["width"],
                            item["height"],
                        )
                        cx, cy = x + w // 2, y + h // 2
                        mouse.mouse_click(x=cx, y=cy, button=click_button)
                        total_matches = len(found)
                        message = (
                            f"Found '{item['text']}' (occurrence {occurrence} of {total_matches} total matches) "
                            f"and clicked at ({cx},{cy})."
                        )
                        output_data = {
                            "message": message,
                            "clicked_text": item["text"],
                            "clicked_at_x": cx,
                            "clicked_at_y": cy,
                            "bounding_box": {"x": x, "y": y, "width": w, "height": h},
                            "ocr_confidence": item["conf"],
                            "total_matches_found": total_matches,
                        }

            elif tool_name == "list_windows":
                titles = window_manager.list_windows(tool_args.get("title_filter"))
                output_data = {"window_titles": titles, "count": len(titles)}

            elif tool_name == "get_active_window_title":
                active_title = window_manager.get_active_window_title()
                if active_title is not None:
                    output_data = {"active_window_title": active_title}
                else:
                    output_data = {
                        "error_type": "WindowError",
                        "details": "Could not get active window title.",
                    }
                    status_code = "error"

            elif tool_name == "focus_window":
                title = tool_args.get("title")
                if not title:
                    raise ValueError("'title' is required.")
                success = await asyncio.to_thread(window_manager.focus_window, title)
                output_data = {
                    "message": f"Focus attempt on '{title}'.",
                    "focused": success,
                }
                if not success:
                    logger.warning(
                        f"Focus attempt for '{title}' failed or window not found."
                    )

            elif tool_name == "get_window_geometry":
                title = tool_args.get("title")
                if not title:
                    raise ValueError("'title' is required.")
                geometry = window_manager.get_window_geometry(title)
                if geometry:
                    output_data = geometry
                else:
                    output_data = {
                        "error_type": "WindowNotFoundError",
                        "details": f"Window '{title}' not found for geometry.",
                    }
                    status_code = "error"

            elif tool_name == "list_directory":
                path = tool_args.get("path")
                if not path:
                    raise ValueError("'path' is required.")
                list_result = file_system.list_directory(path)  # This is synchronous
                if "error" in list_result:
                    status_code = "error"
                    output_data = {
                        "error_type": "FileSystemError",
                        "details": list_result["error"],
                    }
                else:
                    output_data = list_result

            elif tool_name == "read_text_file":
                path, max_chars = tool_args.get("path"), tool_args.get("max_chars")
                if not path:
                    raise ValueError("'path' is required.")
                read_result = await asyncio.to_thread(
                    file_system.read_text_file, path, max_chars
                )
                if "error" in read_result:
                    status_code = "error"
                    output_data = {
                        "error_type": "FileSystemError",
                        "details": read_result["error"],
                    }
                    if (
                        "warning" in read_result
                    ):  # Preserve warning if error also occurs
                        output_data["warning"] = read_result["warning"]
                else:
                    output_data = read_result

            # --- New File System Tool Dispatching ---
            elif tool_name == "write_text_file":
                path = tool_args.get("path")
                content = tool_args.get("content")
                overwrite = tool_args.get("overwrite", False) # Default to False if not provided
                if not path or content is None: # content can be empty string, so check for None
                    raise ValueError("'path' and 'content' are required for write_text_file.")
                output_data = await asyncio.to_thread(file_system.write_text_file, path, content, overwrite)

            elif tool_name == "append_text_to_file":
                path = tool_args.get("path")
                content = tool_args.get("content")
                if not path or content is None:
                    raise ValueError("'path' and 'content' are required for append_text_to_file.")
                output_data = await asyncio.to_thread(file_system.append_text_to_file, path, content)

            elif tool_name == "create_directory":
                path = tool_args.get("path")
                if not path:
                    raise ValueError("'path' is required for create_directory.")
                output_data = await asyncio.to_thread(file_system.create_directory, path)

            elif tool_name == "delete_file_or_directory":
                path = tool_args.get("path")
                if not path:
                    raise ValueError("'path' is required for delete_file_or_directory.")
                # Potentially add a confirmation step here if LLM could be asked
                # For now, directly execute. User of this tool must be cautious.
                logger.warning(f"Executing delete_file_or_directory for path: {path}. This is a destructive operation.")
                output_data = await asyncio.to_thread(file_system.delete_file_or_directory, path)

            elif tool_name == "move_file_or_directory":
                source_path = tool_args.get("source_path")
                destination_path = tool_args.get("destination_path")
                if not source_path or not destination_path:
                    raise ValueError("'source_path' and 'destination_path' are required for move_file_or_directory.")
                output_data = await asyncio.to_thread(file_system.move_file_or_directory, source_path, destination_path)

            elif tool_name == "copy_file_or_directory":
                source_path = tool_args.get("source_path")
                destination_path = tool_args.get("destination_path")
                if not source_path or not destination_path:
                    raise ValueError("'source_path' and 'destination_path' are required for copy_file_or_directory.")
                output_data = await asyncio.to_thread(file_system.copy_file_or_directory, source_path, destination_path)

            elif tool_name == "get_file_properties":
                path = tool_args.get("path")
                if not path:
                    raise ValueError("'path' is required for get_file_properties.")
                output_data = await asyncio.to_thread(file_system.get_file_properties, path)

            # --- End of New File System Tool Dispatching ---

            # --- Application Management Tool Dispatching ---
            elif tool_name == "start_application":
                app_path_or_name = tool_args.get("application_path_or_name")
                app_args = tool_args.get("arguments") # This can be None
                if not app_path_or_name:
                    raise ValueError("'application_path_or_name' is required.")
                # Import app_manager locally to avoid circular dependencies if it grows
                from . import app_manager
                output_data = await asyncio.to_thread(app_manager.start_application, app_path_or_name, app_args)

            elif tool_name == "get_running_processes":
                from . import app_manager
                output_data = await asyncio.to_thread(app_manager.get_running_processes)

            elif tool_name == "close_application_by_pid":
                pid = tool_args.get("pid")
                force = tool_args.get("force", False)
                if pid is None: # PID 0 can be valid in some contexts, but usually not for user apps
                    raise ValueError("'pid' must be a valid integer.")
                try:
                    pid_int = int(pid)
                except ValueError:
                    raise ValueError("'pid' must be an integer.")
                from . import app_manager
                output_data = await asyncio.to_thread(app_manager.close_application_by_pid, pid_int, force)

            elif tool_name == "close_application_by_title":
                title_substring = tool_args.get("window_title_substring")
                force = tool_args.get("force", False)
                if not title_substring:
                    raise ValueError("'window_title_substring' is required.")
                from . import app_manager
                output_data = await asyncio.to_thread(app_manager.close_application_by_title, title_substring, force)

            # --- End of Application Management Tool Dispatching ---

            # --- System Information & Control Tool Dispatching ---
            elif tool_name == "get_clipboard_text":
                from . import system_control # Local import
                output_data = await asyncio.to_thread(system_control.get_clipboard_text)

            elif tool_name == "set_clipboard_text":
                text_to_set = tool_args.get("text")
                if text_to_set is None: # Allow empty string, but not None
                    raise ValueError("'text' argument is required for set_clipboard_text.")
                from . import system_control
                output_data = await asyncio.to_thread(system_control.set_clipboard_text, text_to_set)

            elif tool_name == "get_system_volume":
                from . import system_control
                output_data = await asyncio.to_thread(system_control.get_system_volume)

            elif tool_name == "set_system_volume":
                level = tool_args.get("level")
                if level is None:
                    raise ValueError("'level' argument (0-100) is required for set_system_volume.")
                try:
                    level_int = int(level)
                except ValueError:
                    raise ValueError("'level' must be an integer between 0 and 100.")
                if not (0 <= level_int <= 100):
                    raise ValueError("'level' must be between 0 and 100.")
                from . import system_control
                output_data = await asyncio.to_thread(system_control.set_system_volume, level_int)

            elif tool_name == "lock_windows_session":
                from . import system_control
                output_data = await asyncio.to_thread(system_control.lock_windows_session)

            elif tool_name == "shutdown_windows_system":
                mode = tool_args.get("mode", "shutdown")
                force = tool_args.get("force", False)
                delay_seconds = tool_args.get("delay_seconds", 0)
                if not isinstance(mode, str) or mode not in ["shutdown", "restart", "logoff"]:
                    raise ValueError("Invalid 'mode'. Must be 'shutdown', 'restart', or 'logoff'.")
                if not isinstance(force, bool):
                    raise ValueError("'force' must be a boolean.")
                if not isinstance(delay_seconds, int) or delay_seconds < 0:
                    raise ValueError("'delay_seconds' must be a non-negative integer.")

                from . import system_control
                logger.warning(f"Executing shutdown_windows_system: mode={mode}, force={force}, delay={delay_seconds}s. This is a system-altering operation.")
                output_data = await asyncio.to_thread(system_control.shutdown_windows_system, mode, force, delay_seconds)

            # --- End of System Information & Control Tool Dispatching ---

            # --- Basic Web Interaction Tool Dispatching ---
            elif tool_name == "open_url_in_default_browser":
                url_to_open = tool_args.get("url")
                if not url_to_open:
                    raise ValueError("'url' argument is required.")
                from . import web_interaction # Local import
                output_data = await asyncio.to_thread(web_interaction.open_url_in_default_browser, url_to_open)

            # --- End of Basic Web Interaction Tool Dispatching ---


            # --- Code Editor Tool Dispatching (Phase 2) ---
            elif tool_name == "show_code_in_editor":
                code_content = tool_args.get("code_content")
                language = tool_args.get("language", "python") # Default to python
                window_title = tool_args.get("window_title") # Can be None

                if code_content is None: # Must have content
                    raise ValueError("'code_content' is required for show_code_in_editor.")

                from . import code_editor # Local import
                # This function is designed to run Tkinter in a separate thread and return quickly.
                # So, direct await asyncio.to_thread might not be strictly necessary if the function
                # itself manages the threading correctly for non-blocking behavior.
                # However, to be consistent with other tools that might do heavy work,
                # using to_thread is a safe pattern.
                output_data = await asyncio.to_thread(
                    code_editor.display_code_in_popup,
                    code_content,
                    language=language,
                    window_title=window_title if window_title else f"{language.capitalize()} Code Viewer"
                )
            # --- End of Code Editor Tool Dispatching ---

            elif tool_name == "get_contextual_code_info":
                code_text = tool_args.get("code_text")
                line_number = tool_args.get("line_number")
                column_number = tool_args.get("column_number") # Optional
                language = tool_args.get("language", "python") # Default to python

                if not code_text or line_number is None:
                    raise ValueError("'code_text' and 'line_number' are required.")
                try:
                    line_number_int = int(line_number)
                except ValueError:
                    raise ValueError("'line_number' must be an integer.")

                column_number_int = None
                if column_number is not None:
                    try:
                        column_number_int = int(column_number)
                    except ValueError:
                        raise ValueError("'column_number' must be an integer if provided.")

                if language != "python":
                    output_data = {"status": "error", "message": f"Language '{language}' is not supported for contextual code info. Only 'python' is currently supported."}
                else:
                    from . import code_parser # Local import
                    output_data = await asyncio.to_thread(
                        code_parser.find_contextual_structure,
                        code_text,
                        line_number_int,
                        column_number_int
                    )

            else:
                output_data = {
                    "error_type": "ToolNotFound",
                    "details": f"Tool '{tool_name}' is not recognized.",
                }
                status_code = "error"
                logger.warning(f"Unrecognized tool call: {tool_name}")

        except (
            ValueError
        ) as ve:  # Specifically for argument validation errors raised by dispatcher
            status_code = "error"
            output_data = {"error_type": "ToolArgumentError", "details": str(ve)}
            logger.error(
                f"ToolArgumentError for {tool_name} with args {tool_args}: {ve}",
                exc_info=True,
            )
        except (
            Exception
        ) as e:  # Catch-all for other unexpected errors during tool execution
            status_code = "error"
            output_data = {"error_type": e.__class__.__name__, "details": str(e)}
            logger.error(
                f"Error executing tool {tool_name} with args {tool_args}: {e}",
                exc_info=True,
            )

        final_response_payload = {"status": status_code}
        final_response_payload.update(
            output_data
        )  # Merge tool-specific results or error details

        return {
            "id": tool_call_id_from_llm,
            "name": tool_name,
            "response": final_response_payload,
        }


if __name__ == "__main__":

    async def run_example():
        dispatcher = DesktopToolDispatcher()
        fc_res = FunctionCall(name="get_screen_resolution", args={})
        result_res = await dispatcher.execute_tool_call(fc_res)
        print(f"Result for {fc_res.name}: {json.dumps(result_res, indent=2)}")
        try:
            res_w, res_h = screen.get_screen_resolution()
            if res_w > 200 and res_h > 200:
                fc_capture = FunctionCall(
                    name="capture_screen_region",
                    args={
                        "x": 0,
                        "y": 0,
                        "width": 200,
                        "height": 200,
                        "save_path": "example_capture.png",
                    },
                )
                result_capture = await dispatcher.execute_tool_call(fc_capture)
                print(
                    f"Result for {fc_capture.name}: {json.dumps(result_capture, indent=2)}"
                )
                if result_capture["response"]["status"] == "success":
                    image_id = result_capture["response"].get("image_reference_id")
                    if image_id:
                        fc_analyze = FunctionCall(
                            name="analyze_image_with_vision_model",
                            args={
                                "prompt_text": "What is in this image?",
                                "image_reference_id": image_id,
                            },
                        )
                        result_analyze = await dispatcher.execute_tool_call(fc_analyze)
                        print(
                            f"Result for {fc_analyze.name}: {json.dumps(result_analyze, indent=2)}"
                        )
                    else:
                        print(
                            "Skipping analyze_image: No image_reference_id from capture."
                        )
            else:
                print("Screen resolution too small for capture_screen_region example.")
        except Exception as e:
            print(
                f"Could not run screen capture examples (may need a display server): {e}"
            )

        print("\n--- File System Tool Examples ---")
        from pathlib import Path  # Ensure Path is imported for the example

        Path("./dummy_test_dir_disp").mkdir(exist_ok=True)
        Path("./dummy_test_dir_disp/sample.txt").write_text("Hello Dispatcher!")

        fc_ls = FunctionCall(
            name="list_directory", args={"path": "./dummy_test_dir_disp"}
        )
        res_ls = await dispatcher.execute_tool_call(fc_ls)
        print(f"Result for {fc_ls.name}: {json.dumps(res_ls, indent=2)}")

        fc_read = FunctionCall(
            name="read_text_file", args={"path": "./dummy_test_dir_disp/sample.txt"}
        )
        res_read = await dispatcher.execute_tool_call(fc_read)
        print(f"Result for {fc_read.name}: {json.dumps(res_read, indent=2)}")

        Path("./dummy_test_dir_disp/sample.txt").unlink(missing_ok=True)
        Path("./dummy_test_dir_disp").rmdir()

    asyncio.run(run_example())
