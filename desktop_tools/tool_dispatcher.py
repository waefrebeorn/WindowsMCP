import json
import logging
import asyncio
import uuid
from typing import Any, Dict, NamedTuple
from PIL import Image
from pathlib import Path

# Local module imports
from . import screen, mouse, keyboard, ocr_service, window_manager, file_system
from .moondream_interaction import MoondreamV2 # Import the class

logger = logging.getLogger(__name__)

class FunctionCall(NamedTuple):
    name: str
    args: Dict[str, Any]
    id: str = None

class DesktopToolDispatcher:
    def __init__(self, config_data: dict = None):
        self.captured_images_store: Dict[str, Image.Image] = {}
        self.config_data = config_data if config_data is not None else {}
        self.moondream_vision: MoondreamV2 | None = None

        vision_config = self.config_data.get('vision', {})
        moondream_config = vision_config.get('moondream', {})
        if vision_config.get('enabled', True) and moondream_config.get('enabled', True): # Default to enabled
            try:
                model_id = moondream_config.get('model_id', 'vikhyatk/moondream2')
                revision = moondream_config.get('revision', '2025-06-21') # Keep this updated
                logger.info(f"Initializing MoondreamV2 for dispatcher (model: {model_id}, rev: {revision})")
                self.moondream_vision = MoondreamV2(model_id=model_id, revision=revision)
                logger.info("MoondreamV2 initialized successfully in DesktopToolDispatcher.")
            except Exception as e:
                logger.error(f"Failed to initialize MoondreamV2 in DesktopToolDispatcher: {e}", exc_info=True)
                self.moondream_vision = None
        else:
            logger.info("Vision system or MoondreamV2 specifically disabled in config for DesktopToolDispatcher.")

        logger.info(f"DesktopToolDispatcher initialized. Config loaded: {'Yes' if self.config_data else 'No'}. Moondream enabled: {'Yes' if self.moondream_vision else 'No'}")

    def _get_screenshot_save_path(self, tool_name: str, image_id: str = None) -> str | None:
        dt_config = self.config_data.get('desktop_tools', {})
        screenshot_cfg = dt_config.get('screenshots', {})
        base_path_str = screenshot_cfg.get("save_path", None)

        if not base_path_str:
            logger.debug("Screenshot save path not configured. Screenshots will not be saved by default by tools unless path provided in args.")
            return None

        base_path = Path(base_path_str).expanduser().resolve()
        try:
            base_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating screenshot directory {base_path}: {e}. Screenshots may not be saved.")
            return None

        filename_prefix = tool_name.replace('_', '-')
        timestamp_or_id = image_id if image_id else uuid.uuid4().hex[:8]
        filename = f"{filename_prefix}_{timestamp_or_id}.png"
        return str(base_path / filename)

    async def execute_tool_call(self, function_call: FunctionCall) -> Dict[str, Any]:
        tool_name = function_call.name
        tool_args = function_call.args if function_call.args is not None else {}
        tool_call_id_from_llm = function_call.id

        logger.info(f"Executing tool: {tool_name} with args: {tool_args} (LLM Call ID: {tool_call_id_from_llm})")
        output_data: Dict[str, Any] = {}
        status_code = "success"

        try:
            # ... (other tool cases remain the same) ...
            if tool_name == "capture_screen_region":
                x, y, width, height = (
                    tool_args.get("x"), tool_args.get("y"),
                    tool_args.get("width"), tool_args.get("height"),
                )
                if not all(isinstance(arg, int) for arg in [x, y, width, height]):
                    raise ValueError("x, y, width, and height must be integers.")
                if width <= 0 or height <= 0:
                    raise ValueError("width and height must be positive.")

                image_id = str(uuid.uuid4())
                actual_save_path = tool_args.get("save_path") or self._get_screenshot_save_path(tool_name, image_id)

                captured_image = screen.capture_screen_region(x, y, width, height, filename=actual_save_path)
                self.captured_images_store[image_id] = captured_image # Store PIL image
                output_data = {
                    "message": f"Screen region ({x},{y},{width},{height}) captured.",
                    "image_reference_id": image_id,
                    "saved_to": actual_save_path if actual_save_path else "Not saved to disk (no path configured or provided).",
                }

            elif tool_name == "capture_full_screen":
                image_id = str(uuid.uuid4())
                actual_save_path = tool_args.get("save_path") or self._get_screenshot_save_path(tool_name, image_id)
                captured_image = screen.capture_full_screen(filename=actual_save_path)
                self.captured_images_store[image_id] = captured_image # Store PIL image
                output_data = {
                    "message": "Full screen captured.",
                    "image_reference_id": image_id,
                    "saved_to": actual_save_path if actual_save_path else "Not saved to disk (no path configured or provided).",
                }

            elif tool_name == "analyze_image_with_vision_model":
                image_ref_id = tool_args.get("image_reference_id")
                prompt_text = tool_args.get("prompt_text")
                analysis_type = tool_args.get("analysis_type", "query").lower() # query, caption, detect, point

                if not image_ref_id: raise ValueError("image_reference_id is required.")
                if not prompt_text and analysis_type != "caption": # Caption can work without prompt
                     raise ValueError("prompt_text is required for analysis types other than caption.")

                if not self.moondream_vision:
                    output_data = {"error_type": "VisionModelNotAvailable", "details": "MoondreamV2 vision model is not initialized in dispatcher."}; status_code = "error"
                else:
                    image_to_analyze = self.captured_images_store.get(image_ref_id)
                    if not image_to_analyze:
                        output_data = {"error_type": "ImageNotFoundError", "details": f"Image ID '{image_ref_id}' not found in dispatcher store."}; status_code = "error"
                    else:
                        logger.info(f"Analyzing image_ref_id '{image_ref_id}' with Moondream type '{analysis_type}', prompt: '{prompt_text}'")
                        # Moondream methods are synchronous, so run in thread
                        if analysis_type == "query":
                            vision_result = await asyncio.to_thread(self.moondream_vision.query, image_to_analyze, prompt_text)
                        elif analysis_type == "caption":
                            caption_length = tool_args.get("caption_length", "normal")
                            vision_result = await asyncio.to_thread(self.moondream_vision.caption, image_to_analyze, length=caption_length)
                        elif analysis_type == "detect":
                            # prompt_text here is the object_name for detect
                            vision_result = await asyncio.to_thread(self.moondream_vision.detect, image_to_analyze, prompt_text)
                        elif analysis_type == "point":
                            # prompt_text here is the object_name for point
                            vision_result = await asyncio.to_thread(self.moondream_vision.point, image_to_analyze, prompt_text)
                        else:
                            output_data = {"error_type": "UnsupportedAnalysisType", "details": f"Analysis type '{analysis_type}' not supported."}; status_code = "error"
                            vision_result = None # Ensure it's defined

                        if vision_result: # If an analysis type was matched
                            if "error" in vision_result:
                                output_data = {"error_type": "VisionModelError", "details": vision_result["error"]}; status_code = "error"
                            else: # Success from Moondream method
                                output_data = {"result": vision_result} # Wrap the direct dict result

            # ... (other tool cases remain the same, ensure they are complete) ...
            elif tool_name == "get_screen_resolution":
                res_width, res_height = screen.get_screen_resolution()
                output_data = {"width": res_width, "height": res_height}

            elif tool_name == "mouse_move":
                mouse.mouse_move(tool_args.get("x"), tool_args.get("y"), duration=tool_args.get("duration", 0.25))
                output_data = {"message": f"Mouse moved to ({tool_args.get('x')}, {tool_args.get('y')})."}

            elif tool_name == "mouse_click":
                mouse.mouse_click(x=tool_args.get("x"), y=tool_args.get("y"), button=tool_args.get("button", "left"), clicks=tool_args.get("clicks", 1), interval=tool_args.get("interval", 0.1))
                output_data = {"message": "Mouse click performed."}

            elif tool_name == "mouse_drag":
                mouse.mouse_drag(start_x=tool_args.get("start_x"), start_y=tool_args.get("start_y"), end_x=tool_args.get("end_x"), end_y=tool_args.get("end_y"), duration=tool_args.get("duration", 0.5), button=tool_args.get("button", "left"))
                output_data = {"message": "Mouse drag performed."}

            elif tool_name == "mouse_scroll":
                mouse.mouse_scroll(amount=tool_args.get("amount"), x=tool_args.get("x"), y=tool_args.get("y"))
                output_data = {"message": f"Mouse scrolled by {tool_args.get('amount')} units."}

            elif tool_name == "keyboard_type":
                keyboard.keyboard_type(tool_args.get("text"), interval=tool_args.get("interval", 0.01))
                output_data = {"message": f"Typed text: '{tool_args.get('text')}'."}

            elif tool_name == "keyboard_press_key":
                keyboard.keyboard_press_key(tool_args.get("key_name"))
                output_data = {"message": f"Key(s) '{tool_args.get('key_name')}' pressed."}

            elif tool_name == "keyboard_hotkey":
                keys_arg = tool_args.get("keys")
                if not isinstance(keys_arg, list) or not keys_arg: raise ValueError("'keys' must be a non-empty list.")
                keyboard.keyboard_hotkey(keys_arg)
                output_data = {"message": f"Hotkey '{'+'.join(keys_arg)}' pressed."}

            elif tool_name == "find_text_on_screen_and_click":
                text_to_find = tool_args.get("text_to_find")
                click_button = tool_args.get("click_button", "left")
                occurrence = tool_args.get("occurrence", 1)
                if not text_to_find: raise ValueError("text_to_find is required.")

                full_screen_image = screen.capture_full_screen()
                if not full_screen_image: raise RuntimeError("Failed to capture screen for OCR.")

                ocr_data_list = await asyncio.to_thread(ocr_service.get_text_and_bounding_boxes, full_screen_image)
                if not ocr_data_list:
                    output_data = {"error_type": "OCRError", "details": "OCR found no text or service failed."}; status_code = "error"
                else:
                    found = [item for item in ocr_data_list if text_to_find.lower() in item.get("text", "").lower()]
                    if not found or occurrence <= 0 or occurrence > len(found):
                        output_data = {"error_type": "TextNotFoundError", "details": f"Text '{text_to_find}' (occurrence {occurrence}) not found. Found {len(found)} matches."}; status_code = "error"
                    else:
                        item = found[occurrence - 1]
                        x, y, w, h = item["left"], item["top"], item["width"], item["height"]
                        cx, cy = x + w // 2, y + h // 2
                        mouse.mouse_click(x=cx, y=cy, button=click_button)
                        output_data = {
                            "message": f"Found '{item['text']}' (occurrence {occurrence}/{len(found)}) and clicked at ({cx},{cy}).",
                            "clicked_text": item["text"], "clicked_at_x": cx, "clicked_at_y": cy,
                            "bounding_box": {"x": x, "y": y, "width": w, "height": h},
                            "ocr_confidence": item["conf"], "total_matches_found": len(found),
                        }

            elif tool_name == "list_windows":
                output_data = {"window_titles": window_manager.list_windows(tool_args.get("title_filter"))}

            elif tool_name == "get_active_window_title":
                active_title = window_manager.get_active_window_title()
                if active_title is not None: output_data = {"active_window_title": active_title}
                else: output_data = {"error_type": "WindowError", "details": "Could not get active window title."}; status_code = "error"

            elif tool_name == "focus_window":
                title = tool_args.get("title");
                if not title: raise ValueError("'title' is required.")
                success = await asyncio.to_thread(window_manager.focus_window, title)
                output_data = {"message": f"Focus attempt on '{title}'.", "focused": success}

            elif tool_name == "get_window_geometry":
                title = tool_args.get("title");
                if not title: raise ValueError("'title' is required.")
                geometry = window_manager.get_window_geometry(title)
                if geometry: output_data = geometry
                else: output_data = {"error_type": "WindowNotFoundError", "details": f"Window '{title}' not found."}; status_code = "error"

            elif tool_name == "list_directory":
                path = tool_args.get("path");
                if not path: raise ValueError("'path' is required.")
                list_result = file_system.list_directory(path)
                if "error" in list_result: status_code = "error"; output_data = {"error_type": "FileSystemError", "details": list_result["error"]}
                else: output_data = list_result

            elif tool_name == "read_text_file":
                path, max_chars = tool_args.get("path"), tool_args.get("max_chars")
                if not path: raise ValueError("'path' is required.")
                read_result = await asyncio.to_thread(file_system.read_text_file, path, max_chars)
                if "error" in read_result:
                    status_code = "error"
                    output_data = {"error_type": "FileSystemError", "details": read_result["error"]}
                    if "warning" in read_result: output_data["warning"] = read_result["warning"] # Keep warning if error
                else:
                    output_data = read_result


            elif tool_name == "write_text_file":
                path, content, overwrite = tool_args.get("path"), tool_args.get("content"), tool_args.get("overwrite", False)
                if path is None or content is None: raise ValueError("'path' and 'content' are required.")
                output_data = await asyncio.to_thread(file_system.write_text_file, path, content, overwrite)

            elif tool_name == "append_text_to_file":
                path, content = tool_args.get("path"), tool_args.get("content")
                if path is None or content is None: raise ValueError("'path' and 'content' are required.")
                output_data = await asyncio.to_thread(file_system.append_text_to_file, path, content)

            elif tool_name == "create_folder":
                path = tool_args.get("path");
                if not path: raise ValueError("'path' is required.")
                output_data = await asyncio.to_thread(file_system.create_folder, path)

            elif tool_name == "delete_item":
                path, force = tool_args.get("path"), tool_args.get("force_delete_non_empty_folder", False)
                if not path: raise ValueError("'path' is required.")
                logger.warning(f"Executing delete_item for path: {path} with force: {force}. This is a destructive operation.")
                output_data = await asyncio.to_thread(file_system.delete_item, path, force)

            elif tool_name == "move_or_rename_item":
                source, new_name = tool_args.get("source_path"), tool_args.get("new_path_or_name")
                if not source or not new_name: raise ValueError("'source_path' and 'new_path_or_name' are required.")
                output_data = await asyncio.to_thread(file_system.move_or_rename_item, source, new_name)

            elif tool_name == "copy_item":
                source, dest = tool_args.get("source_path"), tool_args.get("destination_path")
                if not source or not dest: raise ValueError("'source_path' and 'destination_path' are required.")
                output_data = await asyncio.to_thread(file_system.copy_file_or_directory, source, dest)

            elif tool_name == "get_file_properties":
                path = tool_args.get("path");
                if not path: raise ValueError("'path' is required.")
                output_data = await asyncio.to_thread(file_system.get_file_properties, path)

            elif tool_name == "start_application":
                from . import app_manager # Local import
                app_path, app_args = tool_args.get("application_path_or_name"), tool_args.get("arguments")
                if not app_path: raise ValueError("'application_path_or_name' is required.")
                output_data = await asyncio.to_thread(app_manager.start_application, app_path, app_args)

            elif tool_name == "get_running_processes":
                from . import app_manager
                output_data = await asyncio.to_thread(app_manager.get_running_processes)

            elif tool_name == "close_application_by_pid":
                from . import app_manager
                pid, force = tool_args.get("pid"), tool_args.get("force", False)
                if pid is None: raise ValueError("'pid' must be a valid integer.")
                try: pid_int = int(pid)
                except ValueError: raise ValueError("'pid' must be an integer.")
                output_data = await asyncio.to_thread(app_manager.close_application_by_pid, pid_int, force)

            elif tool_name == "close_application_by_title":
                from . import app_manager
                title, force = tool_args.get("window_title_substring"), tool_args.get("force", False)
                if not title: raise ValueError("'window_title_substring' is required.")
                output_data = await asyncio.to_thread(app_manager.close_application_by_title, title, force)

            elif tool_name == "control_active_window":
                from . import window_manager # ensure imported
                action = tool_args.get("action")
                if not action: raise ValueError("'action' is required.")
                if action == "minimize": output_data = {"success": await asyncio.to_thread(window_manager.minimize_active_window)}
                elif action == "maximize": output_data = {"success": await asyncio.to_thread(window_manager.maximize_active_window)}
                elif action == "restore": output_data = {"success": await asyncio.to_thread(window_manager.restore_active_window)}
                elif action == "close": output_data = {"success": await asyncio.to_thread(window_manager.close_active_window)}
                elif action == "get_title":
                    title = await asyncio.to_thread(window_manager.get_active_window_title)
                    if title is not None: output_data = {"active_window_title": title}
                    else: output_data = {"error_type": "WindowError", "details": "Could not get active window title."}; status_code = "error"
                elif action == "get_geometry":
                    active_title = await asyncio.to_thread(window_manager.get_active_window_title)
                    if active_title is not None:
                        geometry = await asyncio.to_thread(window_manager.get_window_geometry, active_title)
                        if geometry: output_data = geometry
                        else: output_data = {"error_type": "WindowError", "details": f"Could not get geometry for active window '{active_title}'."}; status_code = "error"
                    else: output_data = {"error_type": "WindowError", "details": "No active window title for geometry."}; status_code = "error"
                else: raise ValueError(f"Unsupported action '{action}' for control_active_window.")

            elif tool_name == "get_system_information":
                from . import system_monitor
                query, path_arg = tool_args.get("query"), tool_args.get("path")
                if not query: raise ValueError("'query' is required.")
                if query == "cpu_usage": output_data = await asyncio.to_thread(system_monitor.get_cpu_usage)
                elif query == "memory_usage": output_data = await asyncio.to_thread(system_monitor.get_memory_usage)
                elif query == "disk_usage": output_data = await asyncio.to_thread(system_monitor.get_disk_usage, path_arg or "/")
                elif query == "battery_status": output_data = await asyncio.to_thread(system_monitor.get_battery_status)
                else: raise ValueError(f"Unsupported query '{query}'.")

            elif tool_name == "get_clipboard_text":
                from . import system_control
                output_data = await asyncio.to_thread(system_control.get_clipboard_text)
            elif tool_name == "set_clipboard_text":
                from . import system_control
                text = tool_args.get("text")
                if text is None: raise ValueError("'text' is required.")
                output_data = await asyncio.to_thread(system_control.set_clipboard_text, text)
            elif tool_name == "get_system_volume":
                from . import system_control
                output_data = await asyncio.to_thread(system_control.get_system_volume)
            elif tool_name == "set_system_volume":
                from . import system_control
                level = tool_args.get("level")
                if level is None: raise ValueError("'level' (0-100) is required.")
                try: level_int = int(level)
                except ValueError: raise ValueError("'level' must be an integer.")
                if not (0 <= level_int <= 100): raise ValueError("'level' must be 0-100.")
                output_data = await asyncio.to_thread(system_control.set_system_volume, level_int)
            elif tool_name == "lock_windows_session":
                from . import system_control
                output_data = await asyncio.to_thread(system_control.lock_windows_session)
            elif tool_name == "shutdown_windows_system":
                from . import system_control
                mode, force, delay = tool_args.get("mode", "shutdown"), tool_args.get("force", False), tool_args.get("delay_seconds", 0)
                if mode not in ["shutdown", "restart", "logoff"]: raise ValueError("Invalid 'mode'.")
                if not isinstance(force, bool): raise ValueError("'force' must be boolean.")
                if not isinstance(delay, int) or delay < 0: raise ValueError("'delay_seconds' must be non-negative int.")
                logger.warning(f"Executing shutdown_windows_system: mode={mode}, force={force}, delay={delay}s.")
                output_data = await asyncio.to_thread(system_control.shutdown_windows_system, mode, force, delay)

            elif tool_name == "open_url_or_search_web":
                from . import web_interaction
                query, is_search = tool_args.get("query_or_url"), tool_args.get("is_search", False)
                if not query: raise ValueError("'query_or_url' is required.")
                if is_search: output_data = await asyncio.to_thread(web_interaction.search_web, query)
                else: output_data = await asyncio.to_thread(web_interaction.open_url_in_default_browser, query)

            elif tool_name == "show_code_in_editor":
                from . import code_editor
                code, lang, title = tool_args.get("code_content"), tool_args.get("language", "python"), tool_args.get("window_title")
                if code is None: raise ValueError("'code_content' is required.")
                output_data = await asyncio.to_thread(code_editor.display_code_in_popup, code, language=lang, window_title=title or f"{lang.capitalize()} Code")

            elif tool_name == "get_contextual_code_info":
                from . import code_parser
                code, line, col, lang = tool_args.get("code_text"), tool_args.get("line_number"), tool_args.get("column_number"), tool_args.get("language", "python")
                if not code or line is None: raise ValueError("'code_text' and 'line_number' are required.")
                try: line_int = int(line)
                except ValueError: raise ValueError("'line_number' must be an integer.")
                col_int = None
                if col is not None:
                    try: col_int = int(col)
                    except ValueError: raise ValueError("'column_number' must be an integer if provided.")
                if lang != "python":
                    output_data = {"status": "error", "message": f"Language '{lang}' not supported. Only 'python'."}
                    status_code="error"
                else:
                    output_data = await asyncio.to_thread(code_parser.find_contextual_structure, code, line_int, col_int)

            else:
                output_data = {"error_type": "ToolNotFound", "details": f"Tool '{tool_name}' is not recognized by WuBu."}
                status_code = "error"; logger.warning(f"Unrecognized tool call: {tool_name}")

        except ValueError as ve: # Argument validation errors
            status_code = "error"; output_data = {"error_type": "ToolArgumentError", "details": str(ve)}
            logger.error(f"ToolArgumentError for {tool_name} with args {tool_args}: {ve}", exc_info=True)
        except Exception as e:  # Other unexpected errors during tool execution
            status_code = "error"; output_data = {"error_type": e.__class__.__name__, "details": str(e)}
            logger.error(f"Error executing tool {tool_name} with args {tool_args}: {e}", exc_info=True)

        final_response_payload = {"status": status_code}
        # Ensure output_data is a dict before update. If a tool returns a simple string, wrap it.
        if not isinstance(output_data, dict):
            output_data = {"result": output_data} # Default wrap
        final_response_payload.update(output_data)

        return {"id": tool_call_id_from_llm, "name": tool_name, "response": final_response_payload}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s")
    # Example config for testing _get_screenshot_save_path
    test_cfg = {"desktop_tools": {"screenshots": {"save_path": "~/WuBu_Test_Screenshots"}}}

    async def run_example():
        dispatcher = DesktopToolDispatcher(config_data=test_cfg)
        # ... (rest of example usage from previous version, adapted for new dispatcher init) ...
        fc_res = FunctionCall(name="get_screen_resolution", args={}, id="t1")
        result_res = await dispatcher.execute_tool_call(fc_res)
        print(f"Result for {fc_res.name}: {json.dumps(result_res, indent=2)}")

        # Example for screenshot path test
        save_path_example = dispatcher._get_screenshot_save_path("example_tool", "img123")
        print(f"Example screenshot save path: {save_path_example}")
        if save_path_example:
            try:
                Path(save_path_example).touch() # Create dummy file to check path
                print(f"Touched dummy file at {save_path_example}")
            except Exception as e:
                print(f"Could not touch dummy file at {save_path_example}: {e}")


    asyncio.run(run_example())
