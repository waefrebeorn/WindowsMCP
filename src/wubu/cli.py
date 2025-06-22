# WuBu Command Line Interface (CLI) - Enhanced
# Main entry point for interacting with WuBu.

import argparse
import asyncio
import sys
import os
import time
from pathlib import Path
import logging # For consistent logging

# Third-party for UI and history
from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.formatted_text import HTML
from rich.panel import Panel
# console object will be used for Rich output, similar to main.py's usage
from rich.console import Console as RichConsole # Explicit import

# WuBu module imports
try:
    from .core.engine import WuBuEngine
    from .utils.resource_loader import load_config
    from .ui.wubu_ui import WuBuUI # WuBu's own UI for engine messages
    # Desktop tools and context provider (ported from main.py's logic)
    from desktop_tools.context_provider import ContextProvider
    from desktop_tools import voice_input, voice_output # For voice I/O
except ImportError as e:
    print(f"CRITICAL ImportError in wubu/cli.py: {e}. Check PYTHONPATH or if running as module from project root.")
    print("Ensure 'desktop_tools' is accessible from 'src/wubu/' (e.g. if src is in PYTHONPATH).")
    # This structure assumes 'desktop_tools' is a sibling of 'wubu' inside 'src',
    # or that 'src' is added to sys.path allowing `from desktop_tools...`
    # Let's try to adjust sys.path for common dev scenario where src/ is the root for these packages
    # current_script_dir = os.path.dirname(os.path.abspath(__file__)) # .../src/wubu
    # src_level_dir = os.path.dirname(current_script_dir) # .../src
    # if src_level_dir not in sys.path:
    #    sys.path.insert(0, src_level_dir)
    # print(f"Temporarily added {src_level_dir} to sys.path. Attempting re-imports...")
    # from desktop_tools.context_provider import ContextProvider
    # from desktop_tools import voice_input, voice_output
    raise # Re-raise to make the problem visible and halt if imports still fail


# --- Global Console for Rich Output ---
console = RichConsole()
logger = logging.getLogger("wubu_cli") # Logger for this CLI file
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s")


async def _process_user_command_with_context(
    user_input_str: str,
    wubu_engine: WuBuEngine,
    context_provider: ContextProvider | None,
    is_test_mode: bool = False
) -> bool:
    """
    Gathers context, prepares the command, and sends it to WuBuEngine.
    Returns True if successful, False otherwise.
    """
    if not user_input_str.strip() and not is_test_mode: # Allow empty test commands if needed by test suite
        return True # No actual command from user

    final_llm_input_message = user_input_str
    context_gathered_for_log = "No context provider or no relevant context found."

    if context_provider:
        try:
            gathered_context = context_provider.gather_context(user_query=user_input_str)
            context_string_parts = []
            context_summary_parts_log = []

            if gathered_context.get("current_file"):
                cf = gathered_context["current_file"]
                relative_path = cf['path']
                context_string_parts.append(f"Current File: {relative_path}\nContent:\n{cf['content']}")
                context_summary_parts_log.append(f"CurrentFile:{relative_path}")


            if gathered_context.get("referenced_files"):
                for rf_idx, rf in enumerate(gathered_context["referenced_files"]):
                    # Avoid duplicating current file if it was also @-referenced
                    if gathered_context.get("current_file") and rf['path'] == gathered_context["current_file"]['path'] and rf_idx == 0:
                        continue
                    relative_path_rf = rf['path']
                    context_string_parts.append(f"Referenced File: {relative_path_rf}\nContent:\n{rf['content']}")
                    context_summary_parts_log.append(f"RefFile:{relative_path_rf}")

            # Simplified open files context for now to avoid too much verbosity
            if gathered_context.get("open_files"):
                 open_files_context = ", ".join([of['path'] for of in gathered_context["open_files"][:2]]) # Max 2 other open files
                 if open_files_context:
                    context_string_parts.append(f"Other Open Files (Paths): {open_files_context}")
                    context_summary_parts_log.append(f"OpenFiles:{open_files_context}")


            if context_string_parts:
                context_header = "Relevant Context Provided:\n" + "\n\n".join(context_string_parts) + "\n\nUser Query: "
                final_llm_input_message = context_header + user_input_str
                context_gathered_for_log = "; ".join(context_summary_parts_log)

            # Update editor state for next turn if @-references were used
            if gathered_context.get("referenced_files"):
                first_ref_file_info = gathered_context["referenced_files"][0]
                current_open_rel_paths = [str(p.relative_to(context_provider.project_root)) for p in context_provider.open_file_paths if context_provider.project_root]
                next_turn_open_files = list(set(
                    [rf['path'] for rf in gathered_context.get("referenced_files", [])] + current_open_rel_paths
                ))
                if context_provider.project_root: # Ensure project_root is valid
                    context_provider.update_editor_state(
                        current_file_rel_path=first_ref_file_info['path'],
                        open_files_rel_paths=next_turn_open_files
                    )
                    logger.info(f"WuBuCLI: ContextProvider editor state updated. Current: '{first_ref_file_info['path']}'.")


        except Exception as e:
            logger.error(f"WuBuCLI: Error during context gathering: {e}", exc_info=True)
            console.print(Panel(f"[bold red]Error gathering context: {e}. Proceeding without extra context.[/bold red]", title="[red]Context Error[/red]"))
            # final_llm_input_message remains user_input_str

    logger.info(f"WuBuCLI: Processing command with context summary: [{context_gathered_for_log}]. Full input length: {len(final_llm_input_message)}")

    # Send the potentially context-augmented command to WuBuEngine
    # WuBuEngine's LLMProcessor will then handle actual LLM interaction.
    try:
        wubu_engine.process_text_command(final_llm_input_message)
        return True
    except Exception as e:
        logger.error(f"WuBuCLI: Error during wubu_engine.process_text_command: {e}", exc_info=True)
        console.print(Panel(f"[bold red]Error processing command in WuBu Engine: {e}[/bold red]", title="[red]Engine Error[/red]"))
        return False


async def async_main_cli():
    """Async main function for the WuBu CLI, incorporating features from old main.py."""
    load_dotenv() # Load .env file if present

    parser = argparse.ArgumentParser(description="WuBu AI Assistant CLI.")
    parser.add_argument("--config", type=str, default="wubu_config.yaml", help="Path to WuBu config file.")
    parser.add_argument("--text-input", type=str, help="Send a single text command to WuBu and exit.")
    # Voice related arguments from old main.py
    parser.add_argument("--voice", action="store_true", help="Enable voice input mode.")
    parser.add_argument("--no-voice-input", action="store_true", help="Explicitly disable voice input (overrides --voice if both set).")
    # LLM provider/model override (might be better handled purely by config, but kept for parity)
    # Note: WuBuEngine/LLMProcessor use config; these args would need to modify config at runtime if used.
    # For now, these are parsed but not directly plumbed to override LLMProcessor's config choice.
    parser.add_argument("--llm_provider_override", type=str, choices=["gemini", "ollama"], help="Override LLM provider from config.")
    parser.add_argument("--ollama_model_override", type=str, help="Override Ollama model from config.")
    # Test execution arguments from old main.py
    parser.add_argument("--test_command", type=str, help="Execute a single test command and exit.")
    parser.add_argument("--test_file", type=str, help="Path to a file with test commands (one per line).")
    args = parser.parse_args()

    console.print(Panel("[bold magenta]WuBu AI Assistant CLI Initializing...[/bold magenta]", title="[white]WuBu Startup[/white]"))

    config_data = load_config(args.config)
    if not config_data:
        console.print(Panel(f"[bold red]Error: Could not load WuBu configuration from '{args.config}'. Exiting.[/bold red]", title="[red]Config Error[/red]"))
        sys.exit(1)
    logger.info(f"WuBu CLI: Configuration loaded from '{args.config}'.")

    # Initialize ContextProvider (from old main.py)
    context_provider = None
    try:
        project_root_for_context = config_data.get('context_provider',{}).get('project_root', ".") # Configurable root
        if os.path.exists(project_root_for_context):
            console.print(f"[dim]WuBuCLI: Initializing ContextProvider for project root: {os.path.abspath(project_root_for_context)}...[/dim]")
            context_provider = ContextProvider(project_root_for_context)
            logger.info(f"WuBuCLI: ContextProvider initialized for root: {context_provider.project_root.resolve()}")
        else:
            logger.warning(f"WuBuCLI: project_root for ContextProvider '{project_root_for_context}' not found. Context features may be limited.")
            console.print(Panel(f"[yellow]Warning: ContextProvider project_root '{project_root_for_context}' not found. Using default behavior.[/yellow]", title="[yellow]Context Warning[/yellow]"))
            context_provider = ContextProvider(".") # Fallback to current dir

    except Exception as e:
        logger.error(f"WuBuCLI: Failed to initialize ContextProvider: {e}", exc_info=True)
        console.print(Panel(f"[bold red]Error initializing ContextProvider: {e}. Context features disabled.[/bold red]", title="[red]Context Error[/red]"))
        # context_provider remains None or could be a dummy version

    wubu_engine: WuBuEngine | None = None
    try:
        logger.info("WuBuCLI: Initializing WuBu Core Engine...")
        wubu_engine = WuBuEngine(config=config_data)
        logger.info("WuBuCLI: WuBu Core Engine initialized.")
    except Exception as e:
        logger.critical(f"WuBuCLI: Fatal Error initializing WuBu Core Engine: {e}", exc_info=True)
        console.print(Panel(f"[bold red]Fatal Error: Could not initialize WuBu Core Engine: {e}[/bold red]", title="[red]Engine Init Error[/red]"))
        sys.exit(1)

    # Initialize WuBuUI (for engine messages)
    wubu_console_ui = WuBuUI(wubu_core_engine=wubu_engine)
    wubu_engine.set_ui(wubu_console_ui)
    wubu_console_ui.start() # Start its message processing loop
    time.sleep(0.2)
    wubu_console_ui.display_message("STATUS_UPDATE", "WuBu CLI Ready.")


    # Handle direct text input from args (from original cli.py)
    if args.text_input:
        console.print(Panel(f"[cyan]Processing direct input to WuBu:[/cyan] '{args.text_input}'"))
        await _process_user_command_with_context(args.text_input, wubu_engine, context_provider, is_test_mode=True)
        # Crude wait for TTS, if any, from engine
        time.sleep(config_data.get('tts', {}).get('estimated_max_speech_duration', 3))
        console.print(Panel("[cyan]Direct input processed. Exiting WuBu CLI.[/cyan]"))
        # Proper shutdown
        if wubu_console_ui.is_running: wubu_console_ui.stop()
        if wubu_engine: wubu_engine.shutdown()
        sys.exit(0)

    # Handle test_command or test_file from args (ported from old main.py)
    if args.test_command:
        console.print(Panel(f"[cyan]Executing Test Command:[/cyan] {args.test_command}"))
        await _process_user_command_with_context(args.test_command, wubu_engine, context_provider, is_test_mode=True)
        # Wait for TTS
        time.sleep(config_data.get('tts', {}).get('estimated_max_speech_duration', 3))
        console.print(Panel("[cyan]Test command finished. Exiting WuBu CLI.[/cyan]"))
    elif args.test_file:
        console.print(Panel(f"[yellow]Test File Mode:[/yellow] '{args.test_file}'"))
        try:
            with open(args.test_file, "r") as f:
                commands = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
            total_cmds = len(commands)
            console.print(f"[info]Found {total_cmds} commands in test file.[/info]")
            for i, cmd_str in enumerate(commands):
                console.print(f"\n[bold cyan]>>> Executing from file ({i+1}/{total_cmds}):[/bold cyan] {cmd_str}")
                await _process_user_command_with_context(cmd_str, wubu_engine, context_provider, is_test_mode=True)
                if i < total_cmds - 1:
                    await asyncio.sleep(config_data.get('testing',{}).get('delay_between_file_commands_sec', 2))
            console.print(f"\n[bold green]>>> Test file processing complete. <<<[/bold green]")
        except FileNotFoundError:
            console.print(Panel(f"[bold red]Error: Test file '{args.test_file}' not found.[/bold red]", title="[red]Test File Error[/red]"))
        except Exception as e:
            logger.error(f"WuBuCLI: Error processing test file '{args.test_file}': {e}", exc_info=True)
            console.print(Panel(f"[bold red]Unexpected error processing test file: {e}[/bold red]", title="[red]Test File Error[/red]"))

    if args.test_command or args.test_file: # Exit after test modes
        if wubu_console_ui.is_running: wubu_console_ui.stop()
        if wubu_engine: wubu_engine.shutdown()
        sys.exit(0)


    # Interactive Loop (ported and adapted from old main.py)
    history_file_path_str = config_data.get('cli',{}).get('history_file_path', ".wubu_cli_history") # Default in project root
    history_file = Path(history_file_path_str)
    try:
        if not history_file.parent.exists(): history_file.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        console.print(Panel(f"[yellow]Warning: Could not create/access history file dir '{history_file.parent}': {e}[/yellow]", title="[yellow]History Warning[/yellow]"))

    prompt_session = PromptSession(history=FileHistory(str(history_file)))
    enable_voice_mode = args.voice and not args.no_voice_input

    console.print(Panel(f"[green]Welcome to WuBu Interactive Mode![/green] Type 'exit' or Ctrl+D to quit.",
                        subtitle="Voice mode is " + ("[bold green]ENABLED[/bold green]" if enable_voice_mode else "[bold yellow]DISABLED[/bold yellow]")))
    if enable_voice_mode:
        console.print("[cyan]Say 'Hey WuBu' or your activation phrase, then your command.[/cyan]")

    try:
        while True:
            user_input_str = ""
            try:
                if enable_voice_mode:
                    console.print("[cyan]WuBu Listening (Ctrl+C to cancel recording)...[/cyan]")
                    record_duration = config_data.get('asr',{}).get('voice_record_duration', 5)
                    whisper_model = config_data.get('asr',{}).get('whisper_model_name', voice_input.DEFAULT_WHISPER_MODEL)
                    audio_device = config_data.get('asr',{}).get('audio_input_device_name_or_id')

                    temp_audio_file = await asyncio.to_thread(voice_input.record_audio, duration_seconds=record_duration, device=audio_device)
                    if temp_audio_file:
                        console.print(f"[dim]Audio recorded to {temp_audio_file}, transcribing with Whisper ({whisper_model})...[/dim]")
                        transcribed_text = await asyncio.to_thread(voice_input.transcribe_audio_with_whisper, temp_audio_file, whisper_model)
                        os.remove(temp_audio_file) # Clean up temp file

                        if transcribed_text:
                            console.print(HTML(f"<ansigreen><b>You (voice): </b></ansigreen>{transcribed_text}"))
                            activation_phrases = config_data.get('asr',{}).get('activation_phrases', ["hey wubu", "wubu"])
                            normalized_input = transcribed_text.lower().strip()
                            activated = False
                            actual_command = ""
                            for phrase in activation_phrases:
                                if normalized_input.startswith(phrase.lower()):
                                    activated = True
                                    actual_command = transcribed_text[len(phrase):].strip()
                                    console.print(HTML(f"<ansimagenta><b>WuBu Activated! Command: </b></ansimagenta>{actual_command if actual_command else '(Listening for command...)'}"))
                                    break
                            if not activated:
                                console.print("[yellow]Activation phrase not detected. Please start with 'Hey WuBu' or similar.[/yellow]")
                                continue
                            user_input_str = actual_command
                            if not user_input_str: # Only activation phrase said
                                user_input_str = await asyncio.to_thread(prompt_session.prompt, HTML("<ansiblue><b>WuBu listens: </b></ansiblue>"))
                        else:
                            console.print("[yellow]Transcription failed or no speech. Try typing or speak again.[/yellow]")
                            continue
                    else:
                        console.print("[yellow]Audio recording failed. Try typing.[/yellow]")
                        continue
                else: # Text input mode
                    user_input_str = await asyncio.to_thread(prompt_session.prompt, HTML("<ansiblue><b>You: </b></ansiblue>"))

            except KeyboardInterrupt:
                if enable_voice_mode: console.print("\n[yellow]Voice recording cancelled. Type or try voice again.[/yellow]"); continue
                else: console.print("\n[bold yellow]Exiting WuBu...[/bold yellow]"); break
            except EOFError: console.print("\n[bold yellow]Exiting WuBu (EOF)...[/bold yellow]"); break

            if not user_input_str.strip(): continue
            if user_input_str.strip().lower() == "exit": console.print("[bold yellow]Exiting WuBu...[/bold yellow]"); break

            await _process_user_command_with_context(user_input_str, wubu_engine, context_provider)
            time.sleep(0.1) # Brief pause

    finally:
        console.print(Panel("[bold magenta]Shutting down WuBu CLI...[/bold magenta]"))
        if wubu_console_ui and wubu_console_ui.is_running:
            logger.info("WuBuCLI: Stopping WuBuUI...")
            wubu_console_ui.stop()
        if wubu_engine:
            logger.info("WuBuCLI: Shutting down WuBuEngine...")
            wubu_engine.shutdown()
        console.print(Panel("[bold magenta]WuBu CLI shutdown complete.[/bold magenta]"))

if __name__ == '__main__':
    try:
        asyncio.run(async_main_cli())
    except KeyboardInterrupt:
        console.print("\n[bold red]WuBu CLI interrupted globally. Exiting.[/bold red]")
        logger.info("WuBu CLI interrupted globally by user (Ctrl+C).")
    except SystemExit: # Handles sys.exit calls from argument parsing or init failures
        pass # Error messages should have been printed by the exiting code
    except Exception as e:
        logger.critical(f"WuBuCLI: Unhandled critical exception at top level: {e}", exc_info=True)
        console.print(Panel(f"[bold red]Critical unhandled error in WuBu CLI: {e}[/bold red]", title="[red]FATAL ERROR[/red]"))
