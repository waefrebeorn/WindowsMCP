import customtkinter as ctk
import threading # For running LLM calls in a separate thread (now managed by engine)
import os # For path checking in Zonos Dashboard
# from ..core.llm_processor import LLMProcessor # Engine will have this
from ..utils.resource_loader import load_config # Still needed for UI main
from ..tts.base_tts_engine import TTSPlaybackSpeed # For Zonos Dashboard speed control
# from ..core.engine import WuBuEngine # For type hinting, causes circular if engine imports WubuApp for type hint

class WubuApp(ctk.CTk):
    def __init__(self, engine): # Receives WuBuEngine instance
        super().__init__()

        self.engine = engine
        self.config = engine.config # Get config from engine

        # self.llm_processor = None # No longer needed, engine handles it
        # if self.config:
        #     try:
        #         # self.llm_processor = LLMProcessor(config=self.config) # Done by engine
        #         # print("WubuApp: LLMProcessor initialized.")
        #         pass
        #     except Exception as e:
        #         # print(f"WubuApp: Error initializing LLMProcessor: {e}")
        #         pass
        # else:
        #     # print("WubuApp: Config not provided, LLMProcessor not initialized.")
        #     pass

        self.title(f"{self.config.get('wubu_name', 'WuBu')} Assistant GUI")
        self.geometry("800x600")

        # --- Chat History ---
        self.chat_history = [] # To store conversation for context

        # --- Main Frames ---
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill=ctk.BOTH, expand=True, padx=10, pady=10)

        # --- Input Area ---
        self.input_frame = ctk.CTkFrame(self.main_frame)
        self.input_frame.pack(side=ctk.BOTTOM, fill=ctk.X, pady=(0, 10))

        self.prompt_entry = ctk.CTkEntry(self.input_frame, placeholder_text="Type your command or question...")
        self.prompt_entry.pack(side=ctk.LEFT, fill=ctk.X, expand=True, padx=(0, 5))
        self.prompt_entry.bind("<Return>", self.on_send_prompt)

        self.send_button = ctk.CTkButton(self.input_frame, text="Send", width=70, command=self.on_send_prompt)
        self.send_button.pack(side=ctk.LEFT)

        self.mic_button = ctk.CTkButton(self.input_frame, text="🎤", width=50, command=self.on_mic_button_pressed) # Placeholder
        self.mic_button.pack(side=ctk.LEFT, padx=(5,0))


        # --- Chat/Display Area ---
        self.chat_display = ctk.CTkTextbox(self.main_frame, state="disabled", wrap=ctk.WORD)
        self.chat_display.pack(fill=ctk.BOTH, expand=True)


        # --- Status Bar (Placeholder) ---
        self.status_bar_frame = ctk.CTkFrame(self, height=25)
        self.status_bar_frame.pack(side=ctk.BOTTOM, fill=ctk.X, padx=10, pady=(0,5))

        self.status_label = ctk.CTkLabel(self.status_bar_frame, text="Status: Ready")
        self.status_label.pack(side=ctk.LEFT, padx=5)

        self.zonos_dashboard_button = ctk.CTkButton(self.status_bar_frame, text="Zonos TTS Dashboard", command=self.open_zonos_dashboard)
        self.zonos_dashboard_button.pack(side=ctk.RIGHT, padx=5)
        self.zonos_dashboard_window = None


    def open_zonos_dashboard(self):
        if self.zonos_dashboard_window is None or not self.zonos_dashboard_window.winfo_exists():
            # Pass the main app's TTS Engine Manager instance, or the Zonos engine directly
            # This assumes TTSEngineManager is part of WubuApp or accessible
            # For now, let's assume we'll pass the Zonos engine instance if available
            # This part will be cleaner with a central WubuEngine
            # The engine should have tts_manager
            zonos_engine_instance = None
            if self.engine and self.engine.tts_manager:
                from ..tts.tts_engine_manager import ZONOS_LOCAL_ENGINE_ID # Use ZONOS_LOCAL_ENGINE_ID
                zonos_engine_instance = self.engine.tts_manager.get_engine(ZONOS_LOCAL_ENGINE_ID)

            # Pass the engine instance to the dashboard window if available
            self.zonos_dashboard_window = ZonosDashboardWindow(self, config=self.config, zonos_engine_override=zonos_engine_instance)
            self.zonos_dashboard_window.attributes("-topmost", True) # Keep on top
        else:
            self.zonos_dashboard_window.focus()

    def show_thinking_indicator(self):
        """Shows the thinking popup and disables input fields."""
        if hasattr(self, 'thinking_popup_instance') and self.thinking_popup_instance and self.thinking_popup_instance.winfo_exists():
            self.thinking_popup_instance.focus() # Bring to front if already exists
        else:
            self.thinking_popup_instance = self.display_thinking_popup()

        self.prompt_entry.configure(state="disabled")
        self.send_button.configure(state="disabled")
        self.mic_button.configure(state="disabled")
        self.status_label.configure(text=f"{self.config.get('wubu_name', 'WuBu')} is thinking...")


    def hide_thinking_indicator(self):
        """Hides the thinking popup and re-enables input fields."""
        if hasattr(self, 'thinking_popup_instance') and self.thinking_popup_instance:
            if self.thinking_popup_instance.winfo_exists():
                self.close_popup(self.thinking_popup_instance)
            self.thinking_popup_instance = None

        self.prompt_entry.configure(state="normal")
        self.send_button.configure(state="normal")
        self.mic_button.configure(state="normal")
        self.status_label.configure(text="Status: Ready")
        self.prompt_entry.focus()


    def on_send_prompt(self, event=None):
        prompt = self.prompt_entry.get()
        if prompt:
            self.add_message_to_chat("You", prompt) # Display user's message immediately
            # self.chat_history.append({"role": "user", "content": prompt}) # History managed by engine
            self.prompt_entry.delete(0, ctk.END)

            if self.engine:
                self.engine.process_user_prompt(prompt)
            else:
                self.add_message_to_chat("System", "Error: WuBu Engine not available.")
                self.hide_thinking_indicator() # Ensure UI is usable

    # _get_llm_response and _finalize_llm_interaction are removed as engine handles this.

    def on_mic_button_pressed(self):
        if self.engine:
            self.engine.toggle_asr_listening()
        else:
            self.display_message_popup("Error", "ASR System (Engine) not available.", "error")


    def set_prompt_input_text(self, text: str):
        """Allows the engine to set text in the prompt entry (e.g., from ASR)."""
        self.prompt_entry.delete(0, ctk.END)
        self.prompt_entry.insert(0, text)

    def add_message_to_chat(self, sender, message):
        # This method can now be called by the engine to display LLM responses
        self.chat_display.configure(state="normal")
        self.chat_display.insert(ctk.END, f"{sender}: {message}\n\n")
        self.chat_display.configure(state="disabled")
        self.chat_display.see(ctk.END)

    def display_thinking_popup(self):
        """Example of a 'thinking' popup."""
        popup = ctk.CTkToplevel(self)
        popup.geometry("300x100")
        popup.title("WuBu")
        # Make popup modal and on top
        popup.transient(self)
        popup.grab_set()
        popup.attributes("-topmost", True)

        label = ctk.CTkLabel(popup, text="WuBu is thinking...")
        label.pack(padx=20, pady=20, expand=True, fill=ctk.BOTH)

        # Center the popup
        self.update_idletasks() # Update main window geometry
        x = self.winfo_x() + (self.winfo_width() // 2) - (popup.winfo_width() // 2)
        y = self.winfo_y() + (self.winfo_height() // 2) - (popup.winfo_height() // 2)
        popup.geometry(f"+{x}+{y}")

        # This popup is now closed by _finalize_llm_interaction
        # self.after(2000, lambda: self.close_popup(popup)) # Old demo auto-close

        return popup # Return the instance so it can be managed

    def display_asr_popup(self, text):
        """Example of an ASR status popup."""
        popup = ctk.CTkToplevel(self)
        popup.geometry("300x100")
        popup.title("WuBu ASR")
        popup.transient(self)
        popup.grab_set()
        popup.attributes("-topmost", True)

        label = ctk.CTkLabel(popup, text=text)
        label.pack(padx=20, pady=20, expand=True, fill=ctk.BOTH)
        self.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() // 2) - (popup.winfo_width() // 2)
        y = self.winfo_y() + (self.winfo_height() // 2) - (popup.winfo_height() // 2)
        popup.geometry(f"+{x}+{y}")
        # Lifecycle of this popup will be managed by ASR interaction logic
        # self.after(2500, lambda: self.close_popup(popup)) # Old demo auto-close
        return popup # Return instance for external management

    def update_popup_text(self, popup_instance, new_text):
        """Updates the text of a label within a given popup window."""
        if popup_instance and popup_instance.winfo_exists():
            for widget in popup_instance.winfo_children():
                if isinstance(widget, ctk.CTkLabel):
                    widget.configure(text=new_text)
                    break # Assuming one main label per simple popup

    def close_popup(self, popup_instance):
        if popup_instance and popup_instance.winfo_exists():
            popup_instance.grab_release() # Use the passed argument name
            popup_instance.destroy()

    def display_confirmation_popup(self, title: str, message: str, yes_callback, no_callback=None):
        """Displays a modal confirmation popup with Yes/No buttons."""
        popup = ctk.CTkToplevel(self)
        popup.title(title)
        popup.transient(self)
        popup.grab_set()
        popup.attributes("-topmost", True)

        label = ctk.CTkLabel(popup, text=message, wraplength=350) # Wraplength for longer messages
        label.pack(padx=20, pady=(20,10), expand=True, fill=ctk.X)

        button_frame = ctk.CTkFrame(popup, fg_color="transparent")
        button_frame.pack(pady=(10,20))

        def _on_yes():
            self.close_popup(popup)
            if yes_callback:
                yes_callback()

        def _on_no():
            self.close_popup(popup)
            if no_callback:
                no_callback()

        yes_button = ctk.CTkButton(button_frame, text="Yes", command=_on_yes, width=80)
        yes_button.pack(side=ctk.LEFT, padx=(0,10))

        no_button = ctk.CTkButton(button_frame, text="No", command=_on_no, width=80, fg_color="gray") # Different color for No
        no_button.pack(side=ctk.LEFT)

        popup.update_idletasks()
        # Center popup
        x = self.winfo_x() + (self.winfo_width() // 2) - (popup.winfo_width() // 2)
        y = self.winfo_y() + (self.winfo_height() // 2) - (popup.winfo_height() // 2)
        popup.geometry(f"{popup.winfo_width()}x{popup.winfo_height()}+{x}+{y}")

        popup.protocol("WM_DELETE_WINDOW", _on_no) # Handle window close as "No"

        return popup

    def display_message_popup(self, title: str, message: str, message_type: str = 'info'):
        """Displays a simple modal message popup with an OK button."""
        popup = ctk.CTkToplevel(self)
        popup.title(title)
        popup.transient(self)
        popup.grab_set()
        popup.attributes("-topmost", True)

        # TODO: Could add an icon based on message_type if desired, e.g. using CTkImage
        # icon_label = ctk.CTkLabel(popup, text="[ICON]")
        # icon_label.pack(side=ctk.LEFT, padx=(10,0))

        label = ctk.CTkLabel(popup, text=message, wraplength=350) # Wraplength for longer messages
        label.pack(padx=20, pady=20, expand=True, fill=ctk.X)

        ok_button = ctk.CTkButton(popup, text="OK", command=lambda: self.close_popup(popup), width=80)
        ok_button.pack(pady=(0,20))

        popup.update_idletasks()
        # Center popup
        x = self.winfo_x() + (self.winfo_width() // 2) - (popup.winfo_width() // 2)
        y = self.winfo_y() + (self.winfo_height() // 2) - (popup.winfo_height() // 2)
        popup.geometry(f"{popup.winfo_width()}x{popup.winfo_height()}+{x}+{y}")

        popup.protocol("WM_DELETE_WINDOW", lambda: self.close_popup(popup)) # Handle window close

        return popup


def main():
    # Ensure an X server is running if on Linux and not using a virtual framebuffer
    # import os
    # if os.name == "posix" and not os.environ.get("DISPLAY"):
    #     print("Warning: DISPLAY environment variable not set. UI may not appear.")
    #     # For headless environments, you might need to use Xvfb or similar.
    #     # Example: os.environ['DISPLAY'] = ':0'

    config = load_config() # Load wubu_config.yaml from project root
    if not config:
        print("CRITICAL: WuBu UI could not load wubu_config.yaml. LLM and other features will be impaired.")
        # Create a minimal default config for UI to run with basic info
        config = {
            'wubu_name': "WuBu (Config Missing!)",
            'llm': {'provider': 'none'}, # Indicate no LLM provider
            'logging': {'level': "DEBUG"} # Default logging
        }


    ctk.set_appearance_mode("System")  # Modes: "System" (default), "Dark", "Light"
    ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "green", "dark-blue"

    app = WubuApp(config=config) # Pass config to the app
    app.mainloop()

if __name__ == "__main__":
    main()


class ZonosDashboardWindow(ctk.CTkToplevel):
    def __init__(self, master, config: dict, zonos_engine_override=None): # Added zonos_engine_override
        super().__init__(master)
        self.config = config
        self.master_app = master # Reference to the main WubuApp if needed

        self.zonos_engine = zonos_engine_override

        # Fallback logic to old Docker-based ZonosVoice has been removed.
        # The dashboard now relies solely on the zonos_engine_override (ZonosLocalVoice)
        # passed from the main application.
        if not self.zonos_engine:
            print("ZonosDashboard: ZonosLocalVoice engine instance was not provided or failed to initialize in the main app.")
            # The following existing 'if not self.zonos_engine:' block will handle UI indication.

        if not self.zonos_engine:
            self.title("Zonos TTS (Engine Error)")
            label = ctk.CTkLabel(self, text="Zonos TTS Engine not available or failed to load.\nPlease check main application logs and configuration.")
            label.pack(padx=20, pady=20)
            self.after(100, self.focus) # Ensure it gets focus to show message
            return # Don't build the rest of the UI

        self.title("Zonos TTS Dashboard")
        self.geometry("600x700")
        self.transient(master) # Keep on top of master
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.current_speaker_embedding = None
        self.reference_audio_path = ctk.StringVar()

        # --- UI Elements ---
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill=ctk.BOTH, expand=True, padx=10, pady=10)

        # 1. Reference Audio Selection
        ref_audio_frame = ctk.CTkFrame(main_frame)
        ref_audio_frame.pack(fill=ctk.X, pady=(0,10))
        ctk.CTkLabel(ref_audio_frame, text="Reference Audio (.wav, .mp3):").pack(side=ctk.LEFT, padx=(0,5))
        self.ref_audio_entry = ctk.CTkEntry(ref_audio_frame, textvariable=self.reference_audio_path, width=300)
        self.ref_audio_entry.pack(side=ctk.LEFT, expand=True, fill=ctk.X, padx=(0,5))
        self.browse_button = ctk.CTkButton(ref_audio_frame, text="Browse", command=self.browse_reference_audio, width=80)
        self.browse_button.pack(side=ctk.LEFT)

        # 2. Speaker Embedding
        embedding_frame = ctk.CTkFrame(main_frame)
        embedding_frame.pack(fill=ctk.X, pady=5)
        self.generate_embedding_button = ctk.CTkButton(embedding_frame, text="Generate Speaker Embedding", command=self.generate_embedding)
        self.generate_embedding_button.pack(side=ctk.LEFT, padx=(0,10))
        self.embedding_status_label = ctk.CTkLabel(embedding_frame, text="Status: No embedding generated.")
        self.embedding_status_label.pack(side=ctk.LEFT)

        # 3. Text for Synthesis
        ctk.CTkLabel(main_frame, text="Text to Synthesize:").pack(fill=ctk.X, pady=(10,0))
        self.text_input = ctk.CTkTextbox(main_frame, height=150, wrap=ctk.WORD)
        self.text_input.pack(fill=ctk.BOTH, expand=True, pady=5)
        self.text_input.insert("1.0", "Hello world, this is a test of the Zonos Text to Speech engine integrated into WuBu.")

        # 4. Playback Speed (Optional - Zonos has 'rate')
        speed_frame = ctk.CTkFrame(main_frame)
        speed_frame.pack(fill=ctk.X, pady=5)
        ctk.CTkLabel(speed_frame, text="Playback Speed:").pack(side=ctk.LEFT, padx=(0,10))
        self.speed_var = ctk.StringVar(value="NORMAL")
        speed_options = [s.name for s in TTSPlaybackSpeed]
        self.speed_menu = ctk.CTkOptionMenu(speed_frame, variable=self.speed_var, values=speed_options)
        self.speed_menu.pack(side=ctk.LEFT)

        # TODO: Add controls for Pitch, Energy if desired

        # 5. Synthesize and Play Button
        self.synthesize_button = ctk.CTkButton(main_frame, text="Synthesize and Play", command=self.synthesize_and_play)
        self.synthesize_button.pack(pady=10)

        self.synthesis_status_label = ctk.CTkLabel(main_frame, text="")
        self.synthesis_status_label.pack(fill=ctk.X)

        # --- Zonos Generation Parameters ---
        params_outer_frame = ctk.CTkFrame(main_frame)
        params_outer_frame.pack(fill=ctk.BOTH, expand=True, pady=10)

        params_label = ctk.CTkLabel(params_outer_frame, text="Zonos Generation Parameters", font=("Arial", 14, "bold"))
        params_label.pack(pady=(0,5))

        params_scroll_frame = ctk.CTkScrollableFrame(params_outer_frame, height=300) # Set a fixed height or make it expand
        params_scroll_frame.pack(fill=ctk.BOTH, expand=True)


        # Helper to create label and slider/entry
        # Moved definition inside __init__ or make it a staticmethod/global if preferred
        # For now, defining here for clarity of what's being added.
        # This helper needs to be defined before it's used.

        self._init_zonos_params_vars()
        self._create_zonos_params_ui(params_scroll_frame)


        self.after(100, self.focus) # Bring window to front

    def _init_zonos_params_vars(self):
        # Initialize CTk variables for Zonos parameters
        self.max_new_tokens_var = ctk.IntVar()
        self.cfg_scale_var = ctk.DoubleVar()
        self.temperature_var = ctk.DoubleVar()
        self.top_p_var = ctk.DoubleVar()
        self.top_k_var = ctk.IntVar()
        self.min_p_var = ctk.DoubleVar()
        self.linear_var = ctk.DoubleVar()
        self.confidence_var = ctk.DoubleVar()
        self.quadratic_var = ctk.DoubleVar()
        self.repetition_penalty_var = ctk.DoubleVar()
        self.repetition_penalty_window_var = ctk.IntVar()

    def _create_zonos_params_ui(self, parent_frame):
        # Helper to create label and slider/entry
        def add_param_control(parent, text, variable, from_, to, steps=None, default_value=0.0, is_int=False, is_float=False, precision=2):
            config_key = text.lower().replace(" ", "_").replace(":", "").replace("(", "").replace(")", "")

            current_value = default_value
            if self.config and 'zonos_settings' in self.config and config_key in self.config['zonos_settings']:
                current_value = self.config['zonos_settings'][config_key]

            if is_int:
                variable.set(int(current_value))
            elif is_float: # For DoubleVar, ensure it's float
                variable.set(float(current_value))
            else: # DoubleVar by default
                 variable.set(float(current_value))


            row_frame = ctk.CTkFrame(parent)
            row_frame.pack(fill=ctk.X, pady=2, padx=5)

            label = ctk.CTkLabel(row_frame, text=text, width=170, anchor="w")
            label.pack(side=ctk.LEFT, padx=(0,5))

            slider_steps = steps
            if steps is not None and isinstance(steps, float): # e.g. for float ranges like 0.01 steps
                # CTkSlider number_of_steps expects int. We calculate based on range and desired float step.
                # This might not be perfect for all float step scenarios with CTkSlider.
                # Consider direct entry for very fine float steps if slider is problematic.
                num_steps_calc = int((to - from_) / steps) if steps > 0 else 0
                slider_steps = num_steps_calc if num_steps_calc > 0 else None


            control_element = ctk.CTkSlider(row_frame, from_=from_, to=to, variable=variable, number_of_steps=slider_steps)
            control_element.pack(side=ctk.LEFT, expand=True, fill=ctk.X, padx=5)

            entry_width = 70
            value_entry = ctk.CTkEntry(row_frame, textvariable=variable, width=entry_width)
            value_entry.pack(side=ctk.LEFT, padx=5)

        # Max New Tokens
        add_param_control(parent_frame, "Max New Tokens:", self.max_new_tokens_var, 86, 86*30, steps=(86*30-86), default_value=86*30, is_int=True)
        # CFG Scale
        add_param_control(parent_frame, "CFG Scale:", self.cfg_scale_var, 1.01, 5.0, steps=0.01, default_value=2.0, is_float=True)
        # Temperature
        add_param_control(parent_frame, "Temperature:", self.temperature_var, 0.0, 2.0, steps=0.01, default_value=1.0, is_float=True)
        # Top P
        add_param_control(parent_frame, "Top P:", self.top_p_var, 0.0, 1.0, steps=0.01, default_value=0.0, is_float=True)
        # Top K
        add_param_control(parent_frame, "Top K:", self.top_k_var, 0, 1024, steps=1024, default_value=0, is_int=True)
        # Min P
        add_param_control(parent_frame, "Min P:", self.min_p_var, 0.0, 1.0, steps=0.01, default_value=0.1, is_float=True)

        ctk.CTkLabel(parent_frame, text="Unified Sampler:", font=("Arial", 12, "bold")).pack(pady=(10,0), anchor="w", padx=5)
        # Linear (Unified)
        add_param_control(parent_frame, "  Linear:", self.linear_var, -2.0, 2.0, steps=0.01, default_value=0.0, is_float=True)
        # Confidence (Unified)
        add_param_control(parent_frame, "  Confidence:", self.confidence_var, -2.0, 2.0, steps=0.01, default_value=0.0, is_float=True)
        # Quadratic (Unified)
        add_param_control(parent_frame, "  Quadratic:", self.quadratic_var, -2.0, 2.0, steps=0.01, default_value=0.0, is_float=True)

        ctk.CTkLabel(parent_frame, text="Repetition Penalty:", font=("Arial", 12, "bold")).pack(pady=(10,0), anchor="w", padx=5)
        # Repetition Penalty
        add_param_control(parent_frame, "  Penalty:", self.repetition_penalty_var, 1.0, 5.0, steps=0.01, default_value=1.2, is_float=True)
        # Repetition Penalty Window
        add_param_control(parent_frame, "  Window:", self.repetition_penalty_window_var, 0, 500, steps=500, default_value=50, is_int=True)

        # Button to save these settings
        self.save_zonos_settings_button = ctk.CTkButton(parent_frame, text="Save Zonos Settings to Config", command=self.save_zonos_settings)
        self.save_zonos_settings_button.pack(pady=15, padx=5)

    def save_zonos_settings(self):
        if not self.config:
            self.master_app.display_message_popup("Error", "Main config not loaded.", "error")
            return

        if 'zonos_settings' not in self.config:
            self.config['zonos_settings'] = {}

        self.config['zonos_settings']['max_new_tokens'] = self.max_new_tokens_var.get()
        self.config['zonos_settings']['cfg_scale'] = round(self.cfg_scale_var.get(), 2)
        self.config['zonos_settings']['temperature'] = round(self.temperature_var.get(), 2)
        self.config['zonos_settings']['top_p'] = round(self.top_p_var.get(), 2)
        self.config['zonos_settings']['top_k'] = self.top_k_var.get()
        self.config['zonos_settings']['min_p'] = round(self.min_p_var.get(), 2)
        self.config['zonos_settings']['linear'] = round(self.linear_var.get(), 2)
        self.config['zonos_settings']['confidence'] = round(self.confidence_var.get(), 2)
        self.config['zonos_settings']['quadratic'] = round(self.quadratic_var.get(), 2)
        self.config['zonos_settings']['repetition_penalty'] = round(self.repetition_penalty_var.get(), 2)
        self.config['zonos_settings']['repetition_penalty_window'] = self.repetition_penalty_window_var.get()

        try:
            # Attempt to save the main wubu_config.yaml
            # This assumes WuBuEngine or a config manager has a save method
            if hasattr(self.engine, 'save_main_config'): # Check if engine has a save method
                self.engine.save_main_config() # This method would save self.engine.config
                self.master_app.display_message_popup("Success", "Zonos settings saved to main config file.", "info")
            else: # Fallback: try to save directly if we know the path (less ideal)
                # This requires knowing the config path. For now, just update in-memory.
                # config_path = "wubu_config.yaml" # This is risky if path is not guaranteed
                # from ..utils.resource_loader import save_config
                # save_config(self.config, config_path)
                print("ZonosDashboard: Settings updated in memory. Main config save method not found on engine.")
                self.master_app.display_message_popup("Info", "Zonos settings updated in memory.\n(Full save to file requires engine support or restart)", "info")

        except Exception as e:
            print(f"Error saving Zonos settings to config: {e}")
            self.master_app.display_message_popup("Error", f"Failed to save Zonos settings to config file: {e}", "error")


    def browse_reference_audio(self):
        from tkinter import filedialog
        filepath = filedialog.askopenfilename(
            title="Select Reference Audio",
            filetypes=(("Audio Files", "*.wav *.mp3"), ("All files", "*.*"))
        )
        if filepath:
            self.reference_audio_path.set(filepath)
            self.current_speaker_embedding = None # Reset embedding if new file selected
            self.embedding_status_label.configure(text="Status: New reference audio selected. Generate embedding.")

    def generate_embedding(self):
        path = self.reference_audio_path.get()
        if not path:
            self.embedding_status_label.configure(text="Status: No reference audio selected.")
            self.master_app.display_message_popup("Error", "Please select a reference audio file first.", "error")
            return
        if not os.path.exists(path):
            self.embedding_status_label.configure(text=f"Status: File not found: {path}")
            self.master_app.display_message_popup("Error", f"Reference audio file not found:\n{path}", "error")
            return

        self.embedding_status_label.configure(text="Status: Generating embedding...")
        self.update_idletasks() # Ensure label updates

        try:
            # This call is synchronous, might freeze UI for a moment.
            # For long operations, consider threading. Zonos embedding is usually fast.
            self.current_speaker_embedding = self.zonos_engine._get_speaker_embedding(path) # Accessing protected member for now
            if self.current_speaker_embedding is not None:
                self.embedding_status_label.configure(text="Status: Speaker embedding ready.")
            else:
                self.embedding_status_label.configure(text="Status: Failed to generate embedding.")
                self.master_app.display_message_popup("Error", "Failed to generate speaker embedding. Check console for Zonos logs.", "error")

        except Exception as e:
            self.current_speaker_embedding = None
            self.embedding_status_label.configure(text=f"Status: Error generating embedding.")
            self.master_app.display_message_popup("Embedding Error", f"Error: {e}", "error")


    def synthesize_and_play(self):
        text_to_speak = self.text_input.get("1.0", ctk.END).strip()
        if not text_to_speak:
            self.synthesis_status_label.configure(text="Status: No text to synthesize.")
            self.master_app.display_message_popup("Input Error", "Please enter some text to synthesize.", "error")
            return

        if self.current_speaker_embedding is None:
            # Option: try to use default_voice from ZonosEngine if set, or prompt to generate.
            # For now, require explicit embedding generation in this UI.
            self.synthesis_status_label.configure(text="Status: No speaker embedding. Please generate one first.")
            self.master_app.display_message_popup("Input Error", "Please generate a speaker embedding first using a reference audio.", "error")

            return

        selected_speed_name = self.speed_var.get()
        playback_speed = TTSPlaybackSpeed[selected_speed_name]

        self.synthesis_status_label.configure(text="Status: Synthesizing audio...")
        self.synthesize_button.configure(state="disabled")
        self.update_idletasks()

        def _synthesis_task():
            try:
                # Use ZonosEngine's synthesize_to_bytes, then its play method
                # The voice_id for ZonosEngine is the reference audio path used for the current_speaker_embedding
                # However, ZonosVoice's synthesize_to_bytes itself takes the speaker_embedding directly if we modify it,
                # or we rely on it re-calculating from path if we pass self.reference_audio_path.get().
                # For efficiency, it's better if ZonosVoice can take a pre-made embedding.
                # Let's assume ZonosVoice.synthesize_to_bytes can take an optional 'speaker_embedding' kwarg.
                # If not, ZonosVoice needs a way to use a cached/passed embedding.
                # For now, we pass the path, and ZonosVoice's _get_speaker_embedding will use its cache.

                audio_bytes = self.zonos_engine.synthesize_to_bytes(
                    text_to_speak,
                    voice_id=self.reference_audio_path.get(), # Path used for current embedding
                    speed=playback_speed
                    # TODO: Add pitch, energy kwargs if UI controls are added
                )

                if audio_bytes:
                    self.synthesis_status_label.configure(text="Status: Playing audio...")
                    self.zonos_engine.play_synthesized_bytes(audio_bytes, speed=playback_speed) # Speed here is illustrative; Zonos applies during synthesis
                    self.synthesis_status_label.configure(text="Status: Synthesis and playback complete.")
                else:
                    self.synthesis_status_label.configure(text="Status: Synthesis failed. Check logs.")
                    self.master_app.display_message_popup("Synthesis Error", "Failed to synthesize audio. See console for details.", "error")

            except Exception as e:
                self.synthesis_status_label.configure(text=f"Status: Error during synthesis/playback.")
                self.master_app.display_message_popup("Error", f"An error occurred: {e}", "error")
            finally:
                self.synthesize_button.configure(state="normal")

        threading.Thread(target=_synthesis_task, daemon=True).start()


    def on_close(self):
        # Clean up resources if necessary, e.g., clear Zonos model from GPU if loaded by this window
        # For now, just destroy the window.
        # If ZonosVoice was instantiated by this window, it should be cleaned up.
        # If self.zonos_engine was created by this window:
        # if hasattr(self, 'zonos_engine_is_local') and self.zonos_engine_is_local and self.zonos_engine:
        #     del self.zonos_engine # or a proper cleanup method
        self.destroy()
        if self.master_app:
             self.master_app.zonos_dashboard_window = None # Unset reference in main app
# main() function and if __name__ == "__main__": block removed.
# This will be handled by a new main.py at the project root.
