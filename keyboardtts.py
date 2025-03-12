#This is optional
#allows typing if you need to type text in the app. The app is supposed to be fully voice interactive but this app allows you to input audio by generating speech from keyboard input by clicking enter to speak out the sentence. 

import keyboard
import pyttsx3
from queue import Queue
import threading
import tkinter as tk
from tkinter import ttk
import subprocess
import os
import sys
import tempfile

# Try to import optional packages
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

class KeyboardTTS:
    def __init__(self):
        # Queue for thread-safe communication
        self.input_queue = Queue()
        self.current_text = []

        # Global variables
        self.current_voice_id = None
        self.available_voices = []
        self.voice_names = []
        self.tts_engine_type = "pyttsx3"  # Default engine
        
        # Initialize UI
        self.root = tk.Tk()
        self.root.title("Advanced TTS App")
        self.root.geometry("450x450")
        
        self.setup_ui()
        
        # Initialize the app
        keyboard.hook(self.on_key_press)
        thread = threading.Thread(target=self.text_to_speech, daemon=True)
        thread.start()
        
        # Populate voice list
        self.refresh_voices()

    def on_key_press(self, event):
        # Only process key down events
        if event.event_type != 'down':
            return

        if event.name == 'enter':
            if self.current_text:
                full_text = ''.join(self.current_text)
                self.input_queue.put(full_text)
                self.current_text = []
                self.current_input_label.config(text=f"Current input: ")
        elif event.name == 'backspace':
            if self.current_text:
                self.current_text.pop()
                self.current_input_label.config(text=f"Current input: {''.join(self.current_text)}")
        elif event.name == 'space':
            self.current_text.append(' ')
            self.current_input_label.config(text=f"Current input: {''.join(self.current_text)}")
        elif len(event.name) == 1:
            self.current_text.append(event.name)
            self.current_input_label.config(text=f"Current input: {''.join(self.current_text)}")

    def get_available_voices(self):
        self.voice_names = []
        
        if self.tts_engine_type == "pyttsx3":
            engine = pyttsx3.init()
            self.available_voices = engine.getProperty('voices')
            
            for voice in self.available_voices:
                name = f"{voice.name} ({voice.languages[0] if voice.languages else 'Unknown'})"
                self.voice_names.append(name)
            
            engine.stop()
        elif self.tts_engine_type == "gtts":
            # Google TTS supported languages
            # This is a simplified list - gTTS supports many more
            self.voice_names = [
                "English (en)", "Finnish (fi)", "French (fr)", "German (de)", 
                "Italian (it)", "Japanese (ja)", "Korean (ko)", "Spanish (es)"
            ]
            self.available_voices = self.voice_names
        
        return self.voice_names

    def play_audio_file(self, file_path):
        """Play an audio file using the appropriate command based on the OS"""
        try:
            if sys.platform == 'win32':
                # Windows - use a more reliable method for MP3 files
                from playsound import playsound
                playsound(file_path)
            elif sys.platform == 'darwin':
                # macOS
                subprocess.call(['afplay', file_path])
            else:
                # Linux (requires mpg123)
                subprocess.call(['mpg123', file_path])
        except ImportError:
            # Fallback if playsound is not available
            if sys.platform == 'win32':
                # Use subprocess to open with default player
                subprocess.call(['start', file_path], shell=True)
            else:
                print(f"Error playing file: playsound module not found")
                self.status_label.config(text="Error: Install playsound package")

    def text_to_speech(self):
        while True:
            text = self.input_queue.get()
            if not text:
                continue
                
            try:
                if self.tts_engine_type == "pyttsx3":
                    # SAPI5 engine (Windows default)
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 150)
                    engine.setProperty('volume', 1.0)
                    
                    if self.current_voice_id:
                        engine.setProperty('voice', self.current_voice_id)
                    
                    # Get the current voice for display purposes
                    voices = engine.getProperty('voices')
                    current_voice = engine.getProperty('voice')
                    voice_name = "Default"
                    for voice in voices:
                        if voice.id == current_voice:
                            voice_name = voice.name
                            break
                    
                    self.status_label.config(text=f"Speaking with: {voice_name}")
                    print(f"Speaking with voice '{voice_name}': {text}")
                    
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()
                    
                elif self.tts_engine_type == "gtts" and GTTS_AVAILABLE:
                    # Google TTS
                    selected_idx = self.voice_combo.current()
                    if 0 <= selected_idx < len(self.available_voices):
                        selected_voice = self.available_voices[selected_idx]
                        # Extract language code from the display name
                        lang_code = selected_voice.split('(')[1].split(')')[0].strip()
                        
                        self.status_label.config(text=f"Using Google TTS ({lang_code})")
                        print(f"Speaking with Google TTS ({lang_code}): {text}")
                        
                        # Create a temporary file
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                        temp_file.close()
                        
                        # Generate speech
                        tts = gTTS(text=text, lang=lang_code, slow=False)
                        tts.save(temp_file.name)
                        
                        # Play the audio
                        self.play_audio_file(temp_file.name)
                        
                        # Clean up after a delay (allows audio to finish playing)
                        def cleanup_temp_file():
                            try:
                                os.unlink(temp_file.name)
                            except:
                                pass
                        
                        self.root.after(5000, cleanup_temp_file)
                else:
                    self.status_label.config(text="No TTS engine available")
                    
            except Exception as e:
                print(f"TTS Error: {e}")
                self.status_label.config(text=f"Error: {e}")

    def on_voice_select(self, event):
        selected_idx = self.voice_combo.current()
        
        if self.tts_engine_type == "pyttsx3":
            if 0 <= selected_idx < len(self.available_voices):
                self.current_voice_id = self.available_voices[selected_idx].id
                self.status_label.config(text=f"Selected: {self.available_voices[selected_idx].name}")
                print(f"Voice set to: {self.available_voices[selected_idx].name}")
        else:
            if 0 <= selected_idx < len(self.available_voices):
                self.status_label.config(text=f"Selected: {self.available_voices[selected_idx]}")
                print(f"Voice set to: {self.available_voices[selected_idx]}")

    def change_engine(self):
        self.tts_engine_type = self.engine_var.get()
        self.refresh_voices()
        print(f"Changed TTS engine to: {self.tts_engine_type}")

    def refresh_voices(self):
        voices = self.get_available_voices()
        self.voice_combo['values'] = voices
        if voices:
            self.voice_combo.current(0)
            self.on_voice_select(None)  # Trigger the selection event
        self.status_label.config(text=f"Found {len(voices)} voices with {self.tts_engine_type}")

    def install_packages(self):
        """Guide the user to install required packages"""
        instructions = (
            "To install the required packages, open a command prompt or terminal and run:\n\n"
            "pip install gtts playsound==1.2.2\n\n"
            "Note: We specifically use playsound 1.2.2 as newer versions may have issues.\n\n"
            "After installation, restart this application."
        )
        
        # Create instructions window
        install_win = tk.Toplevel(self.root)
        install_win.title("Installation Instructions")
        install_win.geometry("500x200")
        
        text_widget = tk.Text(install_win, wrap="word")
        text_widget.pack(fill="both", expand=True)
        text_widget.insert("1.0", instructions)

    def show_debug_info(self):
        if self.tts_engine_type == "pyttsx3":
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            current_voice = engine.getProperty('voice')
            
            debug_text = "Available SAPI5 Voices:\n"
            for i, voice in enumerate(voices):
                languages = voice.languages[0] if voice.languages else "Unknown"
                gender = voice.gender if hasattr(voice, 'gender') else "Unknown"
                age = voice.age if hasattr(voice, 'age') else "Unknown"
                debug_text += f"{i+1}. ID: {voice.id}\n   Name: {voice.name}\n   Lang: {languages}\n   Gender: {gender}\n   Age: {age}\n"
            
            debug_text += f"\nCurrent Voice ID: {current_voice}\n"
            engine.stop()
        else:
            debug_text = "Using Google TTS (gTTS):\n"
            debug_text += "This service uses Google's online TTS service and supports many languages.\n"
            debug_text += "Note: Google TTS requires an internet connection and the playsound package.\n"
            if not GTTS_AVAILABLE:
                debug_text += "\nNOTE: gTTS package is not installed. Click 'Install Packages' to install it.\n"
        
        # Add Windows voice info
        debug_text += "\n\nHow to add more Windows voices:\n"
        debug_text += "1. Go to Settings → Time & Language → Language & Region\n"
        debug_text += "2. Add the language you want (e.g., Finnish)\n"
        debug_text += "3. Click on the language, select Options, and download Text-to-Speech\n"
        debug_text += "4. After installation, restart this app and click 'Refresh Voice List'\n"
        
        # Create debug window
        debug_win = tk.Toplevel(self.root)
        debug_win.title("Voice Debug Information")
        debug_win.geometry("600x450")
        
        text_widget = tk.Text(debug_win, wrap="word")
        text_widget.pack(fill="both", expand=True)
        text_widget.insert("1.0", debug_text)

    def quit_app(self):
        print("Exiting application...")
        self.root.destroy()

    def on_escape(self, event):
        self.quit_app()

    def speak_current(self):
        """Speak the current text without waiting for Enter"""
        if self.current_text:
            full_text = ''.join(self.current_text)
            self.input_queue.put(full_text)

    def setup_ui(self):
        frame = ttk.Frame(self.root, padding="10")
        frame.pack(fill="both", expand=True)

        # Engine selection
        ttk.Label(frame, text="TTS Engine:").pack(pady=(10, 5), anchor='w')
        self.engine_var = tk.StringVar(value="pyttsx3")
        engine_frame = ttk.Frame(frame)
        engine_frame.pack(fill='x', pady=(0, 10))

        ttk.Radiobutton(engine_frame, text="Windows SAPI5", variable=self.engine_var, 
                        value="pyttsx3", command=self.change_engine).pack(side='left', padx=(0, 10))
        ttk.Radiobutton(engine_frame, text="Google TTS", variable=self.engine_var, 
                        value="gtts", command=self.change_engine).pack(side='left')

        if not GTTS_AVAILABLE:
            ttk.Button(frame, text="Install Packages", command=self.install_packages).pack(pady=(0, 10))

        # Voice selection
        ttk.Label(frame, text="Select Voice:").pack(pady=(5, 5), anchor='w')
        self.voice_combo = ttk.Combobox(frame, state="readonly", width=40)
        self.voice_combo.pack(pady=(0, 10), fill='x')
        self.voice_combo.bind("<<ComboboxSelected>>", self.on_voice_select)

        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill='x', pady=(0, 10))

        refresh_btn = ttk.Button(button_frame, text="Refresh Voice List", command=self.refresh_voices)
        refresh_btn.pack(side='left', padx=(0, 5))

        debug_btn = ttk.Button(button_frame, text="Voice Info", command=self.show_debug_info)
        debug_btn.pack(side='left', padx=5)

        speak_btn = ttk.Button(button_frame, text="Speak", command=self.speak_current)
        speak_btn.pack(side='left', padx=5)

        # Status and input display
        self.status_label = ttk.Label(frame, text="Select a voice to begin")
        self.status_label.pack(pady=5)

        self.current_input_label = ttk.Label(frame, text="Current input: ")
        self.current_input_label.pack(pady=5)

        ttk.Separator(frame).pack(fill='x', pady=10)

        # Instructions
        instr_text = "Recording keyboard input...\n" + \
                    "Type and press Enter to speak\n" + \
                    "Press 'Esc' to exit"
        instr_label = ttk.Label(frame, text=instr_text)
        instr_label.pack(pady=10)

        quit_button = ttk.Button(frame, text="Quit", command=self.quit_app)
        quit_button.pack(pady=10)

        self.root.bind('<Escape>', self.on_escape)

    def run(self):
        self.root.mainloop()


# Create and run the application
if __name__ == "__main__":
    app = KeyboardTTS()
    app.run()
