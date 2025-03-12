# Copyright (C) <2025>  <Roni Sam Daniel Tervo>
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from num2words import num2words
import re
import queue
import io
import wave
import numpy as np
import threading
from datetime import timedelta
import dateutil.parser
import subprocess
import ollama
from pydub import AudioSegment
from io import BytesIO
import wavio
from openai import OpenAI  # Ensure this import matches your OpenAI client library
from pathlib import Path
import requests
import json
import base64
from PIL import Image, ImageTk, ImageDraw  # <-- Note we import ImageDraw for PIL drawing
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk  # For Combobox and Notebook
import threading
import queue
import numpy as np
import os
import time
import sys
from datetime import datetime
import platform
import logging
import re  # Added for regex in transcription filtering
import sounddevice as sd
import chess  # pip install python-chess
import chess.pgn
from scipy.signal import resample
import noisereduce as nr  # pip install noisereduce

# Sorry for the mix of English and Finnish in commenting and code. 
#I got help from LLM in the coding process, so the coding and commenting style changes sometimes, and the code is sometimes unorganized.
# This is a work in progress, and I am fixing bugs and adding new features as often as possible. 

# --------------------------- Configuration Parameters ---------------------------
DEFAULT_VOL_THRESHOLD = 10  # Volume threshold for relevance
DEFAULT_MIN_RECORD_DURATION = 2  # Minimum duration (in seconds) to trigger transcription
DEFAULT_MAX_SILENCE_DURATION = 2  # Maximum allowed silence between segments (in seconds)

CONFIG_FILE = "config.json"
TASKS_FILE = "tasks.json"

# Load existing config
config = {}
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)

DEFAULT_IMAGE_DIR = Path(config.get('DEFAULT_IMAGE_DIR', r"C:\Users\ronit\Pictures\Nayttokuvat"))

RECORDED_AUDIO_PATH = "recorded_audio.wav"
OUTPUT_SPEECH_PATH = "output.wav"

# Add this near the top of the file with other constants
CHESS_SAVE_FILE = "chess_game_state.json"

BREAK_WORK_PROMPTS = {
    "work_start": "Time to get back to work.",
    "work_end": "Time to take short break.",
    "break_end": "End of break."
}

# Initialize Conversation History
messages = []

# Initialize Queues for Audio Data
audio_queue1 = queue.Queue()  # Queue for Audio Source 1
audio_queue2 = queue.Queue()  # Queue for Audio Source 2



LANGUAGE_PROMPTS_FEN = {

            "Finnish": "Ole hyvä ja analysoi seuraava shakin asema (FEN: ",

            "English": "Please analyze the following chess position (FEN: ",

            "Swedish": "Vänligen analysera följande schackposition (FEN: ",

            "Spanish": "Por favor analice la siguiente posición de ajedrez (FEN:"
        }
LANGUAGE_PROMPTS_AFTER_FEN_A = {

            "Finnish": "\n\nSuunnittele shakkiliikkeet sisäisesti ehdottaaksesi parasta shakkiliikettä tässä tilanteessa. Shakkipelin edistyminen riippuu ehdottamasi liikkeen tasosta, joten harkitse siirtoasi huolellisesti.",

            "English": "\n\nPlan chess moves internally to suggest the best chess move in this situation. The progress of the chess game depends on the level of the move you propose, so consider your move carefully.",

            "Swedish": "\n\nPlanera schackdrag internt för att föreslå det bästa schackdraget i denna situation. Framstegen i schackspelet beror på nivån på det drag du föreslår, så överväg ditt drag noga.",

            "Spanish": "\n\nPlanifique movimientos de ajedrez internamente para sugerir el mejor movimiento de ajedrez en esta situación. El progreso de la partida de ajedrez depende del nivel del movimiento que propongas, así que considera tu movimiento con atención."
        }
LANGUAGE_PROMPTS_AFTER_FEN_B = {

            "Finnish": "Varmista että valitsemasi shakkisiirto sisältyy alla olevaan sallittujen shakkisiirtojen luetteloon. Jos se ei ole, toista sisäinen analyysisi, kunnes se on. ",

            "English": "Verify that your chosen move is included in the list of allowed moves below. If it isn’t, repeat your internal analysis until it is. ",

            "Swedish": "Verifiera sedan att ditt valda schackdrag finns med i listan över tillåtna schackdrag nedan. Om det inte är det, upprepa din interna analys tills den är det. ",

            "Spanish": "Verifica que el movimiento de ajedrez elegido esté incluido en la lista de movimientos de ajedrez permitidos a continuación. Si no es así, repita su análisis interno hasta que lo sea. "
        }
LANGUAGE_PROMPTS_NO_EXPLANATION = {
            #"Finnish": "siirron jälkeen selitä se lyhyesti suomeksi.",
            "Finnish": "\n\nVastaa vain kyseisellä shakkisiirrolla UCI-muodossa – älä liitä selityksiä tai lisätekstiä.",
            #"English": "After the move, provide a brief explanation in English.",
            "English": "\n\nRespond only with that chess move in UCI format—do not include any explanation or additional text.",
            #"Swedish": "Efter draget, ge en kort förklaring på svenska.",
            "Swedish": "\n\nSvara endast med det schackdraget i UCI-format – inkludera ingen förklaring eller ytterligare text.",
            #"Spanish": "Después del movimiento, proporciona una breve explicación en español."
            "Spanish": "\n\nResponda solo con esa jugada de ajedrez en formato UCI; no incluya ninguna explicación ni texto adicional."
        }
LANGUAGE_PROMPTS_EXPLAIN_MOVE = {

            "Finnish": "Shakkisiirron jälkeen selitä siirtosi lyhyesti Suomeksi.",

            "English": "After the move, provide a brief explanation in English.",

            "Swedish": "Efter schackdraget, ge en kort förklaring på svenska.",

            "Spanish": "Después de la jugada de ajedrez, brinde una breve explicación en español."
        }

# --------------------------- Logging Configuration ---------------------------
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.INFO  # Changed to INFO to log filtered transcriptions
)

# --------------------------- Helper Functions ---------------------------
def encode_image(image_path):
    """Encode image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"Error encoding image: {e}")
        return None

def select_latest_image():
    """Select the latest image from the default directory."""
    if not DEFAULT_IMAGE_DIR.exists() or not DEFAULT_IMAGE_DIR.is_dir():
        logging.error(f"The directory {DEFAULT_IMAGE_DIR} does not exist.")
        return None

    latest_file = max(DEFAULT_IMAGE_DIR.glob('*'), key=os.path.getctime, default=None)

    if latest_file is None:
        logging.error(f"There are no files in the directory {DEFAULT_IMAGE_DIR}.")
        return None

    if not latest_file.is_file():
        logging.error(f"The latest file {latest_file} is not a valid file.")
        return None

    return str(latest_file)


def load_config():
    config = {}  # Initialize config as an empty dictionary
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            try:
                config = json.load(f)
            except json.JSONDecodeError:
                # Handle the case where the file is empty or corrupt
                messagebox.showerror("Error", "config.json is corrupted or empty.  Using default settings.")
                config = {}  # Reset to default settings

    # Jos avain "LLM_PERSONALITIES" ei ole olemassa, luodaan se tyhjänä
    if 'LLM_PERSONALITIES' not in config:
        config['LLM_PERSONALITIES'] = {}
        save_config(config)  # Tallenna muutos

    # Muut konfiguraatioon liittyvät asiat…
    if 'MODELS_LIST' not in config:
        config['MODELS_LIST'] = [
            "openai/gpt-4o-2024-11-20",
            "deepseek / deepseek - r1:free"
            "openai/o1-mini",
            "openai/o1-preview",
            "x-ai/grok-vision-beta",
            "meta-llama/llama-3.2-90b-vision-instruct",
            "openai/gpt-4o-mini",
            "anthropic/claude-3.5-sonnet-20240620",
            "ollama/qwq:32b",  # Add a default Ollama model.  IMPORTANT: User needs to pull this.
        ]
        save_config(config)


    # Initialize Mureka settings if they don't exist
    if 'MUREKA_API_URL' not in config:
        config['MUREKA_API_URL'] = ''
    if 'MUREKA_ACCOUNT' not in config:
        config['MUREKA_ACCOUNT'] = ''
    if 'MUREKA_API_TOKEN' not in config:
        config['MUREKA_API_TOKEN'] = ''

    return config


def save_config(config):
    """Save API keys, default image directory, blacklist, and models list to config file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

def is_valid_transcription(text, min_length=3, blacklist=None):
    """Determine whether the transcription text is valid for sending to the LLM."""
    text = text.strip()

    # Check minimum length
    if len(text.split()) < min_length:
        return False

    # Blacklist phrases
    if blacklist:
        normalized_text = text.lower()
        if normalized_text in [phrase.lower() for phrase in blacklist]:
            return False

    # MUUTOS (LISÄYS), Sallii ainoastaan A-Ö ja normaalit kirjoitusmerkit. Ei kiinan, japanin, arabian yms kirjoitusmerkkejä.
    allowed_chars_pattern = re.compile(
        r'^[A-Za-zÀ-ÖØ-öø-ÿ0-9\s\.,!?;:\'\"()\-\[\]]+$'
    )
    if not allowed_chars_pattern.match(text):
        return False

    # Patterns to identify non-speech inputs
    patterns = [
        r'^(\w{2,}\s?){2,}$',  # Repeated words
        r'^\.{2,}$',          # Series of dots
        r'^[\.\s]+$',         # Dots and spaces only
        r'^[^\w\s]+$',        # Non-word characters
    ]
    for pattern in patterns:
        if re.match(pattern, text, re.UNICODE):
            return False

    return True

def make_audio_callback(q):
    """Factory to create an audio callback function that puts data into the given queue."""
    def callback(indata, frames, time_info, status):
        if status:
            #print(f"Stream status: {status}", file=sys.stderr)
            pass
        audio_data = indata.copy()
        q.put(audio_data)
    return callback


def process_audio_stream(stream_stop_event, app, selected_language, openai_api_key, openrouter_api_key, selected_model):
    """Process audio data from the active queue and handle relevant inputs."""
    global messages
    recorded_frames = []
    recording = False
    record_start_time = None
    last_sound_time = None
    
    # Initialize OpenAI client inside the thread
    client = OpenAI(api_key=openai_api_key)
    
    while not stream_stop_event.is_set():
        try:
            # Get current toggle state
            active_source = app.audio_source_toggle_var.get()
            
            if active_source == "as1":
                audio_queue = app.audio_queue1
            elif active_source == "as2":
                audio_queue = app.audio_queue2
            elif active_source == "mute":
                # Skip processing
                time.sleep(0.1)
                continue
            else:
                # Unknown option, skip
                time.sleep(0.1)
                continue
            
            # Attempt to get audio data from the active queue
            audio_data = audio_queue.get(timeout=0.5)
            volume = np.linalg.norm(audio_data) / len(audio_data)
            
            current_time = time.time()
            
            vol_threshold = app.get_vol_threshold()
            min_record_duration = app.get_min_record_duration()
            max_silence_duration = app.get_max_silence_duration()
            
            if recording:
                # Always append frames while recording is active
                recorded_frames.append(audio_data)
            
            if volume > vol_threshold:
                if not recording:
                    # Start recording
                    recording = True
                    record_start_time = current_time
                    last_sound_time = current_time
                    recorded_frames = []
                    print("Start of relevant audio detected.")
                    app.update_status("Recording...")
                else:
                    # Update the last sound time
                    last_sound_time = current_time
            else:
                if recording:
                    silence_duration = current_time - last_sound_time
                    
                    # Only end recording if silence has lasted longer than max_silence_duration
                    if silence_duration > max_silence_duration:
                        total_duration = current_time - record_start_time
                        if total_duration >= min_record_duration:
                            print(f"End of relevant audio detected. Duration: {total_duration:.2f} seconds.")
                            app.update_status("Processing...")
                            app.save_and_process_audio(
                                recorded_frames,
                                selected_language,
                                client,
                                openrouter_api_key,
                                selected_model
                            )
                        else:
                            print(f"Recorded audio discarded. Duration: {total_duration:.2f} seconds.")
                            app.append_message(
                                f"Recorded audio discarded. Duration: {total_duration:.2f} seconds.",
                                sender="system"
                            )
                        recording = False
                        recorded_frames = []
                        app.update_status("Listening...")
        except queue.Empty:
            continue
        except KeyboardInterrupt:
            print("Audio processing stopped by user.")
            break
        except Exception as e:
            logging.error(f"Unexpected error in audio processing thread: {e}")
            app.handle_error(f"Audio processing error: {e}")
            break

def split_text_into_chunks(text, max_length=3000):
    """
    Pilkkoo annetun tekstin osiin niin, ettei yksittäinen osa
    ylitä max_length-merkkimäärää. Palauttaa listan tekstipaloista.
    """

    # Poistetaan mahdolliset ylimääräiset välilyönnit alusta ja lopusta
    text = text.strip()
    # Jos teksti jo valmiiksi lyhyempi kuin raja, palautetaan suoraan
    if len(text) <= max_length:
        return [text]

    chunks = []
    current_chunk = []

    # Voidaan jakaa ensin pisteen (.) perusteella, tai halutessasi pelkkien sanojenkin
    sentences = text.split('.')

    for sentence in sentences:
        # Lisätään piste takaisin lauseeseen
        sentence = sentence.strip()
        if sentence:
            sentence_with_dot = sentence + "."
        else:
            sentence_with_dot = sentence  # Jos on tyhjä pätkä

        # Jos lause mahtuu nykyiseen pätkään, lisätään se
        if sum(len(s) for s in current_chunk) + len(sentence_with_dot) <= max_length:
            current_chunk.append(sentence_with_dot)
        else:
            # Jos ei mahdu, suljetaan nykyinen chunk ja aloitetaan uusi
            chunks.append(" ".join(current_chunk).strip())
            current_chunk = [sentence_with_dot]

    # Lisää viimeinen chunk, jos sitä on
    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    # Varmistetaan, ettei yhdenkään chunkin pituus ole 0
    chunks = [c for c in chunks if c]

    return chunks

class TaskManager:
    def __init__(self):
        self.tasks = []
        self.load_tasks()
    
    def load_tasks(self):
        if os.path.exists(TASKS_FILE):
            with open(TASKS_FILE, 'r') as f:
                self.tasks = json.load(f)
    
    def save_tasks(self):
        with open(TASKS_FILE, 'w') as f:
            json.dump(self.tasks, f, indent=4, default=str)
    
    def add_task(self, description, due_date, priority="Medium"):
        self.tasks.append({
            "description": description,
            "due_date": due_date.isoformat(),
            "added": datetime.now().isoformat(),
            "priority": priority,
            "completed": False
        })
        self.save_tasks()
    
    def get_due_tasks(self):
        due = []
        for task in self.tasks:
            if not task['completed'] and datetime.now() > dateutil.parser.parse(task['due_date']):
                due.append(task)
        return due
        
    # New methods to handle completed tasks and deletion
    def get_all_tasks(self):
        """Return all tasks (both completed and incomplete)"""
        return self.tasks
        
    def toggle_task_completion(self, index):
        """Toggle the completion status of the task at the given index"""
        if 0 <= index < len(self.tasks):
            self.tasks[index]['completed'] = not self.tasks[index]['completed']
            self.save_tasks()
            return True
        return False
        
    def delete_task(self, index):
        """Delete the task at the given index"""
        if 0 <= index < len(self.tasks):
            del self.tasks[index]
            self.save_tasks()
            return True
        return False


# --------------------------- Tkinter User Interface ---------------------------
class ChatAudioApp5(tk.Tk):
    def __init__(self):
        super().__init__()
        self.active_threads = []
        self.thread_lock = threading.RLock()  # For thread-safe operations
        self.save_image_path_button = None
        self.title("Chat Audio Generator")
        # Aseta ikkuna kokoruutuun
        self.set_fullscreen()
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # We'll store available devices and sample rate after user selection:
        self.available_input_devices1 = []
        self.available_input_devices2 = []
        self.available_output_devices = []
        self.audio_source1_var = None
        self.audio_source2_var = None
        self.audio_output_var = None

        # --------------------- Early Initialization ---------------------
        self.config_data = load_config()
        self.excluded_phrases = self.config_data.get('BLACKLIST', [
            "Kiitos kun katsoit",
            "Kiitos kun katsoit.",
            "Kiitos kun katsoit!",
            "Kiitos, että katsoitte.",
            "Kiitos, että katsoitte",
            "Kiitos, että katsoitte!",
        ])

        # Add trace to update last_audio_source when radiobuttons change it
        self.audio_source_toggle_var = tk.StringVar(value="as1")
        self.audio_source_toggle_var.trace("w", self.on_audio_source_change)

        # Initialize last audio source
        self.last_audio_source = "as1"  # Default to microphone

        # Initialize Conversation History
        global messages
        messages = []
        self.chat_history = []
        # First, create the Notebook and tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.main_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.main_tab, text="Main")
        self.notebook.add(self.settings_tab, text="Settings")



        

        # Status label at the bottom
        self.status_label = tk.Label(self, text="Idle", fg="blue", anchor="w", font=("Arial", 12))
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        # Lisää progress bar status-labelin alapuolelle
        self.progress_bar = ttk.Progressbar(self, orient='horizontal', length=400, mode='determinate')
        self.progress_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 5))

        # Audio streams and thread
        self.stream1 = None
        self.stream2 = None
        self.processing_thread = None
        self.stream_stop_event = threading.Event()

        # API keys
        self.openai_key = self.config_data.get('OPENAI_API_KEY', '')
        self.openrouter_key = self.config_data.get('OPENROUTER_API_KEY', '')

        # --- CHESS INTEGRATION ---
        self.chess_board = chess.Board()
        self.chess_canvas = None
        self.selected_square = None
        self.square_size = 60
        self.chess_images = {}
        self.raw_chess_images = {}



        self.load_chess_images()
        self.ai_tried_chess_move = False
        self.human_tried_chess_move = False
        self.chess_game_ended = False
        self.show_ai_only_valid_moves_var = tk.BooleanVar(value=False)
        self.show_ai_only_valid_moves = self.show_ai_only_valid_moves_var.get()
        self.chess_frame = None

        # Create the settings and main tabs
        self.create_settings_tab()
        self.create_main_tab()


        self.active_canvas = self.chat_canvas

        self.draw_chessboard()
        self.hide_chess_ui()  # Hide by default
        self.load_chess_game_state() # Try to load saved chess game




        # Initialize Task Manager
        self.task_manager = TaskManager()


        
        # Add tasks tab
        self.tasks_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.tasks_tab, text="Tasks")
        self.create_tasks_tab()

        # Work/Break Schedule Variables
        self.schedule_running = False
        self.current_state = None
        self.work_duration = 0  # in seconds
        self.break_duration = 0  # in seconds
        self.task_manager = TaskManager()
        self.task_check_interval = 60  # seconds
        self.start_task_checker()

        # Initialize Queues for Audio Data
        self.audio_queue1 = queue.Queue()  # Queue for Audio Source 1
        self.audio_queue2 = queue.Queue()  # Queue for Audio Source 2

        

        self.thinking_tts_queue = queue.Queue()
        self.thinking_tts_active = False
        self.thinking_tts_worker = None

        #Tämä (Kun True) merkitsee että TTS-queue on tyhjä ja streamaus valmis, mutta vielä voi olla äänentoistoa.
        self.tts_queue_completed = False
        #Tämä (kun true) merkitsee että TTS-nykyinen äänentuotto on valmis
        self.current_tts_stream_completed = False
        

        # Check if API keys are available
        self.api_keys_provided = False
        self.check_api_keys()



        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)  # Lisätty!
        self.bind_mousewheel()  # Tämän voi kutsua initin lopussa

        self.after(1000, self.start_listening)  # 1-second delay to ensure UI is ready






    def start_thread(self, target, args=(), kwargs={}, thread_name=None, daemon=True):
        """Create, register and start a new thread with proper tracking."""
        thread = threading.Thread(target=target, args=args, kwargs=kwargs)
        
        if thread_name:
            thread.name = thread_name
        thread.daemon = daemon
        
        with self.thread_lock:
            self.active_threads.append({
                'thread': thread,
                'name': thread.name,
                'start_time': datetime.now(),
                'status': 'starting'
            })
        
        thread.start()
        return thread

    def cleanup_threads(self, timeout=1.0):
        """Attempt to clean up all active threads."""
        with self.thread_lock:
            active_threads = self.active_threads.copy()
        
        for thread_info in active_threads:
            thread = thread_info['thread']
            if thread.is_alive():
                logging.info(f"Waiting for thread {thread.name} to finish...")
                thread.join(timeout=timeout)
        
        # Update active threads list
        with self.thread_lock:
            self.active_threads = [t for t in self.active_threads 
                                if t['thread'].is_alive()]
            
    def show_active_threads(self):
        """Display all active threads in the application."""
        with self.thread_lock:
            if not self.active_threads:
                self.append_message("No active threads running.", sender="system")
                return
                
            thread_info = []
            for t in self.active_threads:
                runtime = datetime.now() - t['start_time']
                thread_info.append(f"{t['name']}: running for {runtime.seconds}s")
            
            self.append_message(
                f"Active threads ({len(self.active_threads)}):\n" + 
                "\n".join(thread_info), 
                sender="system"
            )

    def set_fullscreen(self):
        system = platform.system()
        if system == 'Windows':
            self.state('zoomed')  # Asettaa ikkunan suurennettuun tilaan
        elif system == 'Darwin':
            self.attributes('-zoomed', True)  # MacOS
        else:
            # Linux ja muut käyttöjärjestelmät
            self.attributes('-zoomed', True)
            # Jos '-zoomed' ei toimi, voit käyttää screenin kokoa
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()
            self.geometry(f"{screen_width}x{screen_height}+0+0")

    def on_frame_configure(self, event):
        """
        Aseta vieritysalueen koko Canvasille.
        """
        self.settings_canvas.configure(scrollregion=self.settings_canvas.bbox("all"))

    # --------------------------- Create Main Tab ---------------------------
    def create_main_tab(self):
        # Create a horizontal PanedWindow for resizable panes
        self.paned_window = ttk.PanedWindow(self.main_tab, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # Left pane: Chat frame
        chat_frame = tk.Frame(self.paned_window)
        self.paned_window.add(chat_frame, weight=1)  # Weight allows resizing

        self.chat_canvas = tk.Canvas(chat_frame, bg="#ECE5DD")
        self.chat_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = tk.Scrollbar(chat_frame, orient="vertical", command=self.chat_canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.chat_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.chat_inner_frame = tk.Frame(self.chat_canvas, bg="#ECE5DD")
        self.chat_canvas.create_window((0, 0), window=self.chat_inner_frame, anchor="nw")

        self.chat_inner_frame.bind(
            "<Configure>",
            lambda e: self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all"))
        )
        self.bind_mousewheel()  # Ensure mousewheel binding is still called

        # Right pane: Container for controls, game selection, buttons, and chess
        self.right_frame = tk.Frame(self.paned_window)
        self.paned_window.add(self.right_frame, weight=1)  # Weight allows resizing

        # Controls frame (model selection + buttons)
        controls_frame = tk.Frame(self.right_frame)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)

        self.thread_monitor_button = tk.Button(
            controls_frame,
            text="Show Active Threads",
            command=self.show_active_threads,
            width=15,
            bg="#007BFF",
            fg="white"
        )
        self.thread_monitor_button.pack(side=tk.LEFT, padx=(10, 0))

        self.app2_button = tk.Button(
            controls_frame,
            text="Open PDF Tool",
            command=self.launch_app2,
            bg="#4CAF50",
            fg="white"
        )
        self.app2_button.pack(side=tk.RIGHT, padx=5)

        self.app3_button = tk.Button(
            controls_frame,
            text="Open keyboard tts",
            command=self.launch_app3,
            bg="#4CAF50",
            fg="white"
        )
        self.app3_button.pack(side=tk.LEFT, padx=5)

        model_label = tk.Label(controls_frame, text="Select LLM Model:", anchor="w")
        model_label.pack(side=tk.LEFT, padx=(0, 10))

        self.model_var = tk.StringVar()
        self.model_combobox = ttk.Combobox(controls_frame, textvariable=self.model_var, state="readonly", width=30)
        self.model_combobox['values'] = self.models_list
        if self.models_list:
            self.model_combobox.current(0)
        self.model_combobox.pack(side=tk.LEFT, padx=(0, 10))
        self.model_combobox.bind("<<ComboboxSelected>>", self.on_model_changed)

        self.send_image_var = tk.BooleanVar(value=True)
        self.send_image_check = tk.Checkbutton(
            controls_frame,
            text="Send image with message",
            justify=tk.LEFT,
            variable=self.send_image_var
        )
        self.send_image_check.pack(side=tk.LEFT, padx=10)

        self.skip_tts_var = tk.BooleanVar(value=False)
        self.skip_tts_check = tk.Checkbutton(
            controls_frame,
            text="Skip TTS (Text response only)",
            justify=tk.LEFT,
            variable=self.skip_tts_var
        )
        self.skip_tts_check.pack(side=tk.LEFT, padx=10)

        self.add_model_button = tk.Button(controls_frame, text="Add Model", command=self.add_model, width=10)
        self.add_model_button.pack(side=tk.LEFT, padx=(5, 5))
        self.remove_model_button = tk.Button(controls_frame, text="Remove Model", command=self.remove_model, width=12)
        self.remove_model_button.pack(side=tk.LEFT, padx=(5, 0))

        self.reset_conversation_button = tk.Button(
            controls_frame,
            text="Reset Memory",
            command=self.reset_conversation,
            width=15,
            bg="#FF5733",
            fg="white"
        )
        self.reset_conversation_button.pack(side=tk.LEFT, padx=(10, 0))

        self.clear_chat_button = tk.Button(
            controls_frame,
            text="Clear chat",
            command=self.clear_chat,
            width=15,
            bg="#3498DB",
            fg="white"
        )
        self.clear_chat_button.pack(side=tk.LEFT, padx=(10, 0))

        # Game selection frame
        game_frame = tk.Frame(self.right_frame)
        game_frame.pack(fill=tk.X, padx=10, pady=5)

        game_label = tk.Label(game_frame, text="Select Game:", anchor="w")
        game_label.pack(side=tk.LEFT, padx=(0, 10))

        self.game_var = tk.StringVar(value="None")
        self.game_combobox = ttk.Combobox(
            game_frame,
            textvariable=self.game_var,
            state="readonly",
            values=["None", "Chess"],
            width=20
        )
        self.game_combobox.pack(side=tk.LEFT, padx=(0, 10))
        self.game_combobox.bind("<<ComboboxSelected>>", self.on_game_selected)

        # Button frame (listening status and toggle)
        button_frame = tk.Frame(self.right_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        self.listening_status_label = tk.Label(
            button_frame,
            text="Listening Status: Active",
            fg="green",
            font=("Arial", 11)
        )
        self.listening_status_label.pack(side=tk.LEFT, padx=5)

        self.toggle_listening_button = tk.Button(
            button_frame,
            text="Pause/Resume Listening",
            command=self.toggle_listening_state,
            width=20
        )
        self.toggle_listening_button.pack(side=tk.LEFT, padx=5)

        # Toggle frame for audio source
        toggle_frame = tk.Frame(self.right_frame)
        toggle_frame.pack(fill=tk.X, padx=10, pady=5)

        toggle_label = tk.Label(toggle_frame, text="Audio Source:", anchor="w")
        toggle_label.pack(side=tk.LEFT, padx=(0, 5))

        rb_as1 = tk.Radiobutton(toggle_frame, text="Microphone", variable=self.audio_source_toggle_var, value="as1")
        rb_mute = tk.Radiobutton(toggle_frame, text="Mute", variable=self.audio_source_toggle_var, value="mute")
        rb_as2 = tk.Radiobutton(toggle_frame, text="System Audio", variable=self.audio_source_toggle_var, value="as2")
        rb_as1.pack(side=tk.LEFT)
        rb_mute.pack(side=tk.LEFT)
        rb_as2.pack(side=tk.LEFT)

        # Create chess frame as a child of right_frame (initially hidden)
        self.create_chess_frame(self.right_frame)
        self.hide_chess_ui()  # Ensure it's hidden by default

    # --------------------------- Create Settings Tab ---------------------------
    def create_settings_tab(self):

        self.settings_canvas = tk.Canvas(self.settings_tab, borderwidth=0, background="#f0f0f0")
        self.settings_scrollbar = tk.Scrollbar(self.settings_tab, orient="vertical", command=self.settings_canvas.yview)
        self.settings_canvas.configure(yscrollcommand=self.settings_scrollbar.set)

        self.settings_scrollbar.pack(side="right", fill="y")
        self.settings_canvas.pack(side="left", fill="both", expand=True)

        # Luo sisäinen kehys Canvasin sisälle
        self.settings_inner_frame = tk.Frame(self.settings_canvas, background="#f0f0f0")
        self.settings_canvas.create_window((0, 0), window=self.settings_inner_frame, anchor="nw")

        # Päivitä vieritysominaisuutta, kun sisällön koko muuttuu
        self.settings_inner_frame.bind("<Configure>", self.on_frame_configure)


        # API Keys
        api_key_frame = tk.LabelFrame(self.settings_inner_frame, text="API Keys", padx=10, pady=10)
        api_key_frame.pack(fill=tk.X, padx=10, pady=(10, 0))

        self.openai_key_label = tk.Label(api_key_frame, text="OpenAI API Key:")
        self.openai_key_label.grid(row=0, column=0, sticky="w")
        self.openai_key_var = tk.StringVar()
        self.openai_key_entry = tk.Entry(api_key_frame, textvariable=self.openai_key_var, show="*", width=50)
        self.openai_key_entry.grid(row=0, column=1, padx=5, pady=5)

        self.openrouter_key_label = tk.Label(api_key_frame, text="OpenRouter API Key:")
        self.openrouter_key_label.grid(row=1, column=0, sticky="w")
        self.openrouter_key_var = tk.StringVar()
        self.openrouter_key_entry = tk.Entry(api_key_frame, textvariable=self.openrouter_key_var, show="*", width=50)
        self.openrouter_key_entry.grid(row=1, column=1, padx=5, pady=5)

        self.save_keys_button = tk.Button(api_key_frame, text="Save API Keys", command=self.save_api_keys, width=20)
        self.save_keys_button.grid(row=2, column=0, columnspan=2, pady=10)

        # --- OpenRouter URL ---
        self.openrouter_url_label = tk.Label(api_key_frame, text="OpenRouter API URL:")
        self.openrouter_url_label.grid(row=4, column=0, sticky="w")  # Row 4
        self.openrouter_url_var = tk.StringVar()
        self.openrouter_url_entry = tk.Entry(api_key_frame, textvariable=self.openrouter_url_var, width=50)
        self.openrouter_url_entry.grid(row=4, column=1, padx=5, pady=5)

        # --- Button to Save ---
        # (You already have the save_keys_button, we'll modify its command)
        self.save_keys_button.grid(row=5, column=0, columnspan=2, pady=10)  # Adjust row if needed

        # Language Selection
        language_frame = tk.LabelFrame(self.settings_inner_frame, text="Language Selection", padx=10, pady=10)
        language_frame.pack(fill=tk.X, padx=10, pady=(10, 0))

        self.language_label = tk.Label(language_frame, text="Select Language:", anchor="w")
        self.language_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)

        self.language_var = tk.StringVar(value="Finnish")
        self.language_combobox = ttk.Combobox(language_frame, textvariable=self.language_var, state="readonly", width=20)
        self.language_combobox['values'] = ("Finnish", "English", "Swedish", "Spanish")
        self.language_combobox.current(0)
        self.language_combobox.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        # After creating language_combobox in create_settings_tab method
        self.language_combobox.bind("<<ComboboxSelected>>", self.on_language_changed)

        # Default Image Path
        image_path_frame = tk.LabelFrame(self.settings_inner_frame, text="Default Image Path", padx=10, pady=10)
        image_path_frame.pack(fill=tk.X, padx=10, pady=(10, 0))

        self.image_path_var = tk.StringVar()
        self.image_path_entry = tk.Entry(image_path_frame, textvariable=self.image_path_var, width=50)
        self.image_path_entry.grid(row=0, column=0, padx=5, pady=5)

        # The line that references browse_image_path:
        self.browse_button = tk.Button(image_path_frame, text="Browse",
                                       command=self.browse_image_path, width=10)
        self.browse_button.grid(row=0, column=1, padx=5, pady=5)

        # a Save Path button
        self.save_image_path_button = tk.Button(
            image_path_frame, text="Save Path", command=self.save_image_path, width=10
        )
        self.save_image_path_button.grid(row=1, column=0, columnspan=2, pady=5)

        # Audio Parameters
        audio_params_frame = tk.LabelFrame(self.settings_inner_frame, text="Audio Parameters", padx=10, pady=10)
        audio_params_frame.pack(fill=tk.X, padx=10, pady=(10, 0))

        self.vol_threshold_label = tk.Label(audio_params_frame, text="Volume Threshold:")
        self.vol_threshold_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.vol_threshold_var = tk.StringVar(value=str(config.get('VOL_THRESHOLD', DEFAULT_VOL_THRESHOLD)))
        self.vol_threshold_entry = tk.Entry(audio_params_frame, textvariable=self.vol_threshold_var, width=10)
        self.vol_threshold_entry.grid(row=0, column=1, padx=5, pady=5)

        self.min_record_duration_label = tk.Label(audio_params_frame, text="Min Record Duration (s):")
        self.min_record_duration_label.grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.min_record_duration_var = tk.StringVar(
            value=str(config.get('MIN_RECORD_DURATION', DEFAULT_MIN_RECORD_DURATION))
        )
        self.min_record_duration_entry = tk.Entry(
            audio_params_frame, textvariable=self.min_record_duration_var, width=10
        )
        self.min_record_duration_entry.grid(row=1, column=1, padx=5, pady=5)

        self.max_silence_duration_label = tk.Label(audio_params_frame, text="Max Silence Duration (s):")
        self.max_silence_duration_label.grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.max_silence_duration_var = tk.StringVar(
            value=str(config.get('MAX_SILENCE_DURATION', DEFAULT_MAX_SILENCE_DURATION))
        )
        self.max_silence_duration_entry = tk.Entry(
            audio_params_frame, textvariable=self.max_silence_duration_var, width=10
        )
        self.max_silence_duration_entry.grid(row=2, column=1, padx=5, pady=5)

        self.save_audio_params_button = tk.Button(
            audio_params_frame, text="Save Parameters",
            command=self.save_audio_parameters, width=15
        )
        self.save_audio_params_button.grid(row=3, column=0, columnspan=2, pady=10)

        # Blacklist Management
        blacklist_frame = tk.LabelFrame(self.settings_inner_frame, text="Blacklist Management", padx=10, pady=10)
        blacklist_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 0))

        self.blacklist_listbox = tk.Listbox(blacklist_frame, selectmode=tk.SINGLE, width=50, height=10)
        self.blacklist_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10), pady=5)

        for phrase in self.excluded_phrases:
            self.blacklist_listbox.insert(tk.END, phrase)

        self.blacklist_scrollbar = tk.Scrollbar(blacklist_frame, orient="vertical")
        self.blacklist_scrollbar.config(command=self.blacklist_listbox.yview)
        self.blacklist_listbox.config(yscrollcommand=self.blacklist_scrollbar.set)
        self.blacklist_scrollbar.pack(side=tk.LEFT, fill=tk.Y)

        add_frame = tk.Frame(blacklist_frame)
        add_frame.pack(fill=tk.X, padx=5, pady=5)



        # --------------------- LLM Persoonallisuus ---------------------
        llm_personality_frame = tk.LabelFrame(self.settings_inner_frame, text="LLM Persoonallisuus", padx=10, pady=10)
        llm_personality_frame.pack(fill=tk.X, padx=10, pady=(10, 0))

        # Tekstilaatikko persoonallisuutta varten
        self.llm_personality_text = tk.Text(llm_personality_frame, height=5, width=50)
        self.llm_personality_text.pack(padx=5, pady=5)

        # Lataa mahdollisesti aiemmin tallennettu persoonallisuusteksti
        default_personality = self.config_data.get('LLM_PERSONALITY',
                                                   'Give the LLM a hint or a full description about its desired personality. Then apply as is or click "Help me create persona" to get longer more detailed description."')
        self.llm_personality_text.insert("1.0", default_personality)

        # Näppäin, jolla tallennetaan suoraan persoonallisuusteksti
        save_personality_button = tk.Button(llm_personality_frame, text="Save personality",
                                            command=self.save_llm_personality, width=20)
        save_personality_button.pack(padx=5, pady=5)

        # Uusi nappi: "Auta luomaan persoonallisuus:"
        generate_personality_button = tk.Button(llm_personality_frame, text="Help me creating this persona:",
                                                command=self.generate_personality_prompt, width=25)
        generate_personality_button.pack(padx=5, pady=5)

        # Lisää tekstikenttä persoonallisuuden nimelle
        name_frame = tk.Frame(llm_personality_frame)
        name_frame.pack(padx=5, pady=(5, 0), fill=tk.X)
        tk.Label(name_frame, text="Name of the personality:").pack(side=tk.LEFT, padx=(0, 5))
        self.personality_name_var = tk.StringVar()
        self.personality_name_entry = tk.Entry(name_frame, textvariable=self.personality_name_var, width=30)
        self.personality_name_entry.pack(side=tk.LEFT)



        # --------------------- Tallennetut LLM-persoonallisuudet ---------------------
        saved_personality_frame = tk.LabelFrame(self.settings_inner_frame, text="Saved personalities",
                                                padx=10, pady=10)
        saved_personality_frame.pack(fill=tk.BOTH, padx=10, pady=(10, 0))

        # Listbox, joka näyttää tallennetut persoonallisuudet
        self.saved_personalities_listbox = tk.Listbox(saved_personality_frame, height=5, width=40)
        self.saved_personalities_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5), pady=5)

        # Scrollbar listboxille
        saved_personality_scrollbar = tk.Scrollbar(saved_personality_frame, orient="vertical",
                                                   command=self.saved_personalities_listbox.yview)
        saved_personality_scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.saved_personalities_listbox.config(yscrollcommand=saved_personality_scrollbar.set)

        # Nappi, jolla valittu persoonallisuus ladataan tekstilaatikkoon
        load_personality_button = tk.Button(saved_personality_frame, text="Edit/Apply selected",
                                            command=self.load_selected_personality, width=20)
        load_personality_button.pack(side=tk.TOP, padx=5, pady=5)

        # Nappi tallentamaan ja poistamaan tallennettuja persoonallisuuksia (tarvittaessa)
        delete_personality_button = tk.Button(saved_personality_frame, text="Delete selected",
                                              command=self.delete_selected_personality, width=20)
        delete_personality_button.pack(side=tk.TOP, padx=5, pady=5)



        self.new_phrase_var = tk.StringVar()
        self.new_phrase_entry = tk.Entry(add_frame, textvariable=self.new_phrase_var, width=40)
        self.new_phrase_entry.pack(side=tk.LEFT, padx=(0, 5), pady=5)

        self.add_phrase_button = tk.Button(add_frame, text="Add Phrase", command=self.add_blacklist_phrase)
        self.add_phrase_button.pack(side=tk.LEFT, padx=(0, 5), pady=5)

        self.remove_phrase_button = tk.Button(
            blacklist_frame, text="Remove Selected",
            command=self.remove_blacklist_phrase
        )
        self.remove_phrase_button.pack(pady=(0, 5))

        

        # In the create_settings_tab method, add after another frame (e.g., after reasoning_frame)

        # --------------------- TTS Voice Selection ---------------------
        tts_voice_frame = tk.LabelFrame(self.settings_inner_frame, text="Text-to-Speech Voice", padx=10, pady=10)
        tts_voice_frame.pack(fill=tk.X, padx=10, pady=(10, 0))

        # Voice selection dropdown
        voice_label = tk.Label(tts_voice_frame, text="Select Voice:")
        voice_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)

        self.tts_voice_var = tk.StringVar(value="alloy")
        self.tts_voice_combobox = ttk.Combobox(
            tts_voice_frame,
            textvariable=self.tts_voice_var,
            state="readonly",
            width=20,
            values=["alloy", "echo", "fable", "onyx", "nova", "shimmer", "ash", "coral", "sage"]
        )
        self.tts_voice_combobox.grid(row=0, column=1, sticky="w", padx=5, pady=5)

        # Voice descriptions
        voice_descriptions = (
            "alloy: Versatile, neutral\n"
            "echo: Deep, resonant\n"
            "fable: British, warmth\n"
            "onyx: Deep, authoritative\n"
            "nova: Warm, natural female\n"
            "shimmer: Clear, optimistic\n"
            "ash: Clear explanation style\n"
            "coral: Warm, female style\n"
            "sage: Gentle, thoughtful"
        )
        voice_help = tk.Label(
            tts_voice_frame,
            text=voice_descriptions,
            font=("TkDefaultFont", 8),
            fg="gray",
            justify="left"
        )
        voice_help.grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=5)

        # Save button
        self.save_tts_voice_button = tk.Button(
            tts_voice_frame,
            text="Save TTS Settings",
            command=self.save_tts_settings,
            width=20
        )
        self.save_tts_voice_button.grid(row=2, column=0, columnspan=2, pady=10)

        # Load saved voice setting if available
        if 'TTS_VOICE' in self.config_data:
            self.tts_voice_var.set(self.config_data['TTS_VOICE'])

        # Load existing API keys/parameters
        self.load_existing_api_keys()

        # --------------------- Mureka API Settings ---------------------
        mureka_frame = tk.LabelFrame(self.settings_inner_frame, text="Mureka API Settings", padx=10, pady=10)
        mureka_frame.pack(fill=tk.X, padx=10, pady=(10, 0))

        # API URL
        self.mureka_url_label = tk.Label(mureka_frame, text="Mureka API URL:")
        self.mureka_url_label.grid(row=0, column=0, sticky="w")
        self.mureka_url_var = tk.StringVar()
        self.mureka_url_entry = tk.Entry(mureka_frame, textvariable=self.mureka_url_var, width=50)
        self.mureka_url_entry.grid(row=0, column=1, padx=5, pady=5)

        # Account
        self.mureka_account_label = tk.Label(mureka_frame, text="Mureka Account:")
        self.mureka_account_label.grid(row=1, column=0, sticky="w")
        self.mureka_account_var = tk.StringVar()
        self.mureka_account_entry = tk.Entry(mureka_frame, textvariable=self.mureka_account_var, width=50)
        self.mureka_account_entry.grid(row=1, column=1, padx=5, pady=5)

        # API Token
        self.mureka_token_label = tk.Label(mureka_frame, text="Mureka API Token:")
        self.mureka_token_label.grid(row=2, column=0, sticky="w")
        self.mureka_token_var = tk.StringVar()  # Hide the token
        self.mureka_token_entry = tk.Entry(mureka_frame, textvariable=self.mureka_token_var, width=50, show="*")
        self.mureka_token_entry.grid(row=2, column=1, padx=5, pady=5)

        # Save Mureka Keys Button
        self.save_mureka_keys_button = tk.Button(mureka_frame, text="Save Mureka Keys", command=self.save_mureka_keys,
                                                 width=20)
        self.save_mureka_keys_button.grid(row=3, column=0, columnspan=2, pady=10)

        # --------------------- Added Audio Source 1 and 2 Selection ---------------------
        # Audio Source 1 Selection
        audio_source1_frame = tk.LabelFrame(self.settings_inner_frame, text="Microphone (Input) IMPORTANT: SAME AS WINDOWS MICROPHONE", padx=10, pady=10)
        audio_source1_frame.pack(fill=tk.X, padx=10, pady=(10, 0))

        audio_source1_label = tk.Label(audio_source1_frame, text="Select Input Device:")
        audio_source1_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)

        self.available_input_devices1 = self.get_input_audio_devices_for_combobox()
        self.audio_source1_var = tk.StringVar()
        self.audio_source1_combobox = ttk.Combobox(
            audio_source1_frame,
            textvariable=self.audio_source1_var,
            state="readonly",
            width=50,
            values=[item[0] for item in self.available_input_devices1]
        )
        self.audio_source1_combobox.grid(row=0, column=1, padx=5, pady=5)
        if self.available_input_devices1:
            self.audio_source1_combobox.current(0)

        # Audio Source 2 Selection
        audio_source2_frame = tk.LabelFrame(self.settings_inner_frame, text="System audio (Input) IMPORTANT: 1. SET this and WINDOWS PLAYBACK DEVICE IN SETTINGS TO BE FOR EXAMPLE ...steam streaming speakers... IF YOU WANT THE LLM TO HEAR SYSTEM AUDIO. 2. If does not work: IN WINDOWS AUDIO SETTINGS SET SAMPLE RATE 16-bit 44100Hz. ", padx=10, pady=10)
        audio_source2_frame.pack(fill=tk.X, padx=10, pady=(10, 0))

        audio_source2_label = tk.Label(audio_source2_frame, text="Select Input Device:")
        audio_source2_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)

        self.available_input_devices2 = self.get_input_audio_devices_for_combobox()
        self.audio_source2_var = tk.StringVar()
        self.audio_source2_combobox = ttk.Combobox(
            audio_source2_frame,
            textvariable=self.audio_source2_var,
            state="readonly",
            width=50,
            values=[item[0] for item in self.available_input_devices2]
        )
        self.audio_source2_combobox.grid(row=0, column=1, padx=5, pady=5)
        if self.available_input_devices2:
            self.audio_source2_combobox.current(0)

        # Audio Output Selection
        audio_output_frame = tk.LabelFrame(self.settings_inner_frame, text="Audio Output (Playback) IMPORTANT: CHOOSE A DEVICE THAT CAN PLAY AUDIO OUT LOUD AND IS CONNECTED TO THE COMPUTER", padx=10, pady=10)
        audio_output_frame.pack(fill=tk.X, padx=10, pady=(10, 0))

        audio_output_label = tk.Label(audio_output_frame, text="Select Output Device:")
        audio_output_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)

        self.available_output_devices = self.get_output_audio_devices_for_combobox()
        self.audio_output_var = tk.StringVar()
        self.audio_output_combobox = ttk.Combobox(
            audio_output_frame,
            textvariable=self.audio_output_var,
            state="readonly",
            width=50,
            values=[item[0] for item in self.available_output_devices]
        )
        self.audio_output_combobox.grid(row=0, column=1, padx=5, pady=5)
        if self.available_output_devices:
            self.audio_output_combobox.current(0)




        self.load_mureka_settings()  # Ladataan Mureka asetukset

        # --------------------- AI Reasoning Controls ---------------------
        # Add this section before or after one of your existing sections (e.g., after Mureka API Settings)
        reasoning_frame = tk.LabelFrame(self.settings_inner_frame, text="AI Reasoning Controls", padx=10, pady=10)
        reasoning_frame.pack(fill=tk.X, padx=10, pady=(10, 0))

        # Reasoning effort selection
        reasoning_effort_label = tk.Label(reasoning_frame, text="Reasoning Effort:")
        reasoning_effort_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)

        self.reasoning_effort_var = tk.StringVar(value="medium")
        self.reasoning_effort_combobox = ttk.Combobox(
            reasoning_frame,
            textvariable=self.reasoning_effort_var,
            state="readonly",
            width=20,
            values=["low", "medium", "high"]
        )
        self.reasoning_effort_combobox.grid(row=0, column=1, sticky="w", padx=5, pady=5)

        # Show reasoning checkbox
        self.show_reasoning_var = tk.BooleanVar(value=True)
        self.show_reasoning_check = ttk.Checkbutton(
            reasoning_frame,
            text="Show AI reasoning process",
            variable=self.show_reasoning_var
        )
        self.show_reasoning_check.grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=5)

        # Add a save button to the reasoning frame
        self.save_reasoning_button = tk.Button(
            reasoning_frame,
            text="Save Reasoning Settings",
            command=self.save_reasoning_settings,
            width=20
        )
        self.save_reasoning_button.grid(row=3, column=0, columnspan=2, pady=10)

        # Load reasoning settings
        if 'REASONING_EFFORT' in self.config_data:
            self.reasoning_effort_var.set(self.config_data['REASONING_EFFORT'])
        if 'SHOW_REASONING' in self.config_data:
            self.show_reasoning_var.set(self.config_data['SHOW_REASONING'])

        # Help text
        help_text = ("Low effort: Basic reasoning\n"
                    "Medium effort: Balanced reasoning\n"
                    "High effort: Detailed, multi-step reasoning")
        help_label = tk.Label(
            reasoning_frame,
            text=help_text,
            font=("TkDefaultFont", 8),
            fg="gray"
        )
        help_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=5)

    # Add this method to your ChatAudioApp5 class


    def on_audio_source_change(self, *args):
        """Update last_audio_source when audio source changes, excluding mute."""
        current = self.audio_source_toggle_var.get()
        if current != "mute":
            self.last_audio_source = current    

    def on_language_changed(self, event):
        """Handle language selection changes and restart listening if active"""
        new_language = self.language_var.get()
        self.append_message(f"Language changed to: {new_language}", sender="system")
        
        # Restart the listening process with the new language
        if hasattr(self, 'processing_thread') and self.processing_thread and self.processing_thread.is_alive():
            # Stop current listening process
            self.stop_listening()
            # Small delay to ensure cleanup completes
            self.after(500, self.start_listening)
        else:
            # If not already running, just update the status
            self.append_message("Language will be used for the next conversation", sender="system")

    def on_model_changed(self, event):
        """Handle model selection changes and restart listening if active"""
        new_model = self.model_var.get()
        self.append_message(f"Model changed to: {new_model}", sender="system")

        # Restart the listening process with the new model
        if hasattr(self, 'processing_thread') and self.processing_thread and self.processing_thread.is_alive():
            # Stop current listening process
            self.stop_listening()
            # Small delay to ensure cleanup completes
            self.after(500, self.start_listening)
        else:
            # If not already running, just update the status
            self.append_message("Model will be used for the next conversation", sender="system")        


    def save_tts_settings(self):
        """Save TTS voice settings to config file"""
        voice = self.tts_voice_var.get()
        self.config_data['TTS_VOICE'] = voice
        
        # Save to config file
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config_data, f, indent=4)
        
        messagebox.showinfo("Settings Saved", "TTS voice settings saved successfully.")

    def save_reasoning_settings(self):
        self.config_data['REASONING_EFFORT'] = self.reasoning_effort_var.get()
        self.config_data['SHOW_REASONING'] = self.show_reasoning_var.get()
        
        # Save to config file
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config_data, f, indent=4)
        
        messagebox.showinfo("Settings Saved", "Reasoning settings saved successfully.")

    # Lisätään metodi Mureka-asetusten tallentamiseen:
    def save_mureka_keys(self):
        mureka_url = self.mureka_url_var.get().strip()
        mureka_account = self.mureka_account_var.get().strip()
        mureka_token = self.mureka_token_var.get().strip()

        if not mureka_url or not mureka_account or not mureka_token:
            messagebox.showerror("Error", "All Mureka API fields must be filled.")
            return

        if not messagebox.askyesno("Confirm", "Are you sure you want to save the provided Mureka API keys?"):
            return

        self.config_data['MUREKA_API_URL'] = mureka_url
        self.config_data['MUREKA_ACCOUNT'] = mureka_account
        self.config_data['MUREKA_API_TOKEN'] = mureka_token

        save_config(self.config_data)
        messagebox.showinfo("Success", "Mureka API keys saved successfully.")

        self.mureka_token_var.set("******")  # Don't display the actual token
        self.append_message("Mureka API keys loaded successfully.", sender="system")

    # Lisätään metodi Mureka-asetusten lataamiseen:
    def load_mureka_settings(self):
        mureka_url = self.config_data.get('MUREKA_API_URL', '')
        mureka_account = self.config_data.get('MUREKA_ACCOUNT', '')
        mureka_token = self.config_data.get('MUREKA_API_TOKEN', '')

        self.mureka_url_var.set(mureka_url)
        self.mureka_account_var.set(mureka_account)
        if mureka_token:
            self.mureka_token_entry.insert(0, "******") # Oikein. Lisätään Entry-widgettiin
            self.mureka_token_entry.config(show="*")  # Varmistetaan, että Entry näyttää tähdet

        


    def browse_image_path(self):
        """Let the user pick a default image directory."""
        selected_path = filedialog.askdirectory(
            initialdir=self.image_path_var.get() or DEFAULT_IMAGE_DIR,
            title="Select Default Image Directory"
        )
        if selected_path:
            self.image_path_var.set(selected_path)

    def launch_app2(self):
        """Launch App2 (PDF extractor) as a separate process with correct paths"""
        try:
            # Get the directory of the current script (main app)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            app2_path = os.path.join(script_dir, "SimplerPDFextract.py")
            
            # Launch using current Python executable and absolute path
            subprocess.Popen([sys.executable, app2_path])
            self.append_message("Launched PDF Extractor", sender="system")
        except Exception as e:
            messagebox.showerror("Error", f"Could not start App2: {e}")

    def launch_app3(self):
        """Launch App3 (Keyboard TTS) with proper paths"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            app3_path = os.path.join(script_dir, "keyboardtts.py")
            subprocess.Popen([sys.executable, app3_path])
            self.append_message("Launched Keyboard TTS", sender="system")
        except Exception as e:
            messagebox.showerror("Error", f"Could not start keyboard TTS: {e}")

            

    def map_language_to_code(self, lang_name):
        lookup = {
            "Finnish": "fi",
            "Swedish": "sv",
            "Spanish": "es",
            "English": "en"
        }
        return lookup.get(lang_name, "en")

    def get_input_audio_devices_for_combobox(self):
        devices = []
        try:
            all_devices = sd.query_devices()
            default_host_api = sd.query_hostapis(sd.default.hostapi)
            host_api_name = default_host_api['name'].lower()

            for idx, dev in enumerate(all_devices):
                if dev['max_input_channels'] > 0:
                    is_loopback = False
                    if 'windows wasapi' in host_api_name:
                        if 'loopback' in dev['name'].lower():
                            is_loopback = True
                    label = f"{dev['name']} (Chans: {dev['max_input_channels']})"
                    devices.append((label, idx, is_loopback))
        except Exception as e:
            messagebox.showerror("Error Listing Devices", f"Could not list input audio devices:\n{e}")

        if not devices:
            devices.append(("No valid input devices found", None, False))

        return devices

    def get_output_audio_devices_for_combobox(self):
        devices = []
        try:
            all_devices = sd.query_devices()
            for idx, dev in enumerate(all_devices):
                if dev['max_output_channels'] > 0:
                    label = f"{dev['name']} (Output Chans: {dev['max_output_channels']})"
                    devices.append((label, idx))
        except Exception as e:
            messagebox.showerror("Error Listing Devices", f"Could not list output audio devices:\n{e}")

        if not devices:
            devices.append(("No valid output devices found", None))

        return devices  


    def toggle_listening_state(self):
        """Toggle between listening and paused states, stopping/starting streams."""
        current_state = self.audio_source_toggle_var.get()
        if current_state == "mute":
            # Resuming listening
            self.audio_source_toggle_var.set(self.last_audio_source)
            self.start_listening()
            self.listening_status_label.config(text="Listening Status: Active", fg="green")
        else:
            # Pausing listening
            self.last_audio_source = current_state
            self.stop_listening()
            self.audio_source_toggle_var.set("mute")
            self.listening_status_label.config(text="Listening Status: Paused", fg="red")

    def start_listening(self):
        try:
            if not self.api_keys_provided:
                messagebox.showerror("Error", "API keys are not set. Please enter and save your API keys first.")
                return

            selected_language = self.map_language_to_code(self.language_var.get())
            selected_model = self.model_var.get()
            if not selected_model:
                messagebox.showerror("Error", "Please select an LLM model from the dropdown.")
                return

            # Retrieve device 1 and device 2 from comboboxes
            selected_index1 = self.audio_source1_combobox.current()
            selected_index2 = self.audio_source2_combobox.current()

            if selected_index1 < 0 or not self.available_input_devices1:
                messagebox.showerror("Error", "Please select a valid audio input device for source 1.")
                return

            if selected_index2 < 0 or not self.available_input_devices2:
                messagebox.showerror("Error", "Please select a valid audio input device for source 2.")
                return

            device_label1, device_id1, is_loopback1 = self.available_input_devices1[selected_index1]
            device_label2, device_id2, is_loopback2 = self.available_input_devices2[selected_index2]

            # Determine sample rates and channels for both
            self.sample_rate1 = self.get_device_sample_rate(device_id1)
            channels1 = min(self.get_device_input_channels(device_id1), 2)
            channels1 = max(channels1, 1)

            self.sample_rate2 = self.get_device_sample_rate(device_id2)
            channels2 = min(self.get_device_input_channels(device_id2), 2)
            channels2 = max(channels2, 1)

            # Create callbacks
            callback1 = make_audio_callback(self.audio_queue1)
            callback2 = make_audio_callback(self.audio_queue2)

            # Start worker thread
            self.processing_thread = self.start_thread(
                target=process_audio_stream,
                args=(
                    self.stream_stop_event,
                    self,
                    selected_language,
                    self.openai_key,
                    self.openrouter_key,
                    selected_model
                ),
                thread_name="AudioProcessingThread"
            )

            # Create and start streams
            if is_loopback1:
                wasapi_extras1 = sd.WasapiSettings(loopback=True)
                self.stream1 = sd.InputStream(
                    callback=callback1,
                    channels=channels1,
                    samplerate=self.sample_rate1,
                    device=device_id1,
                    dtype='int16',
                    extra_settings=wasapi_extras1
                )
            else:
                self.stream1 = sd.InputStream(
                    callback=callback1,
                    channels=channels1,
                    samplerate=self.sample_rate1,
                    device=device_id1,
                    dtype='int16'
                )

            if is_loopback2:
                wasapi_extras2 = sd.WasapiSettings(loopback=True)
                self.stream2 = sd.InputStream(
                    callback=callback2,
                    channels=channels2,
                    samplerate=self.sample_rate2,
                    device=device_id2,
                    dtype='int16',
                    extra_settings=wasapi_extras2
                )
            else:
                self.stream2 = sd.InputStream(
                    callback=callback2,
                    channels=channels2,
                    samplerate=self.sample_rate2,
                    device=device_id2,
                    dtype='int16'
                )

            self.stream1.start()
            self.stream2.start()

            self.update_status("Listening...")
            self.append_message("Started listening.", sender="system")
            print("Listening started.")
            self.append_message(
                f"Recording from devices:\nAS1: {device_label1} at {self.sample_rate1} Hz, {channels1} channel(s).\n"
                f"AS2: {device_label2} at {self.sample_rate2} Hz, {channels2} channel(s).",
                sender="system"
            )

        except Exception as e:
            logging.error(f"Failed to start listening: {e}")
            messagebox.showerror("Error", f"Failed to start listening: {e}")
            self.append_message(f"Failed to start listening: {e}", sender="system")

    def stop_listening(self):
        try:
            self.stream_stop_event.set()
            if self.stream1:
                self.stream1.stop()
                self.stream1.close()
                self.stream1 = None
            if self.stream2:
                self.stream2.stop()
                self.stream2.close()
                self.stream2 = None

            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=1)

            #Reset listening queues
            while not self.audio_queue1.empty():
                self.audio_queue1.get()
            while not self.audio_queue2.empty():
                self.audio_queue2.get()
            self.stream_stop_event = threading.Event()
            time.sleep(0.5)

            self.update_status("Idle")
            self.append_message("Stopped listening.", sender="system")
            print("Listening stopped.")
            self.listening_stopped = True
        except Exception as e:
            logging.error(f"Failed to stop listening: {e}")
            messagebox.showerror("Error", f"Failed to stop listening: {e}")
            self.append_message(f"Failed to stop listening: {e}", sender="system")

    # --------------------------- Chess UI ---------------------------
    def create_chess_frame(self, parent):
        """Build the frame holding the chess canvas and controls."""
        self.chess_frame = tk.Frame(parent)

        # Chess board canvas frame
        self.chess_canvas_frame = tk.Frame(self.chess_frame)
        self.chess_canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=False, padx=10, pady=5)

        self.chess_canvas = tk.Canvas(
            self.chess_canvas_frame,
            width=self.square_size * 8,
            height=self.square_size * 8
        )
        self.chess_canvas.pack()
        self.chess_canvas.bind("<Button-1>", self.on_chess_canvas_click)

        # Controls panel below the chess board
        controls_frame = tk.Frame(self.chess_frame)
        controls_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # Left column: buttons
        left_controls = tk.Frame(controls_frame)
        left_controls.grid(row=0, column=0, sticky="w", padx=(0, 20))

        self.ai_move_button = tk.Button(
            left_controls,
            text="Ask AI for a Move",
            command=self.retry_ai_move,
            width=15,
            state=tk.NORMAL,
        )
        self.ai_move_button.pack(side=tk.TOP, pady=5)

        self.reset_game_button = tk.Button(
            left_controls,
            text="Reset game",
            command=self.reset_chess_game,
            width=15,
            bg="#FF5733",
            fg="white"
        )
        self.reset_game_button.pack(side=tk.TOP, pady=5)

        # Right column: checkboxes
        right_controls = tk.Frame(controls_frame)
        right_controls.grid(row=0, column=1, sticky="e")

        self.ai_only_valid_moves_check = tk.Checkbutton(
            right_controls,
            text=("Include a list of legal moves\nwith your move\n"
                "this will be sent to the AI\nDisable this for more interesting gameplay."),
            justify=tk.LEFT,
            variable=self.show_ai_only_valid_moves_var,
            command=self.update_ai_only_valid_moves
        )
        self.ai_only_valid_moves_check.pack(side=tk.TOP, pady=5)

        self.include_explanation_var = tk.BooleanVar(value=True)
        self.include_explanation_check = tk.Checkbutton(
            right_controls,
            text=("Ask AI to include verbal explanation\nfor the move and speak the explanation out loud"),
            justify=tk.LEFT,
            variable=self.include_explanation_var
        )
        self.include_explanation_check.pack(side=tk.TOP, pady=5)

    def show_chess_ui(self):
        self.chess_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

    def hide_chess_ui(self):
        if self.chess_frame is not None:
            self.chess_frame.pack_forget()
            
    def load_chess_images(self):
        piece_names = {
            'K': 'King',
            'Q': 'Queen',
            'B': 'Bishop',
            'N': 'Knight',
            'R': 'Rook',
            'P': 'Pawn'
        }
        colors = {'white': 'w', 'black': 'b'}

        for color in colors:
            for piece in piece_names:
                try:
                    path = f"chess_images/{color}/{piece_names[piece]}.png"
                    raw_img = Image.open(path).convert("RGBA")
                    self.raw_chess_images[f"{colors[color]}{piece}"] = raw_img
                    resized_img = raw_img.resize((self.square_size, self.square_size), Image.Resampling.LANCZOS)
                    self.chess_images[f"{colors[color]}{piece}"] = ImageTk.PhotoImage(resized_img)
                except FileNotFoundError:
                    print(f"Image not found: {path}")
                    self.raw_chess_images[f"{colors[color]}{piece}"] = None
                    self.chess_images[f"{colors[color]}{piece}"] = None

    def draw_chessboard(self):
        if not self.chess_canvas:
            return

        self.chess_canvas.delete("all")
        for row in range(8):
            for col in range(8):
                x1 = col * self.square_size
                y1 = row * self.square_size
                square_index = chess.square(col, 7 - row)
                
                # Determine the base color of the square
                color = "#F0D9B5" if (row + col) % 2 == 0 else "#B58863"
                
                # Draw the square
                self.chess_canvas.create_rectangle(x1, y1, x1 + self.square_size, y1 + self.square_size, fill=color)
                
                # If this is the selected square, add a highlight border
                if self.selected_square == square_index:
                    self.chess_canvas.create_rectangle(
                        x1 + 2, y1 + 2, 
                        x1 + self.square_size - 2, y1 + self.square_size - 2,
                        outline="#32CD32",  # Lime green highlight
                        width=3
                    )

                # Draw the piece if present
                piece = self.chess_board.piece_at(square_index)
                if piece:
                    piece_key = f"{'w' if piece.color == chess.WHITE else 'b'}{piece.symbol().upper()}"
                    if self.chess_images.get(piece_key):
                        self.chess_canvas.create_image(
                            x1 + self.square_size // 2,
                            y1 + self.square_size // 2,
                            image=self.chess_images[piece_key]
                        )
                    else:
                        self.chess_canvas.create_text(
                            x1 + self.square_size // 2,
                            y1 + self.square_size // 2,
                            text=piece.symbol(),
                            font=("Arial", 24),
                            fill="black" if piece.color == chess.BLACK else "white"
                        )

    def on_chess_canvas_click(self, event):
        col = event.x // self.square_size
        row = event.y // self.square_size
        square_index = (7 - row) * 8 + col

        # If no square is selected yet
        if self.selected_square is None:
            # Check if there's a piece on the clicked square
            piece = self.chess_board.piece_at(square_index)
            if piece and piece.color == self.chess_board.turn:
                self.selected_square = square_index
                # Redraw the board to show the selected square
                self.draw_chessboard()
        else:
            # If clicking on the same square, unselect it
            if square_index == self.selected_square:
                self.selected_square = None
                self.draw_chessboard()
            else:
                # Check if clicking on another piece of the same color (change selection)
                piece_at_clicked = self.chess_board.piece_at(square_index)
                piece_at_selected = self.chess_board.piece_at(self.selected_square)
                
                if piece_at_clicked and piece_at_selected and piece_at_clicked.color == piece_at_selected.color:
                    # Change selection to the new piece
                    self.selected_square = square_index
                    self.draw_chessboard()
                else:
                    # Try to make a move from selected square to clicked square
                    from_sq = self.selected_square
                    to_sq = square_index
                    self.selected_square = None
                    self.try_user_move(from_sq, to_sq)

    def try_user_move(self, from_sq, to_sq, promotion=None):
        piece = self.chess_board.piece_at(from_sq)

        # Tarkista onko kyseessä sotilaan korotus
        if piece and piece.piece_type == chess.PAWN and chess.square_rank(to_sq) in [0, 7]:
            if promotion is None:
                promotion = self.prompt_promotion()
                if promotion is None:
                    self.append_message("Promotion cancelled.", sender="system")
                    self.selected_square = None  # Reset selection when promotion is cancelled
                    self.draw_chessboard()  # Redraw board to remove highlighting
                    return False  # Palauta False, koska siirto peruutettiin
            else:
                # Muunna mahdollinen puheesta saatava merkkijono vastaavaksi kokonaisluvuksi
                promotion = promotion.lower()
                if promotion == 'q':
                    promotion = chess.QUEEN
                elif promotion == 'r':
                    promotion = chess.ROOK
                elif promotion == 'b':
                    promotion = chess.BISHOP
                elif promotion == 'n':
                    promotion = chess.KNIGHT
                else:
                    self.append_message("Faulty promotion choise.", sender="system")
                    self.selected_square = None  # Reset selection on invalid promotion
                    self.draw_chessboard()
                    return False

        # Luo siirto olioksi oikealla korotuksella
        move = chess.Move(from_sq, to_sq, promotion=promotion) if promotion else chess.Move(from_sq, to_sq)

        if move in self.chess_board.legal_moves:
            self.chess_board.push(move)
            self.append_message(f"User move: {move.uci()}", sender="user")
            self.draw_chessboard()

            # Save game state after the move
            self.save_chess_game_state()

            # Lisää pelin tilan tarkistus siirron jälkeen
            self.check_game_over()
            if not self.chess_game_ended:
                self.send_chess_state_to_llm(user_move=move.uci())

            return True  # Palauta True, koska siirto on laillinen
        else:
            self.append_message("That move is illegal.", sender="system")
            self.selected_square = None  # Reset selection after illegal move
            self.draw_chessboard()  # Redraw board to remove highlighting
            return False  # Palauta False, koska siirto on laiton

    def prompt_promotion(self):
        """
        Näyttää dialogin, jossa käyttäjä voi valita korotuksen nappulan.
        Palauttaa valitun nappulan tyypin tai None, jos korotus perutaan.
        """

        def on_select(piece):
            selected_piece[0] = piece
            promo_window.destroy()

        promo_window = tk.Toplevel(self)
        promo_window.title("Choose promotion")
        promo_window.geometry("500x200")
        promo_window.resizable(False, False)
        promo_window.grab_set()  # Estää muiden ikkunoiden käytön ennen tämän ikkunan sulkeutumista

        label = tk.Label(promo_window, text="Choose promotion:", font=("Arial", 12))
        label.pack(pady=10)

        button_frame = tk.Frame(promo_window)
        button_frame.pack(pady=5)

        selected_piece = [None]  # Lista käytetään mutaation mahdollistamiseksi suljetussa scope:ssa

        pieces = {
            "Queen": chess.QUEEN,
            "Rook": chess.ROOK,
            "Bishop": chess.BISHOP,
            "Knight": chess.KNIGHT
        }

        for name, piece in pieces.items():
            btn = tk.Button(button_frame, text=name, width=10, command=lambda p=piece: on_select(p))
            btn.pack(side=tk.LEFT, padx=5)

        cancel_button = tk.Button(promo_window, text="Cancel", width=10, command=promo_window.destroy)
        cancel_button.pack(pady=5)

        self.wait_window(promo_window)  # Odottaa ikkunan sulkeutumista

        return selected_piece[0]

    def generate_chessboard_image(self):
        board_size = 8 * self.square_size
        board_image = Image.new("RGB", (board_size, board_size), "white")
        draw = ImageDraw.Draw(board_image)

        for row in range(8):
            for col in range(8):
                color = "#F0D9B5" if (row + col) % 2 == 0 else "#B58863"
                x1 = col * self.square_size
                y1 = row * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                draw.rectangle([x1, y1, x2, y2], fill=color)

                square = chess.square(col, 7 - row)
                piece = self.chess_board.piece_at(square)
                if piece:
                    piece_key = f"{'w' if piece.color == chess.WHITE else 'b'}{piece.symbol().upper()}"
                    piece_img = self.raw_chess_images.get(piece_key)
                    if piece_img:
                        resized = piece_img.resize((self.square_size, self.square_size), Image.Resampling.LANCZOS)
                        board_image.paste(resized, (x1, y1), mask=resized)

        output_path = DEFAULT_IMAGE_DIR / "chessboard.png"
        board_image.save(str(output_path), "PNG")
        return str(output_path)

    def send_chess_state_to_llm(self, user_move=None):
        """
        Send the current chess state to the LLM in a non-blocking way.
        """

        # Create a placeholder message for showing AI thinking
        self.append_message("Thinking about my next move...", sender="assistant")
        
        # Run the LLM call in a background thread to avoid UI freezing
        self.start_thread(
            target=self._background_chess_llm_call,
            args=(user_move,),
            thread_name="ChessAIThread"
        )

    def _background_chess_llm_call(self, user_move):
        """Background thread for processing chess state with LLM"""
        global messages
        fen = self.chess_board.fen()
        legal_moves = [move.uci() for move in self.chess_board.legal_moves]
        legal_moves_str = ", ".join(legal_moves)

        # Get selected language
        selected_language = self.language_var.get()

        # Get prompts based on language
        fen_prompt = LANGUAGE_PROMPTS_FEN.get(selected_language, 
            "Please analyze the following chess position (FEN: ")
        
        after_fen_prompt_a = LANGUAGE_PROMPTS_AFTER_FEN_A.get(selected_language, 
            "Internally deliberate to determine the best move")
            
        after_fen_prompt_b = LANGUAGE_PROMPTS_AFTER_FEN_A.get(selected_language, 
            ", then verify your chosen move is in the allowed moves list.")

        # Set explanation prompt based on checkbox
        if not self.include_explanation_var.get():
            explanation_prompt = LANGUAGE_PROMPTS_NO_EXPLANATION.get(selected_language)
        else:
            explanation_prompt = LANGUAGE_PROMPTS_EXPLAIN_MOVE.get(selected_language)

        # Build the prompt
        if self.show_ai_only_valid_moves:
            board_info = (
                f"{fen_prompt}{fen}). "
                f"{after_fen_prompt_a}{after_fen_prompt_b}"
                f"{explanation_prompt}"
                f"Allowed moves: {legal_moves_str}"
            )
        else:
            board_info = (
                f"Please suggest a move for black in this chess position (FEN: {fen}). "
                f"GIVE THE MOVE IN CORRECT FORMAT (examples: a1a2, a7a8q, o-o-o, o-o) "
                f"AS THE FIRST THING IN YOUR REPLY. {explanation_prompt}"
            )

        if user_move:
            board_info += f"\nUser's last move: {user_move}"

        messages.append({"role": "user", "content": board_info})

        # Generate and send image if needed
        image_path = self.generate_chessboard_image() if self.send_image_var.get() else None
        if image_path:
            self.after(0, self.append_image_in_chat, image_path, "user")

        # Get LLM response with reasoning settings
        response_json = self.send_message(
            messages=messages,
            image_path=image_path,
            language=self.language_var.get(),
            openrouter_api_key=self.openrouter_key,
            model=self.model_var.get(),
            reasoning_effort=self.reasoning_effort_var.get(),  # Pass reasoning settings
            show_reasoning=self.show_reasoning_var.get()
        )

        # Process AI response (in the background thread)
        if response_json:
            try:
                assistant_response = response_json['choices'][0]['message']['content']
                
                # Use after() to update UI from background thread
                self.after(0, lambda: self._process_chess_ai_response(assistant_response))
            except (IndexError, KeyError) as e:
                self.after(0, lambda: self.handle_error(f"Error processing AI response: {e}"))
                
    def _process_chess_ai_response(self, assistant_response):
        """Process the AI's response in the main thread"""
        global messages
        
        # Blocks normal chat during chess game
        self.ai_tried_chess_move = True
        
        # Extract and make AI move
        move = self.extract_move(assistant_response)
        if move:
            self.chess_board.push(move)
            self.append_message(f"AI moved: {move.uci()}", sender="system")
            self.draw_chessboard()
            
            # Save game state after AI move
            self.save_chess_game_state()

            messages.append({"role": "assistant", "content": assistant_response})

            # Draw current chess state
            image_path = self.generate_chessboard_image() if self.send_image_var.get() else None
            if image_path:
                self.append_image_in_chat(image_path, sender="system")
        else:
            self.append_message("AI did not provide a valid move.", sender="system")
            # Instead of just enabling the button, ask if user wants to retry
            if messagebox.askyesno("Invalid Move", "The AI provided an invalid move. Would you like the AI to try again?"):
                self.retry_ai_move()

    def retry_ai_move(self):
        """Retry AI move using streaming API"""
        # Create a placeholder message for showing AI thinking
        self.append_message("Trying to find a valid chess move...", sender="assistant")
        
        # Run the LLM call in a background thread to avoid UI freezing
        threading.Thread(
            target=self._background_retry_chess_move,
            daemon=True
        ).start()

    def _background_retry_chess_move(self):
        """Background thread for retrying AI chess move with streaming"""
        fen = self.chess_board.fen()
        legal_moves = [move.uci() for move in self.chess_board.legal_moves]
        legal_moves_str = ", ".join(legal_moves)

        # Get selected language
        selected_language = self.language_var.get()

        # Get prompts based on language
        fen_prompt = LANGUAGE_PROMPTS_FEN.get(selected_language, 
            "Please analyze the following chess position (FEN: ")
        
        after_fen_prompt_a = LANGUAGE_PROMPTS_AFTER_FEN_A.get(selected_language, 
            "Internally deliberate to determine the best move")
            
        after_fen_prompt_b = LANGUAGE_PROMPTS_AFTER_FEN_A.get(selected_language, 
            ", then verify your chosen move is in the allowed moves list.")

        # Set explanation prompt based on checkbox
        if not self.include_explanation_var.get():
            explanation_prompt = LANGUAGE_PROMPTS_NO_EXPLANATION.get(selected_language)
        else:
            explanation_prompt = LANGUAGE_PROMPTS_EXPLAIN_MOVE.get(selected_language)

        # Build a stronger prompt for retry
        retry_prompt = (
            f"{fen_prompt}{fen}). "
            f"{after_fen_prompt_a}{after_fen_prompt_b}"
            f"{explanation_prompt}"
            f"The previous attempt didn't provide a valid move. Please choose a legal move from this list: {legal_moves_str}"
            f"The first move in your reply MUST be a valid move from the list."
        )
        
        # Create a temporary messages list for this request
        retry_messages = [{"role": "user", "content": retry_prompt}]

        # Generate and send image if needed
        image_path = self.generate_chessboard_image() if self.send_image_var.get() else None
        if image_path:
            self.after(0, self.append_image_in_chat, image_path, "user")

        # Get LLM response with reasoning settings
        response_json = self.send_message(
            messages=retry_messages,
            image_path=image_path,
            language=self.language_var.get(),
            openrouter_api_key=self.openrouter_key,
            model=self.model_var.get(),
            reasoning_effort=self.reasoning_effort_var.get(),
            show_reasoning=self.show_reasoning_var.get()
        )

        # Process AI response (in the background thread)
        if response_json:
            try:
                assistant_response = response_json['choices'][0]['message']['content']
                
                # Use after() to update UI from background thread
                self.after(0, lambda: self._process_chess_ai_response(assistant_response))
            except (IndexError, KeyError) as e:
                self.after(0, lambda: self.handle_error(f"Error processing AI response: {e}"))

    
    def save_chess_game_state(self):
        """Save the current chess game state to a file."""
        if not self.chess_board or self.chess_game_ended:
            # Don't save if the game is ended or board doesn't exist
            if os.path.exists(CHESS_SAVE_FILE):
                os.remove(CHESS_SAVE_FILE)
            return
            
        game_state = {
            "fen": self.chess_board.fen(),
            "moves": [move.uci() for move in self.chess_board.move_stack],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "game_ended": self.chess_game_ended
        }
        
        try:
            with open(CHESS_SAVE_FILE, 'w') as f:
                json.dump(game_state, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save chess game state: {e}")
    
    def load_chess_game_state(self):
        """Load a previously saved chess game if it exists."""
        if not os.path.exists(CHESS_SAVE_FILE):
            return False
            
        try:
            with open(CHESS_SAVE_FILE, 'r') as f:
                game_state = json.load(f)
                
            # Don't load completed games
            if game_state.get("game_ended", False):
                return False
                
            # Set up the board from FEN
            if "fen" in game_state:
                self.chess_board.set_fen(game_state["fen"])
                self.draw_chessboard()
                self.append_message("Previous chess game loaded.", sender="system")
                return True
        except Exception as e:
            logging.error(f"Failed to load saved chess game: {e}")
            # If loading fails, remove the corrupted save file
            try:
                os.remove(CHESS_SAVE_FILE)
            except:
                pass
        return False

    def on_game_selected(self, event):
        selected_game = self.game_var.get()

        # Päivitä "Send Image" -valintaruudun teksti pelin mukaan
        if selected_game == "Chess":
            self.send_image_check.config(text="Send chessboard picture\nto AI with moves")
        else:
            self.send_image_check.config(text="Send image with message")

        if selected_game == "Chess":
            self.show_chess_ui()
        else:
            self.hide_chess_ui()

    def get_device_sample_rate(self, device_id):
        """Query the default sample rate for the given device (input)."""
        try:
            device_info = sd.query_devices(device_id, 'input')
            return int(device_info['default_samplerate'])
        except Exception as e:
            logging.error(f"Could not retrieve sample rate for device {device_id}: {e}")
            return 44100

    def get_device_input_channels(self, device_id):
        """Return max input channels for the selected device or 1 if error."""
        try:
            device_info = sd.query_devices(device_id, 'input')
            return device_info['max_input_channels']
        except Exception as e:
            logging.error(f"Could not retrieve channels for device {device_id}: {e}")
            return 1

    def reset_conversation(self):
        if messagebox.askyesno("Confirm Reset", "Are you sure you want to reset LLM conversation memory?"):
            global messages
            messages = []

    def clear_chat(self):
        """
        Tyhjentää kaikki viestit chat-ikkunasta, mutta säilyttää keskusteluhistorian.
        """
        for widget in self.chat_inner_frame.winfo_children():
            widget.destroy()
        
        # Reset any references to message frames
        self.last_message_frame = None
        
        # Use after() to ensure widgets are fully destroyed before adding new message
        self.after(100, lambda: self.append_message("The chat has been cleared. You can start a new conversation.", sender="system"))
        
        # Scroll to top after widgets are updated
        self.after(200, lambda: self.chat_canvas.yview_moveto(0))


    def update_status(self, status):
        self.after(0, self._update_status, status)

    def _update_status(self, status):
        self.status_label.config(text=status)
        if status in ["Idle", "Listening...", "Speaking completed"]:
            self.status_label.config(fg="green")
        elif status.startswith("Error"):
            self.status_label.config(fg="red")
        else:
            self.status_label.config(fg="blue")

    def append_message(self, text, sender="system"):
        self.after(0, self._append_message, text, sender)

    def _append_message(self, text, sender):
        """
        Display messages in the chat area with different styling for system/user/assistant.
        Consecutive system messages are grouped under a single collapsible indicator.
        """
        # Store all messages in chat history regardless of visibility
        timestamp = datetime.now().strftime('%H:%M')
        self.chat_history.append({"text": text, "sender": sender, "timestamp": timestamp})
        
        # Check if this is a system message that should be grouped
        if sender == "system":
            # Get the last frame that was added (if any)
            last_frame = None
            if hasattr(self, "last_message_frame") and self.last_message_frame:
                last_frame = self.last_message_frame
                
            # Check if the last frame was a system message group
            if last_frame and hasattr(last_frame, "is_system_group") and last_frame.is_system_group:
                # Add this message to the existing group
                content_frame = last_frame.content_frame
                
                # Add a separator if there are already messages
                if content_frame.winfo_children():
                    separator = tk.Frame(content_frame, height=1, bg="#DDDDDD")
                    separator.pack(fill=tk.X, pady=2)
                
                # Add the new message
                msg_bubble = tk.Label(
                    content_frame,
                    text=text,
                    bg="#FFCCCB",
                    fg="black",
                    padx=10,
                    pady=5,
                    wraplength=400,
                    justify="left",
                    relief="groove",
                    borderwidth=1
                )
                msg_bubble.pack(fill=tk.X, anchor="w")
                
                # Add timestamp
                time_label = tk.Label(
                    content_frame,
                    text=timestamp,
                    bg="#ECE5DD",
                    fg="gray",
                    font=("Arial", 8)
                )
                time_label.pack(anchor="w")
                
                # Update the message count in the indicator
                last_frame.message_count += 1
                last_frame.indicator.config(text=f"➕ {last_frame.message_count} System Messages")
                
                # Update scroll
                self.chat_canvas.update_idletasks()
                self.chat_canvas.yview_moveto(1.0)
                return
        
        # If we reached here, we're either:
        # 1. Not a system message
        # 2. First system message in a new group
        
        # Create a new message frame
        if sender == "user":
            bg_color = "#DCF8C6"
            anchor = "e"
            fg_color = "black"
        elif sender == "assistant":
            bg_color = "#FFFFFF"
            anchor = "w"
            fg_color = "black"
        else:  # system messages
            bg_color = "#FFCCCB"
            anchor = "w"
            fg_color = "black"
        
        msg_frame = tk.Frame(self.chat_inner_frame, bg="#ECE5DD", padx=10, pady=5)
        msg_frame.pack(fill=tk.BOTH, expand=True, anchor=anchor)
        self.last_message_frame = msg_frame
        
        # For system messages, create a collapsible group
        if sender == "system":
            # Mark this frame as a system group
            msg_frame.is_system_group = True
            msg_frame.message_count = 1
            
            # Create a collapsed indicator
            collapsed_frame = tk.Frame(msg_frame, bg="#ECE5DD")
            collapsed_frame.pack(fill=tk.X, anchor="w")
            
            indicator = tk.Label(
                collapsed_frame,
                text="➕ 1 System Message",
                bg="#FFCCCB",
                fg="black",
                padx=5,
                pady=2,
                relief="groove",
                borderwidth=1,
                cursor="hand2"
            )
            indicator.pack(side=tk.LEFT, anchor="w")
            msg_frame.indicator = indicator
            
            # Create content frame (initially hidden)
            content_frame = tk.Frame(msg_frame, bg="#ECE5DD")
            msg_frame.content_frame = content_frame
            
            # Add the message to the content frame
            msg_bubble = tk.Label(
                content_frame,
                text=text,
                bg=bg_color,
                fg=fg_color,
                padx=10,
                pady=5,
                wraplength=400,
                justify="left",
                relief="groove",
                borderwidth=1
            )
            msg_bubble.pack(fill=tk.X, anchor=anchor)
            
            # Add timestamp
            time_label = tk.Label(
                content_frame,
                text=timestamp,
                bg="#ECE5DD",
                fg="gray",
                font=("Arial", 8)
            )
            time_label.pack(anchor=anchor)
            
            # Toggle function for this group
            def toggle_system_group(event):
                if content_frame.winfo_ismapped():
                    content_frame.pack_forget()
                    indicator.config(text=f"➕ {msg_frame.message_count} System Message{'s' if msg_frame.message_count > 1 else ''}")
                else:
                    content_frame.pack(fill=tk.X, anchor="w", pady=(5, 0))
                    indicator.config(text=f"➖ {msg_frame.message_count} System Message{'s' if msg_frame.message_count > 1 else ''}")
            
            # Bind click event to the indicator
            indicator.bind("<Button-1>", toggle_system_group)
            
        else:  # Regular user/assistant messages
            bubble = tk.Label(
                msg_frame,
                text=text,
                bg=bg_color,
                fg=fg_color,
                padx=10,
                pady=5,
                wraplength=400,
                justify="left",
                relief="groove",
                borderwidth=1
            )
            bubble.pack(anchor=anchor)

            time_label = tk.Label(
                msg_frame,
                text=timestamp,
                bg="#ECE5DD",
                fg="gray",
                font=("Arial", 8)
            )
            time_label.pack(anchor=anchor)

        self.chat_canvas.update_idletasks()
        self.chat_canvas.yview_moveto(1.0)

    def append_image_in_chat(self, image_path, sender="user"):
        self.after(0, self._append_image_in_chat, image_path, sender)

    def _append_image_in_chat(self, image_path, sender):
        """
        Display an image in the chat area, using a label widget with a PhotoImage.
        """
        import os

        if not os.path.exists(image_path):
            self.append_message(f"Image not found: {image_path}", sender="system")
            return

        if sender == "user":
            bg_color = "#DCF8C6"
            anchor = "e"
        elif sender == "assistant":
            bg_color = "#FFFFFF"
            anchor = "w"
        else:
            bg_color = "#FFCCCB"
            anchor = "w"

        try:
            img = Image.open(image_path)
            max_width = 300
            if img.width > max_width:
                wpercent = (max_width / float(img.width))
                hsize = int(float(img.height) * float(wpercent))
                img = img.resize((max_width, hsize), Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(img)
        except Exception as e:
            self.append_message(f"Failed to load image: {e}", sender="system")
            return

        msg_frame = tk.Frame(self.chat_inner_frame, bg="#ECE5DD", padx=10, pady=5)
        msg_frame.pack(fill=tk.BOTH, expand=True, anchor=anchor)

        bubble = tk.Label(
            msg_frame,
            bg=bg_color,
            image=photo,
            padx=5,
            pady=5,
            relief="groove",
            borderwidth=1
        )
        bubble.image = photo
        bubble.pack(anchor=anchor)

        timestamp = datetime.now().strftime('%H:%M')
        time_label = tk.Label(
            msg_frame,
            text=timestamp,
            bg="#ECE5DD",
            fg="gray",
            font=("Arial", 8)
        )
        time_label.pack(anchor=anchor)

        self.chat_canvas.update_idletasks()
        self.chat_canvas.yview_moveto(1.0)


    #_____________Scrollaus hiirellä________________________

    def on_tab_changed(self, event):
        """
        Tätä kutsutaan, kun Notebookin välilehti vaihtuu.
        Päivitetään active_canvas oikeaksi.
        """
        selected_tab = self.notebook.select()  # Hanki ID:n valitulle tabille.

        if selected_tab == str(self.main_tab):
            self.active_canvas = self.chat_canvas
        elif selected_tab == str(self.settings_tab):
            self.active_canvas = self.settings_canvas
        # Voit lisätä ehtoja, jos sinulla on enemmän tabeja.

        # Varmistetaan, että vieritysalue päivittyy uuden canvaksen koon mukaan
        if self.active_canvas == self.settings_canvas:
            self.on_frame_configure(None)

    def bind_mousewheel(self):
        system = platform.system()
        if system == 'Windows':
            self.bind_all("<MouseWheel>", self.on_mousewheel_windows)
        elif system == 'Darwin':
            self.bind_all("<MouseWheel>", self.on_mousewheel_mac)
        else:
            self.bind_all("<Button-4>", self.on_mousewheel_linux)
            self.bind_all("<Button-5>", self.on_mousewheel_linux)

    def on_mousewheel_windows(self, event):
        # Käytä self.active_canvas-muuttujaa!
        self.active_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def on_mousewheel_mac(self, event):
        # Käytä self.active_canvas-muuttujaa!
        self.active_canvas.yview_scroll(int(-1 * (event.delta)), "units")

    def on_mousewheel_linux(self, event):
        # Käytä self.active_canvas-muuttujaa!
        if event.num == 4:
            self.active_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.active_canvas.yview_scroll(1, "units")


    #_______________________________________________________________________________________________________________

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            try:
                self.update_status("Shutting down...")
                
                # Save chess game state
                self.save_chess_game_state()
                
                # Stop all streaming and processing
                self.stream_stop_event.set()
                
                # Stop audio streams
                if self.stream1:
                    self.stream1.stop()
                    self.stream1.close()
                    self.stream1 = None
                
                if self.stream2:
                    self.stream2.stop()
                    self.stream2.close()
                    self.stream2 = None
                
                # Clean up all threads
                self.cleanup_threads()
                
                # Final cleanup
                self.destroy()
                sys.exit(0)
            except Exception as e:
                logging.error(f"Error during exit: {e}")
                self.destroy()
                sys.exit(1)

    def get_vol_threshold(self):
        try:
            return float(self.vol_threshold_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Volume Threshold must be a number.")
            return DEFAULT_VOL_THRESHOLD

    def get_min_record_duration(self):
        try:
            return float(self.min_record_duration_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Minimum Record Duration must be a number.")
            return DEFAULT_MIN_RECORD_DURATION

    def get_max_silence_duration(self):
        try:
            return float(self.max_silence_duration_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Maximum Silence Duration must be a number.")
            return DEFAULT_MAX_SILENCE_DURATION

    def handle_error(self, error_message):
        user_message = f"An error occurred: {error_message}. Restarting the conversation."
        self.append_message(user_message, sender="system")
        self.update_status("Error encountered")
        self.reset_conversation()


    def stream_api_response(self, api_type, request_params, on_chunk_received, on_complete=None):
        """
        Generic streaming handler that works with both LLM and TTS APIs.
        
        Args:
            api_type: String, either "llm" or "tts"
            request_params: Dictionary containing API-specific parameters
            on_chunk_received: Callback function to process each chunk
            on_complete: Optional callback when streaming completes
        """
        self.update_status(f"Starting {api_type} streaming...")
        print(f"stream_api_response: Starting {api_type} streaming...")
        self.current_tts_stream_completed = False
        try:
            if api_type == "llm":
                # Extract LLM request parameters
                openrouter_api_url = self.config_data.get('OPENROUTER_API_URL', 'https://openrouter.ai/api/v1/chat/completions')
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {request_params["api_key"]}',
                    'X-Title': 'Chat Audio Generator'
                }
                
                # Start LLM streaming
                with requests.post(openrouter_api_url, headers=headers, json=request_params["data"], stream=True) as response:
                    response.raise_for_status()
                    self.update_status("Receiving streaming response...")
                    print("stream_api_response: Receiving streaming response...")
                    buffer = ""
                    response.encoding = 'utf-8'
                    for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                        if chunk:
                            buffer += chunk
                            
                            # Process complete SSE lines
                            while True:
                                line_end = buffer.find('\n')
                                if line_end == -1:
                                    break
                                    
                                line = buffer[:line_end].strip()
                                buffer = buffer[line_end + 1:]
                                
                                if line.startswith('data: '):
                                    data = line[6:]
                                    if data == '[DONE]':
                                        break
                                        
                                    try:
                                        data_obj = json.loads(data)
                                        on_chunk_received(data_obj)
                                        print("stream_api_response: data_obj received")
                                    except json.JSONDecodeError:
                                        pass
                    
                self.update_status(f"{api_type.upper()} streaming completed")
                print(f"stream_api_response: {api_type.upper()} streaming completed")
                
            elif api_type == "tts":
                print("stream_api_response: Entering TTS streaming block")
                client = OpenAI(api_key=request_params["api_key"])
                text = request_params["text"]

                if re.match(r'^[Nn][Oo][.!,]?$', text.strip()):
                    print(f"stream_api_response: Skipping TTS for simple 'No' response: '{text}'")
                    # Call on_complete callback if provided
                    if on_complete:
                        on_complete()
                    return
                                    
                print(f"stream_api_response: Text to be spoken: {text}")
                device_id = request_params.get("device_id")
                estimated_duration = len(text.split()) / 2  # Estimate: 2 words per second
                
                # Start progress bar
                self.after(0, lambda: self.start_progress_bar(estimated_duration))
                print(f"stream_api_response: Progress bar started with estimated duration: {estimated_duration}")
                
                try:
                    with client.audio.speech.with_streaming_response.create(
                            model="tts-1",
                            voice=self.config_data.get('TTS_VOICE', 'alloy'),  # Use selected voice with fallback
                            input=text,
                            response_format='wav'
                    ) as response:
                        audio_buffer = bytearray()
                        stream = None
                        
                        # Set up audio playback queue and processing
                        audio_queue = queue.Queue()
                        playback_done = threading.Event()
                        playback_thread = threading.Thread(
                            target=self._process_tts_audio_queue, 
                            args=(audio_queue, playback_done, device_id),
                            daemon=True
                        )
                        playback_thread.start()
                        
                        # Stream chunks to the audio queue
                        first_chunk = True
                        for data_chunk in response.iter_bytes(chunk_size=4096):
                            audio_queue.put(data_chunk)
                            
                            # Update progress based on data received
                            progress_percent = min(90, len(data_chunk) / (estimated_duration * 16000))
                            self.after(0, lambda p=progress_percent: self.update_progress_bar(p))
                            
                            # Let the callback process each chunk if needed
                            on_chunk_received(data_chunk, first_chunk)
                            first_chunk = False

                            #print(f"stream_api_response: Data chunk received")
                        
                        # Signal completion and wait for playback to finish
                        playback_done.set()
                        playback_thread.join()


                except Exception as e:  # Catch *ALL* exceptions here
                    error_msg = f"Error in TTS streaming: {e}"
                    print(f"stream_api_response: error_msg")
                    logging.error(error_msg)
                    self.after(0, lambda: self.append_message(error_msg, sender='system'))
                    self.after(0, lambda: self.update_status(f"Error during TTS"))
                    
                # Complete the progress bar
                self.progress_bar['value'] = self.progress_bar['maximum']
                print("stream_api_response: Progress bar completed")
                self.update_status("TTS streaming completed")
                self.current_tts_stream_completed = True
                
            # Call on_complete callback if provided
            if on_complete:
                print("stream_api_response: Calling on_complete callback")
                on_complete()
                
        except Exception as e:
            error_msg = f"Error in {api_type} streaming: {e}"
            print(f"stream_api_response: {error_msg}")
            logging.error(error_msg)
            self.after(0, lambda: self.append_message(error_msg, sender='system'))
            self.after(0, lambda: self.update_status(f"Error during {api_type}"))
            return None

    def _process_tts_audio_queue(self, audio_queue, playback_done, device_id):
        """Process and play audio data from the queue"""


        print("_process_tts_audio_queue: Started")
        try:
            stream = None
            first_chunk = True
            wav_header = None
            
            while not playback_done.is_set() or not audio_queue.empty():
                try:
                    audio_chunk = audio_queue.get(timeout=0.5)
                    #print(f"_process_tts_audio_queue: Got chunk: {len(audio_chunk) if audio_chunk else None} bytes")
                    
                    if audio_chunk is not None:
                        if first_chunk:
                            # Extract WAV header info from first chunk
                            with io.BytesIO(audio_chunk) as wav_io:
                                wav = wave.open(wav_io, 'rb')
                                channels = wav.getnchannels()
                                sample_width = wav.getsampwidth()
                                framerate = wav.getframerate()
                                print(f"_process_tts_audio_queue: Channels: {channels}, Sample Width: {sample_width}, Framerate: {framerate}")
                                
                                # Create audio stream
                                stream = sd.OutputStream(
                                    samplerate=framerate,
                                    channels=channels,
                                    dtype=f'int{sample_width*8}',
                                    device=device_id
                                )
                                print(f"_process_tts_audio_queue: Stream created: {stream}")
                                stream.start()
                                print(f"_process_tts_audio_queue: Stream started")
                                
                                # Process first chunk
                                audio_data = np.frombuffer(
                                    wav.readframes(wav.getnframes()),
                                    dtype=np.int16
                                )
                                #print(f"_process_tts_audio_queue: Writing {len(audio_data)} samples to stream")
                                stream.write(audio_data)
                                first_chunk = False
                        else:
                            # Process subsequent chunks
                            audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                            #print(f"_process_tts_audio_queue: Writing {len(audio_data)} samples to stream")
                            if stream is not None:
                                stream.write(audio_data)
                    
                    audio_queue.task_done()
                except queue.Empty:
                    #print("_process_tts_audio_queue: Queue empty") # This can be noisy, so comment it out unless needed
                    continue
                except Exception as e:
                    print(f"_process_tts_audio_queue: Error in playback: {e}") # CRITICAL: Print the exception
                    logging.error(f"Error in audio playback: {e}")
                    break
            
            if stream is not None:
                print("_process_tts_audio_queue: Stopping stream")
                stream.stop()
                print("_process_tts_audio_queue: Closing stream")
                stream.close()
            print("_process_tts_audio_queue: Exiting")
                    
        except Exception as e:
            logging.error(f"Audio playback thread error: {e}")
            print(f"_process_tts_audio_queue: Outer exception: {e}")

        finally:
            print("_process_tts_audio_queue: Playback thread finished")
                    

    def send_message(self, messages, image_path=None, language='fi', openrouter_api_key='', model='default-model', reasoning_effort='medium', show_reasoning=True):
        """
        Unified function to send messages to either Ollama or OpenRouter, handling streaming and TTS.
        """
        if not hasattr(self, 'in_message_processing') or not self.in_message_processing:
            self.saved_audio_state = self.audio_source_toggle_var.get()
            self.in_message_processing = True
            print(f"send_message: Saved audio state: {self.saved_audio_state}")

        if model.startswith("ollama/"):
            # Use Ollama
            return self.send_message_to_ollama(messages, image_path, language, model, reasoning_effort, show_reasoning)
        else:
            # Use OpenRouter (existing logic)
            return self.send_message_to_openrouter(messages, image_path, language, openrouter_api_key, model, reasoning_effort, show_reasoning)

    def send_message_to_ollama(self, messages, image_path=None, language='fi', model='ollama/llama3', reasoning_effort='medium', show_reasoning=True):
        """
        Sends a message to an Ollama model, streams the response, and starts TTS playback.
        """
        try:
            # Debug which Ollama model is being used
            model_name = model.replace("ollama/", "", 1)
            print(f"Using Ollama model: {model_name}")
            
            # Initialize the client
            ollama_client = ollama.Client()
            
            # Create a queue for TTS sentences
            tts_queue = queue.Queue()

            # Start the TTS worker thread
            tts_worker = threading.Thread(target=self._tts_worker, args=(tts_queue,), daemon=True)


            tts_worker.start()

            # Initialize variables for response tracking
            accumulated_text = ""
            processed_up_to = 0
            thinking_buffer = ""
            last_thinking_summary_time = time.time()
            thinking_summary_interval = 3.0
            in_thinking_mode = False
            
            # Add a placeholder message in the chat UI
            self.append_message("Thinking...", sender="assistant")

            # Define chunk handler for Ollama streaming
            def process_ollama_chunk(chunk):
                nonlocal accumulated_text, processed_up_to, thinking_buffer, last_thinking_summary_time, in_thinking_mode

                # Ollama's response structure is different from OpenRouter.
                if hasattr(chunk, 'message') and hasattr(chunk.message, 'content'):
                    # New chat API format
                    partial_content = chunk.message.content or ''
                else:
                    # Old generate API format
                    partial_content = chunk.get('response', '') or ''
                    
                    # Debug the content received
                    print(f"Ollama chunk received: {partial_content}")
                    print(f"Raw chunk: {chunk}")
                
                if partial_content:
                    # Check for thinking mode markers
                    if "<think>" in partial_content and not in_thinking_mode:
                        # Start of thinking mode
                        in_thinking_mode = True
                        # Start accumulated text with thinking emoji
                        if not accumulated_text.startswith("🤔"):
                            accumulated_text = "🤔 "
                        thinking_buffer = ""  # Reset thinking buffer
                    
                    if in_thinking_mode:
                        # Handle content in thinking mode
                        if "</think>" in partial_content:
                            # End of thinking mode
                            in_thinking_mode = False
                            
                            # Split the content before and after </think>
                            parts = partial_content.split("</think>", 1)
                            
                            # Add final part of thinking to buffer
                            thinking_buffer += parts[0]
                            
                            # Create a final summary of the thinking
                            if show_reasoning:
                                summary = self._create_thinking_summary(thinking_buffer, is_final=True)
                                if summary:
                                    self._speak_thinking_summary(summary, True)
                            
                            # Reset accumulated text for the actual response
                            accumulated_text = ""
                            processed_up_to = 0
                            
                            # Process the part after </think> if it exists
                            if len(parts) > 1 and parts[1].strip():
                                accumulated_text += parts[1]
                                self._update_last_message(accumulated_text)
                        else:
                            # Still in thinking mode
                            thinking_buffer += partial_content
                            
                            if show_reasoning:
                                # Add to accumulated text with thinking prefix
                                accumulated_text += partial_content
                                self._update_last_message(accumulated_text)
                                
                                # Check if it's time to create a thinking summary
                                current_time = time.time()
                                if current_time - last_thinking_summary_time > thinking_summary_interval and len(thinking_buffer) > 50:
                                    # Create a summary of the thinking
                                    summary = self._create_thinking_summary(thinking_buffer)
                                    
                                    # Speak the summary without showing it in chat
                                    if summary:
                                        # Use a separate thread to avoid blocking
                                        summary_thread = threading.Thread(
                                            target=self._speak_thinking_summary,
                                            args=(summary,),
                                            daemon=True
                                        )
                                        summary_thread.start()
                                    
                                    # Reset for next summary but keep some recent context
                                    last_thinking_summary_time = current_time
                    else:
                        # Normal content (not in thinking mode)
                        accumulated_text += partial_content
                        
                        # Make sure to update UI immediately
                        self._update_last_message(accumulated_text)
                        
                        # Find complete sentences for TTS (same as in send_message_to_openrouter)
                        while True:
                            sentence_end = accumulated_text.find('.', processed_up_to)
                            if sentence_end == -1:
                                sentence_end = accumulated_text.find('!', processed_up_to)
                            if sentence_end == -1:
                                sentence_end = accumulated_text.find('?', processed_up_to)
                            if sentence_end == -1:
                                break  # No complete sentence yet
                                
                            # Extract and queue the sentence
                            sentence = accumulated_text[processed_up_to:sentence_end + 1].strip()
                            if sentence:
                                # First, remove complete image prompts with brackets
                                cleaned_sentence = re.sub(r'\[CREATE_IMAGE:.*?\]', '', sentence).strip()
                                
                                # Check for any "CREATE_IMAGE:" text with or without brackets
                                if 'CREATE_IMAGE:' in cleaned_sentence:
                                    # Cut off everything from CREATE_IMAGE: to the end, regardless of brackets
                                    cleaned_sentence = cleaned_sentence.split('CREATE_IMAGE:')[0].strip()
                                    
                                if cleaned_sentence:  # Only queue if there's content after cleaning
                                    sentence = cleaned_sentence
                                    tts_queue.put(sentence)
                                sentence = cleaned_sentence
                            processed_up_to = sentence_end + 1
            
            # Define completion handler (Ollama-specific)
            def on_ollama_complete():
                nonlocal accumulated_text, processed_up_to, thinking_buffer, in_thinking_mode

                # If we were in thinking mode and have buffer left, create a final summary
                if in_thinking_mode and thinking_buffer:
                    summary = self._create_thinking_summary(thinking_buffer, is_final=True)
                    if summary and show_reasoning:
                        # Queue the final thinking summary with highest priority
                        self._speak_thinking_summary(summary, True)

                        # Signal the end of thinking summaries
                        self.thinking_tts_queue.put(None)
                else:
                    # If we weren't in thinking mode, make sure any thinking TTS is stopped
                    self.stop_thinking_tts()

                # Add any remaining text to the queue
                if processed_up_to < len(accumulated_text):
                    remaining_text = accumulated_text[processed_up_to:].strip()
                    if remaining_text:
                        tts_queue.put(remaining_text)
                        
                # Signal the TTS worker to stop
                tts_queue.put(None)

                # Check for image generation requests
                if "[CREATE_IMAGE:" in accumulated_text:
                    try:
                        image_start = accumulated_text.find("[CREATE_IMAGE:")
                        image_end = accumulated_text.find("]", image_start)
                        if image_end > image_start:
                            image_prompt = accumulated_text[image_start + 14:image_end].strip()

                            # Clean the response for display
                            cleaned_response = accumulated_text[:image_start].rstrip()
                            self._update_last_message(cleaned_response)

                            # Mark image as processed and generate it
                            self.last_image_processed = image_prompt
                            self.display_generated_image(image_prompt)
                    except Exception as e:
                        logging.error(f"Error extracting image prompt: {e}")

            print(f"Sending conversation with {len(messages)} messages to Ollama") 
            response_iter = ollama_client.chat(model=model_name, messages=messages, stream=True)

            # Process each chunk as it comes in
            for chunk in response_iter:
                process_ollama_chunk(chunk)
                
            # Call completion handler when done
            on_ollama_complete()
            
            # Return a response object similar to OpenRouter's format
            # Clean up any thinking tags in the final response
            final_text = accumulated_text
            if "<think>" in final_text or "</think>" in final_text:
                final_text = re.sub(r'<think>.*?</think>', '', final_text, flags=re.DOTALL).strip()
            
            return {
                "choices": [
                    {
                        "message": {
                            "content": final_text,
                            "role": "assistant"
                        }
                    }
                ]
            }
            
        except Exception as e:
            error_message = f"An unexpected error occurred with Ollama: {e}"
            logging.error(error_message)
            self.after(0, lambda: self.append_message(error_message, sender="system"))
            return None

    def send_message_to_openrouter(self, messages, image_path=None, language='fi', openrouter_api_key='', model='default-model', reasoning_effort='medium', show_reasoning=True):
        """
        Sends a message to the OpenRouter API, streams the LLM response, and starts TTS playback as sentences are completed.

        Args:
            messages (list): List of message dictionaries with 'role' and 'content'.
            image_path (str, optional): Path to an image to include in the request.
            language (str): Language code for the response (default: 'fi').
            openrouter_api_key (str): API key for OpenRouter.
            model (str): LLM model to use (default: 'default-model').
            reasoning_effort (str): Level of reasoning effort ('low', 'medium', 'high').
            show_reasoning (bool): Whether to include reasoning in the response.

        Returns:
            dict: Response object mimicking the non-streaming API response structure.
        """


        # Prepare personality prompt
        base_personality_prompt = self.config_data.get('LLM_PERSONALITY', '')
        current_game = self.game_var.get()

        # Prepare game-specific personality prompt
        if current_game != "Chess":
            image_generation_instruction = (
                "\nIf you feel like providing image or in case user says something like ...I would like to see an image of... "
                "just include description of the image in English (ENGLISH!) on the last line of your reply like this: \n"
                "[CREATE_IMAGE: <prompt text>]\n"
                "(Make sure to include the square brackets as shown)."
                "No math in latex format. Use simple math expressions like 2+2=4."
            )
            final_personality_prompt = base_personality_prompt + image_generation_instruction
        else:
            final_personality_prompt = base_personality_prompt

        # Add system message if needed
        game_changed = hasattr(self, 'previous_game') and current_game != self.previous_game
        self.previous_game = current_game

        if final_personality_prompt and (game_changed or not messages or messages[0].get("role") != "system"):
            if messages and messages[0].get("role") == "system":
                messages.pop(0)
            messages.insert(0, {"role": "system", "content": final_personality_prompt})

        # Process image if present
        if image_path and messages and messages[-1]['role'] == 'user':
            base64_image = encode_image(image_path)  # Assume encode_image is defined elsewhere
            if base64_image:
                if isinstance(messages[-1]['content'], list):
                    messages[-1]['content'].append(
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    )
                else:
                    existing_text = messages[-1]['content']
                    messages[-1]['content'] = [
                        {"type": "text", "text": existing_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]

        # Prepare data for the API request
        data = {
            "model": model,
            "messages": messages,
            "language": language,
            "stream": True,
            "reasoning": {
                "effort": reasoning_effort,
                "exclude": not show_reasoning
            }
        }

        # Create a queue for TTS sentences
        tts_queue = queue.Queue()

        # Start the TTS worker thread
        tts_worker = threading.Thread(target=self._tts_worker, args=(tts_queue,), daemon=True)
        tts_worker.start()

        # Initialize variables for response tracking
        accumulated_text = ""
        processed_up_to = 0
        
        # New variables for tracking thinking process
        thinking_buffer = ""
        last_thinking_summary_time = time.time()
        thinking_summary_interval = 3.0  # Summarize every 3 seconds
        in_thinking_mode = False

        # Define chunk handler for LLM streaming
        def process_llm_chunk(data_obj):
            nonlocal accumulated_text, processed_up_to, thinking_buffer, last_thinking_summary_time, in_thinking_mode
            if "choices" in data_obj and data_obj["choices"] and "delta" in data_obj["choices"][0]:
                delta = data_obj["choices"][0]["delta"]

                # Check for reasoning content in the delta
                reasoning = delta.get("reasoning", "")
                partial_content = delta.get("content", "")
                
                # If we have reasoning content, prepend the thinking emoji
                if reasoning and show_reasoning:
                    # We're in thinking mode
                    in_thinking_mode = True
                    
                    # Add to thinking buffer
                    thinking_buffer += reasoning
                    
                    # Add thinking emoji prefix if this is new reasoning content
                    if not accumulated_text.startswith("🤔"):
                        accumulated_text = "🤔 " + reasoning
                    else:
                        accumulated_text += reasoning
                    
                    # Update UI with reasoning text
                    self._update_last_message(accumulated_text)
                    
                    # Check if it's time to create a thinking summary
                    current_time = time.time()
                    if current_time - last_thinking_summary_time > thinking_summary_interval and len(thinking_buffer) > 50:
                        # Create a summary of the thinking
                        summary = self._create_thinking_summary(thinking_buffer)
                        
                        # Speak the summary without showing it in chat
                        if summary:
                            # Use a separate thread to avoid blocking
                            summary_thread = threading.Thread(
                                target=self._speak_thinking_summary,
                                args=(summary,),
                                daemon=True
                            )
                            summary_thread.start()
                        
                        # Reset for next summary
                        thinking_buffer = ""  # Keep some context
                        last_thinking_summary_time = current_time
                
                # If we have regular content, use that instead
                elif partial_content:
                    # If switching from reasoning to regular content, reset accumulated_text
                    if accumulated_text.startswith("🤔"):

                        accumulated_text = ""
                        processed_up_to = 0
                        in_thinking_mode = False
                        thinking_buffer = ""  # Clear thinking buffer
                        
                        # Stop thinking TTS to prevent audio overlap
                        if self.thinking_tts_active:

                            # Signal end of thinking summaries but with a slight delay
                            self.after(800, lambda: self.thinking_tts_queue.put(None))

                            # Clear the thinking TTS queue
                            while not self.thinking_tts_queue.empty():
                                try:
                                    self.thinking_tts_queue.get_nowait()
                                    self.thinking_tts_queue.task_done()
                                except queue.Empty:
                                    break
                            
                            # Signal end of thinking summaries
                            self.thinking_tts_queue.put(None)
                            
                            # Add a small delay to ensure audio output transitions smoothly
                            time.sleep(0.3)
                    
                    accumulated_text += partial_content

                    # Find complete sentences for TTS
                    while True:
                        # Look for sentence-ending punctuation
                        sentence_end = accumulated_text.find('.', processed_up_to)
                        if sentence_end == -1:
                            sentence_end = accumulated_text.find('!', processed_up_to)
                        if sentence_end == -1:
                            sentence_end = accumulated_text.find('?', processed_up_to)
                        if sentence_end == -1:
                            break  # No complete sentence yet

                        # Extract and queue the sentence
                        sentence = accumulated_text[processed_up_to:sentence_end + 1].strip()
                        if sentence:
                            # First, remove complete image prompts with brackets
                            cleaned_sentence = re.sub(r'\[CREATE_IMAGE:.*?\]', '', sentence).strip()
                            
                            # Check for any "CREATE_IMAGE:" text with or without brackets
                            if 'CREATE_IMAGE:' in cleaned_sentence:
                                # Cut off everything from CREATE_IMAGE: to the end, regardless of brackets
                                cleaned_sentence = cleaned_sentence.split('CREATE_IMAGE:')[0].strip()
                                
                            if cleaned_sentence:  # Only queue if there's content after cleaning
                                sentence = cleaned_sentence
                                tts_queue.put(cleaned_sentence)
                            sentence = cleaned_sentence
                        processed_up_to = sentence_end + 1

                    # Update UI with regular text
                    self._update_last_message(accumulated_text)

        # Define completion handler
        def on_llm_complete():

            
            
            nonlocal accumulated_text, processed_up_to, thinking_buffer, in_thinking_mode
            
            # If we were in thinking mode and have buffer left, create a final summary
            if in_thinking_mode and len(thinking_buffer) < 200:

                summary = self._create_thinking_summary(thinking_buffer, is_final=True)
                if summary:
                    # Queue the final thinking summary with highest priority
                    self._speak_thinking_summary(summary, True)
                    
                    # Signal the end of thinking summaries
                    self.thinking_tts_queue.put(None)
            else:
                # If we weren't in thinking mode, make sure any thinking TTS is stopped
                self.stop_thinking_tts()
            
            # Add any remaining text to the queue
            if processed_up_to < len(accumulated_text):
                remaining_text = accumulated_text[processed_up_to:].strip()
                if remaining_text:
                    tts_queue.put(remaining_text)
            
            # Signal the TTS worker to stop
            tts_queue.put(None)

            # Check for image generation requests
            if "[CREATE_IMAGE:" in accumulated_text:
                try:
                    image_start = accumulated_text.find("[CREATE_IMAGE:")
                    image_end = accumulated_text.find("]", image_start)
                    if image_end > image_start:
                        image_prompt = accumulated_text[image_start + 14:image_end].strip()

                        # Clean the response for display
                        cleaned_response = accumulated_text[:image_start].rstrip()
                        self._update_last_message(cleaned_response)

                        # Mark image as processed and generate it
                        self.last_image_processed = image_prompt
                        self.display_generated_image(image_prompt)
                except Exception as e:
                    logging.error(f"Error extracting image prompt: {e}")

        # Call the streaming function with LLM parameters
        stream_params = {
            "api_key": openrouter_api_key,
            "data": data
        }

        self.stream_api_response("llm", stream_params, process_llm_chunk, on_llm_complete)

        # Construct a response object
        response_obj = {
            "choices": [
                {
                    "message": {
                        "content": accumulated_text,
                        "role": "assistant",
                        "reasoning": None  # Add reasoning if needed
                    }
                }
            ]
        }

        return response_obj
    
    def _create_thinking_summary(self, thinking_text, is_final=False):
        """
        Create a summary of the thinking process text.
        
        Args:
            thinking_text (str): The thinking process text to summarize
            is_final (bool): Whether this is the final summary
            
        Returns:
            str: A summarized version of the thinking text
        """
        
        # For longer thinking, extract a single meaningful complete sentence
        # Split into sentences ensuring we get complete ones
        sentences = re.split(r'(?<=[.!?])\s+', thinking_text)
        
        # Make sure we have valid sentences
        valid_sentences = [s for s in sentences if len(s) > 10 and s.strip().endswith(('.', '!', '?'))]
        if not valid_sentences:
            return None
        
        if is_final:
            return f"My final thought: {valid_sentences[-1]}"
        
        selected = valid_sentences[-1]
        return selected

    def _speak_thinking_summary(self, summary, is_final=False):
        """
        Queue the thinking summary to be spoken without overlapping.
        
        Args:
            summary (str): The summary to speak
            is_final (bool): Whether this is the final thinking summary
        """
        try:
            # Start the worker if not already running
            if not self.thinking_tts_active:
                self.thinking_tts_worker = threading.Thread(
                    target=self._thinking_tts_worker,
                    daemon=True
                )
                self.thinking_tts_worker.start()
            
            # Priority: 1 for final summaries, 0 for regular ones
            priority = 1 if is_final else 0
            
            # If a high priority message and speech is in progress, signal to stop
            if priority == 1 and hasattr(self, 'speech_in_progress') and self.speech_in_progress.is_set():
                self.thinking_tts_should_stop.set()
                # Allow a moment for current speech to stop
                time.sleep(0.2)
                self.thinking_tts_should_stop.clear()
            
            # For regular priority messages, check if we should skip due to ongoing speech
            elif priority == 0 and hasattr(self, 'speech_in_progress') and self.speech_in_progress.is_set():
                # Skip non-final messages if speech is already in progress
                return
                
            # Add to queue
            self.thinking_tts_queue.put((summary, is_final, priority))
            
        except Exception as e:
            logging.error(f"Error queueing thinking summary: {e}")

    def _thinking_tts_worker(self):
        """
        Worker thread that processes thinking summaries from the queue
        and ensures they don't overlap when spoken.
        """
        self.thinking_tts_active = True
        self.thinking_tts_should_stop = threading.Event()
        self.speech_in_progress = threading.Event()  # Track when speech is playing
        
        try:
            last_speech_time = 0
            min_interval_between_speech = 3.0  # Increased interval between thinking summaries
            
            while self.thinking_tts_active and not self.thinking_tts_should_stop.is_set():
                try:
                    # Get the next item (blocking with timeout)
                    item = self.thinking_tts_queue.get(timeout=0.5)
                    if item is None or self.skip_tts_var.get():
                        # None is the signal to stop
                        break
                        
                    summary, is_final, priority = item
                    
                    # Ensure we don't speak too frequently by checking time since last speech
                    time_since_last_speech = time.time() - last_speech_time
                    if time_since_last_speech < min_interval_between_speech and not is_final:
                        # Skip this thinking update if it's too soon after the previous one
                        # unless it's the final summary
                        self.thinking_tts_queue.task_done()
                        continue
                    
                    # Check if we should stop before starting new audio
                    if self.thinking_tts_should_stop.is_set():
                        self.thinking_tts_queue.task_done()
                        break
                    
                    # Process this thinking summary
                    selected_output_index = self.audio_output_combobox.current()
                    device_id = None
                    if selected_output_index >= 0 and self.available_output_devices:
                        _, device_id = self.available_output_devices[selected_output_index]
                    
                    # Convert to speech
                    selected_language = self.map_language_to_code(self.language_var.get())
                    text_with_spoken_numbers = self.convert_numbers_to_words(summary, selected_language)
                    
                    # Mark speech as in progress
                    self.speech_in_progress.set()
                    
                    # Use TTS API
                    stream_params = {
                        "api_key": self.openai_key,
                        "text": text_with_spoken_numbers,
                        "device_id": device_id,
                    }
                    
                    def check_stop_signal(chunk, is_first):
                        # Return whether to continue streaming
                        return not self.thinking_tts_should_stop.is_set()
                    
                    # This will block until the speech is complete
                    self.stream_api_response(
                        "tts", 
                        stream_params, 
                        check_stop_signal
                    )
                    
                    last_speech_time = time.time()
                    
                    # Add a small pause after speech to ensure clean transition
                    time.sleep(0.3)
                    
                    # Mark speech as complete
                    self.speech_in_progress.clear()
                    
                    # Add a longer pause after final speech
                    if is_final:
                        time.sleep(0.5)
                    
                    self.thinking_tts_queue.task_done()
                    
                except queue.Empty:
                    # Queue timeout, check if we should continue
                    continue
        except Exception as e:
            logging.error(f"Error in thinking TTS worker: {e}")
        finally:
            self.thinking_tts_active = False
            self.thinking_tts_should_stop.clear()
            self.speech_in_progress.clear()

    def stop_thinking_tts(self):
        """Stop any ongoing thinking TTS playback"""
        if hasattr(self, 'thinking_tts_should_stop'):
            self.thinking_tts_should_stop.set()
            
        # Clear the queue
        if hasattr(self, 'thinking_tts_queue'):
            while not self.thinking_tts_queue.empty():
                try:
                    self.thinking_tts_queue.get_nowait()
                    self.thinking_tts_queue.task_done()
                except queue.Empty:
                    break
                    
        # Signal end
        self.thinking_tts_queue.put(None)
    
    def _tts_worker(self, tts_queue):
        """Modified worker that handles TTS with automatic muting and unmuting."""
        
        while True:
            text = tts_queue.get()
            if text is None or self.skip_tts_var.get():
                break  # Exit when None is received

            # Combine multiple sentences if available
            while not tts_queue.empty():
                next_text = tts_queue.get()
                if next_text is None:
                    tts_queue.put(None)  # Put back the None to signal end
                    break
                text += " " + next_text
                # Limit text length to avoid overwhelming TTS
                if len(text) > 3000:
                    break

            try:
                self.tts_queue_completed = False  # Reset flag
                # Store current audio source state and mute
                print("muted by tts")
                self.audio_source_toggle_var.set("mute")
                self.update_status("AI speaking...")

                selected_language = self.map_language_to_code(self.language_var.get())

                if selected_language != "en":
                    # Convert numbers to spoken words for OTHER LANGUAGES THAN ENGLISH FOR TTS, because it will not handle numbers correctly in other languages. Additionally "=, and other markers are poorly handled in other than English, but this does not fix that".
                    text_with_spoken_numbers = self.convert_numbers_to_words(text, selected_language)
                else:
                    text_with_spoken_numbers = text
                

                # Get output device
                selected_output_index = self.audio_output_combobox.current()
                device_id = None
                if selected_output_index >= 0 and self.available_output_devices:
                    _, device_id = self.available_output_devices[selected_output_index]

                # Use stream API for TTS with blocking until complete
                self.stream_api_response(
                    api_type = "tts",
                    request_params = {
                        "api_key": self.openai_key,
                        "text": text_with_spoken_numbers,
                        "device_id": device_id,
                    },
                    on_chunk_received = lambda chunk, first: None,
                    on_complete = None
                )

                print("full TTS playback complete")
                self.tts_queue_completed = True  # Set flag in finally block

                def unmute_after_tts():
                    if self.current_tts_stream_completed:
                        # Reset listening queues before unmuting
                        while not self.audio_queue1.empty():
                            self.audio_queue1.get()
                        while not self.audio_queue2.empty():
                            self.audio_queue2.get()
                            
                        # Restore original saved audio state instead of any intermediate state
                        original_state = self.saved_audio_state  # Use the originally saved state
                        print(f"unmuted by tts - restoring original state: {original_state}")
                        self.audio_source_toggle_var.set(original_state)
                        self.update_status("Listening...")                        
                    else:
                        self.after(1000, unmute_after_tts)
                self.after(1000, unmute_after_tts)
                
            except Exception as e:
                logging.error(f"Error during TTS processing: {e}")
                self.after(0, lambda: self.append_message(f"Error during TTS: {e}", sender='system'))

                # Ensure we restore audio state even on error
                original_state = self.saved_audio_state
                print(f"unmuted by tts - restoring original state: {original_state}")
                self.audio_source_toggle_var.set(original_state)
                self.update_status("Listening...")

            tts_queue.task_done()


    def convert_numbers_to_words(self, text, language='fi'):
        """
        Convert numbers in text to their word representations for better TTS.
        
        Args:
            text (str): Text containing numbers
            language (str): Language code for conversion (fi, en, sv, es)
        
        Returns:
            str: Text with numbers converted to words
        """
        if not text:
            return text
        
        # Map language codes from the app to num2words format
        lang_map = {
            'fi': 'fi',
            'en': 'en',
            'sv': 'sv',
            'es': 'es'
        }
        
        # Get appropriate language code
        lang = lang_map.get(language, 'en')
        
        # Function to convert a matched number
        def replace_number(match):
            try:
                num = match.group(0).replace(',', '.')  # Handle decimal comma format
                # Convert only numbers that look reasonable to convert (not very long numbers)
                if len(num) > 15:  # Skip very long numbers
                    return match.group(0)
                    
                # Handle decimal numbers
                if '.' in num:
                    integer_part, decimal_part = num.split('.')
                    if integer_part:
                        integer_words = num2words(int(integer_part), lang=lang)
                    else:
                        integer_words = "zero"
                    
                    # Handle decimal part differently based on language
                    if decimal_part:
                        if lang == 'fi':
                            decimal_words = " pilkku " + " ".join([num2words(int(d), lang=lang) for d in decimal_part])
                        else:
                            decimal_words = " point " + " ".join([num2words(int(d), lang=lang) for d in decimal_part])
                        return f"{integer_words}{decimal_words}"
                    else:
                        return integer_words
                else:
                    # Handle integers
                    return num2words(int(num), lang=lang)
            except Exception as e:
                logging.error(f"Error converting number '{match.group(0)}' to words: {e}")
                return match.group(0)
        
        # Find numbers (allowing for decimal notation) and replace them
        processed_text = re.sub(r'\b\d+[,\.]?\d*\b', replace_number, text)
        return processed_text    

    def reset_conversation(self):
        global messages
        messages = []
        self.append_message("Conversation has been reset. You can start a new conversation.", sender="system")


    def check_game_over(self):
        if self.chess_board.is_game_over():
            if self.chess_board.is_checkmate():
                if self.chess_board.turn == chess.BLACK:  # White (user) won
                    winner = "White (You)"
                    system_prompt = (
                        "The user has checkmated you in chess. Respond graciously about losing. "
                        "Congratulate them on their victory and ask if they want to play again."
                    )
                    # Add a message to the AI's context that the user won
                    context_messages = messages.copy()
                    context_messages.append({"role": "system", "content": system_prompt})
                    context_messages.append({"role": "user", "content": "I've checkmated you! Game over."})
                else:  # Black (AI) won
                    winner = "Black (AI)"
                    system_prompt = (
                        "You have checkmated the user in chess. Announce your victory politely. "
                        "Be gracious in winning and offer to play again if they would like."
                    )
                    # Add a message to the AI's context that it won
                    context_messages = messages.copy()
                    context_messages.append({"role": "system", "content": system_prompt})
                    
                # System message for UI feedback
                self.append_message(f"Checkmate! {winner} wins. For a new game, reset the game", sender="system")
                
            elif self.chess_board.is_stalemate():
                system_prompt = (
                    "The chess game ended in stalemate. Express that it was a close game "
                    "and neither player could secure victory. Suggest playing again."
                )
                context_messages = messages.copy()
                context_messages.append({"role": "system", "content": system_prompt})
                
                # System message for UI feedback
                self.append_message("Stalemate! It's a draw. To play again, restart the game.", sender="system")
            
            elif self.chess_board.is_insufficient_material():
                system_prompt = (
                    "The chess game ended due to insufficient material for checkmate. "
                    "Comment on the game ending in a draw and suggest playing again."
                )
                context_messages = messages.copy()
                context_messages.append({"role": "system", "content": system_prompt})
                
                # System message for UI feedback
                self.append_message("Draw due to insufficient material. To play again, restart the game.", sender="system")
                
            else:
                # Generic game over for other cases
                system_prompt = (
                    "The chess game has ended. Comment on how the game went and suggest playing again."
                )
                context_messages = messages.copy()
                context_messages.append({"role": "system", "content": system_prompt})
                
                # System message for UI feedback
                self.append_message("Game over! To play again, restart the game.", sender="system")

            # Mark the game as ended
            self.chess_game_ended = True
            # Save the final game state with the ended flag
            self.save_chess_game_state()

            # Add a placeholder message for the assistant response
            self.append_message("Thinking about the game outcome...", sender="assistant")

            # Start a background thread to get the LLM response
            self.start_thread(
                target=self._background_game_over_response,
                args=(context_messages,),
                thread_name="GameOverResponseThread"
            )

    def _background_game_over_response(self, context_messages):
        """Process the game over LLM response in a background thread using streaming"""
        selected_language = self.language_var.get()
        
        # Use the streaming function to get the response
        response_json = self.send_message(
            messages=context_messages,
            image_path=None,
            language=self.map_language_to_code(selected_language),
            openrouter_api_key=self.openrouter_key,
            model=self.model_var.get(),
            reasoning_effort=self.reasoning_effort_var.get(),
            show_reasoning=self.show_reasoning_var.get()
        )

        if not response_json:
            return
        
        try:
            assistant_response = response_json['choices'][0]['message']['content']
            
            # Look for image generation
            pattern = r"\[CREATE_IMAGE:\s*(.*?)\]"
            match = re.search(pattern, assistant_response)
            
            if match:
                prompt_for_image = match.group(1).strip()
                cleaned_response = re.sub(pattern, '', assistant_response).strip()
                # Use after() to update UI from background thread
                self.after(0, lambda p=prompt_for_image: self.display_generated_image(p))
            else:
                cleaned_response = assistant_response
            
            # Add the original response to messages for context
            global messages
            messages.append({"role": "assistant", "content": assistant_response})
            
        except (IndexError, KeyError) as e:
            logging.error(f"Unexpected response format: {e}")
            self.after(0, lambda: self.handle_error(f"Unexpected response format from LLM: {e}"))

    

    def _update_last_message(self, text):
        """Update the last message in the chat UI with new content"""
        if hasattr(self, "last_message_frame") and self.last_message_frame:
            # Find the bubble (Label) widget in the last message frame
            for child in self.last_message_frame.winfo_children():
                if isinstance(child, tk.Label) and child.cget("bg") == "#FFFFFF":  # Assistant message background
                    # Ensure proper encoding for Finnish characters
                    try:
                        # Special formatting for reasoning/thinking text
                        if text.startswith("🤔") or text.startswith("🧠"):
                            child.config(
                                text=text,
                                fg="#555555",  # Darker gray text for thinking
                                font=("Arial", 10, "italic")  # Italic font for thinking
                            )
                        else:
                            # Normal text for regular responses
                            child.config(
                                text=text,
                                fg="black",
                                font=("Arial", 10)
                            )
                    except Exception as e:
                        logging.error(f"Error updating text with special characters: {e}")
                        # Fallback method if there are encoding issues
                        try:
                            sanitized_text = text.encode('utf-8', errors='replace').decode('utf-8')
                            child.config(text=sanitized_text)
                        except:
                            # Last resort fallback
                            child.config(text="[Text encoding error]")
                    
                    # Update the canvas to show changes
                    self.chat_canvas.update_idletasks()
                    self.chat_canvas.yview_moveto(1.0)
                    break

    def add_reasoning_controls(self, settings_frame):
        """Add reasoning controls to the settings frame"""
        reasoning_frame = ttk.LabelFrame(settings_frame, text="AI Reasoning")
        reasoning_frame.pack(fill="x", padx=10, pady=5)
        
        # Reasoning effort selection
        ttk.Label(reasoning_frame, text="Reasoning Effort:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.reasoning_effort_var = tk.StringVar(value="medium")
        reasoning_combo = ttk.Combobox(reasoning_frame, textvariable=self.reasoning_effort_var, 
                                    values=["low", "medium", "high"])
        reasoning_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        # Show reasoning checkbox
        self.show_reasoning_var = tk.BooleanVar(value=True)
        show_reasoning_check = ttk.Checkbutton(reasoning_frame, text="Show AI reasoning process", 
                                            variable=self.show_reasoning_var)
        show_reasoning_check.grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        # Help text
        help_text = ("Low effort: Basic reasoning\n"
                    "Medium effort: Balanced reasoning\n"
                    "High effort: Detailed, multi-step reasoning")
        ttk.Label(reasoning_frame, text=help_text, font=("TkDefaultFont", 8), 
                foreground="gray").grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=5)

    def save_and_process_audio(self, recorded_frames, selected_language, client, openrouter_api_key, selected_model):
        """
        Save recorded frames, transcribe, and process through LLM based on context.
        """
        self.ai_tried_chess_move = False
        self.chess_game_ended = False
        image_path = None
        
        # Save audio data to file
        try:
            recorded_audio = np.concatenate(recorded_frames, axis=0)
            wavio.write(RECORDED_AUDIO_PATH, recorded_audio, rate=self.sample_rate1, sampwidth=2)
            self.update_status("Voice input saved")
            self.append_message(f"Tallennettu ääni: {RECORDED_AUDIO_PATH}", "system")
            logging.info(f"Recording saved to {RECORDED_AUDIO_PATH}")
        except Exception as e:
            logging.error(f"Error saving recorded audio: {e}")
            self.handle_error(f"Error saving recorded audio: {e}")
            return

        # Transcribe audio using Whisper
        try:
            user_input = self._transcribe_audio(client, selected_language)
            if not user_input:
                return
        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            self.handle_error(f"Transcription error: {e}")
            return
        
        # Check for attached PDF content
        user_input = self._add_pdf_content_if_exists(user_input)
        
        # Check if user wants music generation
        if self.wants_pop_song(user_input):
            self._handle_music_generation(user_input)
            return
            
        # Process based on current game mode
        current_game = self.game_var.get()
        
        # Handle image if not in chess mode
        if current_game != "Chess" and self.send_image_var.get():
            image_path = select_latest_image()
            if image_path:
                self.append_image_in_chat(image_path, sender="user")
                
        # Process based on game mode
        if current_game == "Chess":
            self._handle_chess_input(user_input, selected_language, openrouter_api_key, selected_model, image_path)
        else:
            self._handle_normal_chat(user_input, selected_language, openrouter_api_key, selected_model, image_path)

    def _transcribe_audio(self, client, selected_language):
        """Transcribe audio file and validate result"""
        try:
            with open(RECORDED_AUDIO_PATH, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=selected_language
                )
                
                lang_display = {
                    'fi': "Finnish",
                    'sv': "Swedish",
                    'es': "Spanish"
                }.get(selected_language, "English")
                self.update_status("Recorded audio converted to text")
                logging.info(f"User said ({lang_display}): {transcription.text}")
                
                # Validate transcription
                if not is_valid_transcription(transcription.text, blacklist=self.excluded_phrases):
                    self.update_status("Voice input failed")
                    logging.info(f"Filtered out transcription: {transcription.text}")
                    self.append_message(
                        "Unrecognized or irrelevant audio was detected and ignored.",
                        sender="system"
                    )
                    return None
                    
                self.append_message(transcription.text, sender="user")
                return transcription.text
                
        except FileNotFoundError:
            logging.error(f"File not found: {RECORDED_AUDIO_PATH}.")
            self.handle_error(f"File not found: {RECORDED_AUDIO_PATH}")
            return None

    def _add_pdf_content_if_exists(self, user_input):
        """Check for and add PDF content if available"""
        text_file_path = Path(r"C:\\Users\\ronit\\OneDrive\\Tiedostot\\extracted_pdf_data\\text\\full_text.txt")
        if text_file_path.exists():
            try:
                extracted_content = text_file_path.read_text(encoding='utf-8')
                user_input += f"\n\n[Extracted PDF Content]\n{extracted_content}"
                text_file_path.unlink()  # Delete the file
                self.append_message("Added extracted PDF content to the message.", sender="system")
            except Exception as e:
                self.append_message(f"Error processing text file: {e}", sender="system")
        return user_input

    def _handle_music_generation(self, user_input):
        """Process music generation request"""
        self.append_message("It seems you want a song. Let me create one for you...", sender="assistant")

        result = self.generate_and_play_music(user_input, is_task=False)

        if not result:
            self.append_message("I encountered an issue creating your music.", sender="assistant")

    def _handle_chess_input(self, user_input, selected_language, openrouter_api_key, selected_model, image_path):
        """Process input in chess game mode"""
        global messages
        
        # Try to identify a chess move
        move_data = self._extract_chess_move(user_input)
        
        if move_data:
            from_sq_index, to_sq_index, promotion = move_data
            # Try to make the move
            move_success = self.try_user_move(from_sq_index, to_sq_index, promotion)
            if not move_success:
                return
        else:
            # Handle as conversation
            self.append_message("No valid chess move detected, treating as conversation.", sender="system")
            
            # Create context-aware prompt
            chess_context_prompt = (
                "The user is currently playing chess with you, but their input wasn't recognized as a chess move. "
                "They might be making small talk or asking a chess-related question. "
                f"Here's what they said: {user_input}\n\n"
                "If they're asking about the game or pieces, respond appropriately. "
                "If they're making small talk, chat naturally while keeping the chess context. "
                "Don't try to make a move yourself yet - wait for them to make a clear move instruction."
            )
            
            user_input += "\nImportant before we start chatting (dont mention that i said this to me in your response, I just need to let you know what is going on); if my last message looks like a move but isn't, please ignore it and continue chatting, maybe stating that there might be an error with how the system is processing user speech input and thus the move did not get recognized by the thess system. If it looks like small talk then i am trying to do small talk with you."
            # Create messages with context
            chess_context_messages = messages.copy()
            chess_context_messages.append({"role": "system", "content": chess_context_prompt})
            chess_context_messages.append({"role": "user", "content": user_input})
            

            # Add a placeholder message for the assistant response
            self.append_message("Thinking about your message...", sender="assistant")
            
            # Send to LLM and process response
            self._process_llm_response(
                chess_context_messages, 
                image_path, 
                selected_language, 
                openrouter_api_key, 
                selected_model
            )

    def _handle_normal_chat(self, user_input, selected_language, openrouter_api_key, selected_model, image_path):
        """Process input in normal chat mode"""
        global messages
        
        # Add user message to conversation history
        messages.append({"role": "user", "content": user_input})

        self.update_status("Having a conversation...")

        # Add placeholder message before streaming begins
        self.append_message("Thinking about your message...", sender="assistant")
        
        # Send to LLM and process response
        self._process_llm_response(
            messages, 
            image_path, 
            selected_language, 
            openrouter_api_key, 
            selected_model
        )

    def _process_llm_response(self, message_list, image_path, selected_language, openrouter_api_key, selected_model):
        """Send messages to LLM and process the response"""
        global messages

        # Get reasoning settings from UI
        reasoning_effort = self.reasoning_effort_var.get()
        show_reasoning = self.show_reasoning_var.get()
        
        # Pass to OpenRouter function
        response_json = self.send_message(
            message_list,
            image_path=image_path,
            language=selected_language,
            openrouter_api_key=openrouter_api_key,
            model=selected_model,
            reasoning_effort=reasoning_effort,
            show_reasoning=show_reasoning
        )
        
        if not response_json:
            return
            
        try:
            assistant_response = response_json['choices'][0]['message']['content']
            logging.info(f"Assistant: {assistant_response}")
            
            # Check if image was already processed during streaming
            already_processed = hasattr(self, 'last_image_processed')
            
            # Check for image generation instructions only if not already processed
            if not already_processed:
                prompt_for_image = None
                pattern = r"\[CREATE_IMAGE:\s*(.*?)\]"
                match = re.search(pattern, assistant_response)
                
                if match:
                    prompt_for_image = match.group(1).strip()
                    cleaned_response = re.sub(pattern, '', assistant_response).strip()
                    self.display_generated_image(prompt_for_image)
                    
                    # If response only contained image prompt
                    if not cleaned_response:
                        return
                else:
                    cleaned_response = assistant_response
            else:
                # Image already processed, just clean the response
                pattern = r"\[CREATE_IMAGE:\s*(.*?)\]"
                cleaned_response = re.sub(pattern, '', assistant_response).strip()
                # Reset the flag
                delattr(self, 'last_image_processed')
            
            # update conversation history
            messages.append({"role": "assistant", "content": assistant_response})
         
            
        except (IndexError, KeyError) as e:
            logging.error(f"Unexpected response format: {e}")
            self.handle_error(f"Unexpected response format from LLM: {e}")

    def _extract_chess_move(self, text):
        """Extract chess move from text, returns (from_sq_index, to_sq_index, promotion) or None"""
        # No need to clean filler words first - we'll handle them directly in our patterns
        text = text.lower().strip()  # Convert to lowercase for easier matching
        
        # Pattern 1: a2 to a4 format with more flexible separators and Finnish variants
        move_match = re.search(
            r".*?([a-h][1-8])[\s,\-]*(?:to|ruutuun|ruutu|ja|siirtyy|tuonne|\-\>|→)?[\s,\-]*([a-h][1-8])(?:\s*(q|r|b|n|d|t|l|r))?",
            text
        )
        
        # Pattern 2: a2a4 format (direct notation)
        if not move_match:
            move_match = re.search(r".*?\b([a-h][1-8][a-h][1-8][qrbndtlr]?)\b[.,!]?", text)
        
        if move_match:
            # Process match groups
            num_groups = len(move_match.groups())
            
            if num_groups >= 2:
                from_sq = move_match.group(1)
                to_sq = move_match.group(2)
                promotion = move_match.group(3) if num_groups >= 3 else None
                
                # Map Finnish promotion pieces to standard notation
                if promotion:
                    promotion_map = {
                        'd': 'q',  # Daami (Queen)
                        't': 'r',  # Torni (Rook)
                        'l': 'b',  # Lähetti (Bishop)
                        'r': 'n',  # Ratsu (Knight)
                    }
                    promotion = promotion_map.get(promotion, promotion)
                    
            elif num_groups == 1:
                match = move_match.group(1)
                from_sq = match[0:2]
                to_sq = match[2:4]
                promotion = match[4] if len(match) > 4 else None
                
                # Map Finnish promotion pieces in direct notation
                if promotion:
                    promotion_map = {
                        'd': 'q', 't': 'r', 'l': 'b', 'r': 'n'
                    }
                    promotion = promotion_map.get(promotion, promotion)
            else:
                return None
                
            # Convert to indexes
            try:
                from_sq_index = chess.parse_square(from_sq)
                to_sq_index = chess.parse_square(to_sq)
                return from_sq_index, to_sq_index, promotion
            except ValueError:
                logging.warning(f"Invalid squares: {from_sq} -> {to_sq}")
                self.append_message("No valid squares to move to in the move.", sender="system")
                return None
        
        # Pattern 3: [piece] takes on [square] - English and Finnish
        # Examples: "king takes on f4", "kuningas lyö f4", "ratsu ottaa e5"
        piece_takes_match = re.search(
            r".*?(king|queen|rook|bishop|knight|pawn|k|q|r|b|n|p|kuningas|daami|torni|lähetti|ratsu|sotilas)\s+"
            r"(?:takes|captures|takes\s+on|captures\s+on|lyö|ottaa|syö|vie)\s+"
            r"([a-h][1-8])[.,!]?",
            text
        )
        
        if piece_takes_match:
            piece_type = piece_takes_match.group(1)
            dest_square = piece_takes_match.group(2)
            
            # Map text piece names to chess.py piece types (English and Finnish)
            piece_map = {
                # English piece names
                'king': chess.KING, 'queen': chess.QUEEN, 'rook': chess.ROOK,
                'bishop': chess.BISHOP, 'knight': chess.KNIGHT, 'pawn': chess.PAWN,
                'k': chess.KING, 'q': chess.QUEEN, 'r': chess.ROOK,
                'b': chess.BISHOP, 'n': chess.KNIGHT, 'p': chess.PAWN,
                
                # Finnish piece names (nominative)
                'kuningas': chess.KING, 'daami': chess.QUEEN, 'torni': chess.ROOK,
                'lähetti': chess.BISHOP, 'ratsu': chess.KNIGHT, 'hevonen': chess.KNIGHT,
                'heppa': chess.KNIGHT, 'sotilas': chess.PAWN,
                
                # Finnish piece names (genitive/accusative)
                'kuninkaan': chess.KING, 'daamin': chess.QUEEN, 'tornin': chess.ROOK,
                'lähetin': chess.BISHOP, 'ratsun': chess.KNIGHT, 'hevosen': chess.KNIGHT,
                'hepan': chess.KNIGHT, 'sotilaan': chess.PAWN
            }



            chess_piece = piece_map.get(piece_type.lower())
            dest_sq_index = chess.parse_square(dest_square)

            # Check for promotion in captures (new code)
            promotion = None
            promotion_match = re.search(
                r"(?:promot(?:e|es|ing|ion)|korottaa|korotus)\s+(?:to\s+)?"
                r"(queen|rook|bishop|knight|q|r|b|n|daami|torni|lähetti|ratsu)[.,!]?",
                text, re.IGNORECASE
            )

            if promotion_match:
                promo_piece = promotion_match.group(1).lower()
                # Map promotion pieces to chess.py values
                promo_map = {
                    'queen': chess.QUEEN, 'q': chess.QUEEN, 'daami': chess.QUEEN,
                    'rook': chess.ROOK, 'r': chess.ROOK, 'torni': chess.ROOK,
                    'bishop': chess.BISHOP, 'b': chess.BISHOP, 'lähetti': chess.BISHOP,
                    'knight': chess.KNIGHT, 'n': chess.KNIGHT, 'ratsu': chess.KNIGHT
                }
                promotion = promo_map.get(promo_piece)
            
            # Find the source square by checking all pieces of that type that can move to the destination
            from_sq_index = self._find_source_square(chess_piece, dest_sq_index)
            if from_sq_index is not None:
                return from_sq_index, dest_sq_index, promotion
        
        # Pattern 4: [square] [piece] takes - English and Finnish
        # Examples: "d4 pawn takes", "e5 lähetti lyö"
        square_piece_takes_match = re.search(
            r".*?([a-h][1-8])\s+(?:piece|pawn|knight|bishop|rook|queen|king|nappula|sotilas|ratsu|lähetti|torni|daami|kuningas)?"
            r"\s+(?:takes|captures|lyö|ottaa|syö|vie)[.,!]?",
            text
        )
        
        if square_piece_takes_match:
            start_square = square_piece_takes_match.group(1)
            from_sq_index = chess.parse_square(start_square)
            
            # Find valid capture destinations from this square
            dest_sq_index = self._find_capture_destination(from_sq_index)
            if dest_sq_index is not None:
                return from_sq_index, dest_sq_index, None
        
        # Pattern 5: castle kingside/queenside - English and Finnish
        # Examples: "castle kingside", "tornitus kuningassivulle", "o-o", "0-0"
        castle_match = re.search(
            r".*?\b(?:castle|castling|tornitus|linnoitus)\s+"
            r"(?:king\s*side|queen\s*side|short|long|kuningas\s*sivulle|daami\s*sivulle|lyhyt|pitkä)\b|"
            r"o-o(?:-o)?|0-0(?:-0)?",
            text
        )
        
        if castle_match:
            castle_type = castle_match.group(0).lower()
            # Long castling (queenside) detection
            is_kingside = not ('queenside' in castle_type or 'queen side' in castle_type or 
                            'daamisivulle' in castle_type or 'daami sivulle' in castle_type or
                            'pitkä' in castle_type or 'long' in castle_type or 
                            'o-o-o' in castle_type or '0-0-0' in castle_type)
            
            # Get king's position
            king_square = self.chess_board.king(self.chess_board.turn)
            if king_square is not None:
                # Calculate target square based on castling type
                target_square = king_square + 2 if is_kingside else king_square - 2
                return king_square, target_square, None
        
        # NEW PATTERN 6: [piece1] takes [piece2] format - English and Finnish
        # Examples: "King takes bishop", "bishop takes bishop", "kuningas lyö lähetin"
        piece_takes_piece_match = re.search(
            r".*?(king|queen|rook|bishop|knight|pawn|k|q|r|b|n|p|kuningas|daami|torni|lähetti|ratsu|hevonen|heppa|sotilas)\s+"
            r"(?:takes|captures|lyö|ottaa|syö|vie)\s+"
            r"(?:the|a)?\s*"
            r"(king|queen|rook|bishop|knight|pawn|k|q|r|b|n|p|kuningas|kuninkaan|daami|daamin|torni|tornin|lähetti|lähetin|ratsu|ratsun|hevosen|hepan|sotilas|sotilaan)[.,!]?",
            text
        )
        
        if piece_takes_piece_match:
            attacker_type = piece_takes_piece_match.group(1).lower()
            target_type = piece_takes_piece_match.group(2).lower()
            
            # Map text piece names to chess.py piece types
            piece_map = {
                # English piece names
                'king': chess.KING, 'queen': chess.QUEEN, 'rook': chess.ROOK,
                'bishop': chess.BISHOP, 'knight': chess.KNIGHT, 'pawn': chess.PAWN,
                'k': chess.KING, 'q': chess.QUEEN, 'r': chess.ROOK,
                'b': chess.BISHOP, 'n': chess.KNIGHT, 'p': chess.PAWN,
                # Finnish piece names
                'kuningas': chess.KING, 'daami': chess.QUEEN, 'torni': chess.ROOK,
                'lähetti': chess.BISHOP, 'ratsu': chess.KNIGHT, 'sotilas': chess.PAWN
            }
            
            attacker_piece_type = piece_map.get(attacker_type)
            target_piece_type = piece_map.get(target_type)
            
            if not attacker_piece_type or not target_piece_type:
                return None
            
            # Get the current player's color and the opponent's color
            current_turn = self.chess_board.turn
            opposite_color = not current_turn
            
            # Find all valid captures of the specified piece type
            valid_moves = []
            
            for square in chess.SQUARES:
                piece = self.chess_board.piece_at(square)
                # Check if this square has a piece of the target type and opposite color
                if piece and piece.piece_type == target_piece_type and piece.color == opposite_color:
                    # Find all pieces of the attacker type that can capture this piece
                    for move in self.chess_board.legal_moves:
                        if move.to_square == square:  # A move to this target
                            attacker = self.chess_board.piece_at(move.from_square)
                            if attacker and attacker.piece_type == attacker_piece_type and attacker.color == current_turn:
                                valid_moves.append((move.from_square, move.to_square, None))
            
            # If there's exactly one valid move, return it
            if len(valid_moves) == 1:
                return valid_moves[0]
            elif len(valid_moves) > 1:
                # Multiple possible moves match the description
                from_squares = [chess.square_name(m[0]) for m in valid_moves]
                to_squares = [chess.square_name(m[1]) for m in valid_moves]
                self.append_message(
                    f"Multiple {chess.piece_name(attacker_piece_type)}s can capture a {chess.piece_name(target_piece_type)}. "
                    f"Please specify which one: {', '.join([f'{f}->{t}' for f, t in zip(from_squares, to_squares)])}",
                    sender="system"
                )
            else:
                # No moves match the description
                self.append_message(
                    f"No {chess.piece_name(attacker_piece_type)} can capture a {chess.piece_name(target_piece_type)}.",
                    sender="system"
                )
            
            return None
        
        # No valid move pattern found
        return None



    def _find_source_square(self, piece_type, dest_sq_index, promotion=None):
        """
        Find the source square for a piece of given type that can move to the destination.
        Now also considers potential promotions.
        Returns square index or None if no valid piece found.
        """
        current_turn = self.chess_board.turn
        possible_sources = []
        
        for move in self.chess_board.legal_moves:
            if move.to_square == dest_sq_index:
                piece = self.chess_board.piece_at(move.from_square)
                
                # Check if this is the right piece type and color
                if piece and piece.piece_type == piece_type and piece.color == current_turn:
                    # If promotion is specified, only match moves with that promotion
                    if promotion is not None:
                        if move.promotion == promotion:
                            possible_sources.append(move.from_square)
                    else:
                        possible_sources.append(move.from_square)
        
        # If there's exactly one source, return it
        if len(possible_sources) == 1:
            return possible_sources[0]
        elif len(possible_sources) > 1:
            # Multiple pieces of same type can move to destination
            self.append_message(f"Multiple {chess.piece_name(piece_type)}s can move to {chess.square_name(dest_sq_index)}. Taking the first valid move.", sender="system")
            return possible_sources[0]
        else:
            self.append_message(f"No {chess.piece_name(piece_type)} can move to {chess.square_name(dest_sq_index)}.", sender="system")
        
        return None
    

    def _find_capture_destination(self, from_sq_index):
        """
        Find the valid capture destination from a given source square.
        If there's only one possible capture, return its destination.
        Returns square index or None if no valid or ambiguous captures exist.
        """
        capture_destinations = []
        
        for move in self.chess_board.legal_moves:
            if move.from_square == from_sq_index and self.chess_board.is_capture(move):
                capture_destinations.append(move.to_square)
        
        # If there's exactly one capture available, return it
        if len(capture_destinations) == 1:
            return capture_destinations[0]
        elif len(capture_destinations) > 1:
            # Multiple capture destinations available
            dest_names = [chess.square_name(sq) for sq in capture_destinations]
            self.append_message(f"Multiple capture targets available from {chess.square_name(from_sq_index)}: {', '.join(dest_names)}. Please specify the destination.", sender="system")
        else:
            # No captures available
            self.append_message(f"No valid captures available from {chess.square_name(from_sq_index)}.", sender="system")
        
        return None



    import re
    import chess
    import logging

    def start_progress_bar(self, estimated_duration):
        self.progress_bar['value'] = 0
        self.progress_bar['maximum'] = estimated_duration * 1000  # ms
        self._progress_start_time = time.time()
        self.update_progress_bar()

    # You need to modify update_progress_bar to accept a progress parameter
    def update_progress_bar(self, progress=None):
        elapsed = (time.time() - self._progress_start_time) * 1000
        self.progress_bar['value'] = elapsed
        if elapsed < self.progress_bar['maximum']:
            self.after(100, self.update_progress_bar)
        else:
            self.progress_bar['value'] = self.progress_bar['maximum']

    def extract_move(self, response_text):
        """
        Etsi ensimmäinen kelvollinen shakkisiirto tekstistä.
        """
        try:
            move_patterns = [
                # Kuningaskatkaukset: O-O, o-o, 0-0, 0-0-0, etc.
                r'\b(?:[O0]-[O0]-[O0]|[O0]-[O0])\b',
                # Nappulasiirto, esim. qd8d1, qd8xd1, ra8a1, etc.
                r'\b([kqrbn])([a-h][1-8])\s?[-x]?\s?([a-h][1-8])(?:=)?([qrbn])?[+#]?\b',
                # Sotilassiirto, esim. e2e4, e7xe5, etc.
                r'\b([a-h][1-8])\s?[-x]?\s?([a-h][1-8])(?:=)?([qrbn])?[+#]?\b',
                # Siirrot, joissa käytetään sanaa "to"
                r'\b([a-h][1-8])\s?to\s?([a-h][1-8])(?:=)?([qrbn])?[+#]?\b'
            ]

            move_match = None
            for pattern in move_patterns:
                move_match = re.search(pattern, response_text, re.IGNORECASE)
                if move_match:
                    break  # Käytetään ensimmäistä osumaa

            if move_match:
                move_str = move_match.group(0).upper()

                # Käsitellään kuningaskatkaukset
                if re.fullmatch(r'(?:O-O|0-0)', move_str):
                    try:
                        move = self.chess_board.parse_san('O-O')
                    except ValueError:
                        logging.warning("Error parsing move 'O-O'.")
                        return None
                elif re.fullmatch(r'(?:O-O-O|0-0-0)', move_str):
                    try:
                        move = self.chess_board.parse_san('O-O-O')
                    except ValueError:
                        logging.warning("Error parsing move 'O-O-O'.")
                        return None
                else:
                    # Rakennetaan UCI-muotoinen siirto capture-ryhmistä
                    groups = move_match.groups()
                    if groups[0].lower() in "kqrbn":
                        # Nappulasiirto, esim. qd8d1
                        if len(groups) == 4:
                            src = groups[1].lower()
                            dst = groups[2].lower()
                            promotion = groups[3].lower() if groups[3] else ''
                            uci_move = src + dst + promotion
                        else:
                            logging.warning("Expected amount of capture groups in the move.")
                            return None
                    else:
                        # Sotilassiirto
                        if len(groups) == 3:
                            src = groups[0].lower()
                            dst = groups[1].lower()
                            promotion = groups[2].lower() if groups[2] else ''
                            uci_move = src + dst + promotion
                        elif len(groups) >= 2:
                            uci_move = groups[0].lower() + groups[1].lower()
                        else:
                            logging.warning("The move does not contain enough information.")
                            return None

                    try:
                        move = chess.Move.from_uci(uci_move)
                    except ValueError:
                        logging.warning(f"Error parsing UCI - formatted move: {uci_move}")
                        return None

                if move in self.chess_board.legal_moves:
                    return move
                else:
                    logging.warning("The given move found is not legal.")
                    return None
            else:
                logging.warning("No valid move or valid move detected.")
                return None
        except Exception as e:
            logging.error(f"Error processing move: {e}")
            return None

    def process_assistant_move(self, response_text):
        """
        Jos avustajan vastauksessa on kelvollinen shakkisiirto, siirto tehdään laudalle.
        """
        try:
            # Käytä extract_move-metodia siirron tunnistamiseen
            move = self.extract_move(response_text)

            if move:
                self.chess_board.push(move)
                self.append_message(f"AI moved: {move.uci()}", sender="system")

                #self.draw_chessboard()
                self.after(0, self.draw_chessboard)

                self.send_chess_state_to_llm()
                return True
            else:
                self.append_message("AI suggested an illegal move or couldn't parse the move.", sender="system")
                return False
        except (ValueError, AttributeError) as e:
            logging.error(f"Error processing AI move: {e}")
            return False

    def reset_chess_game(self):
        self.chess_game_ended = True
        """Nollaa shakkipelin tila."""
        if messagebox.askyesno("Confirm", "Are you sure that you want to start a new game, and reset current one?"):
            try:
                self.chess_board.reset()  # Palauta lauta alkuperäiseen tilaan
                self.draw_chessboard()  # Päivitä laudankuvio
                self.append_message("The chess game has been reset.", sender="system")
                logging.info("Chess game reset by the user.")
                
                # Delete any saved game state
                if os.path.exists(CHESS_SAVE_FILE):
                    os.remove(CHESS_SAVE_FILE)

                # Tyhjennä keskusteluhistoria, jos tarpeen
                global messages
                messages = []
                self.append_message("Messages history has been deleted.", sender="system")

                self.chess_game_ended = False
            except Exception as e:
                logging.error(f"Error while resetting the chessboard: {e}")
                #self.handle_error(f"Error while resetting the chessboard: {e}")

    def update_ai_only_valid_moves(self):
        self.show_ai_only_valid_moves = self.show_ai_only_valid_moves_var.get()
        # (Vaihtoehtoisesti, voit lisätä myös debug-printauksen:
        print("Setting 'show_ai_only_valid_moves' is now:", self.show_ai_only_valid_moves)


    # ------------- Load/Save Config & Blacklist -------------
    def load_existing_api_keys(self):
        openai_key = self.config_data.get('OPENAI_API_KEY', '')
        openrouter_key = self.config_data.get('OPENROUTER_API_KEY', '')
        default_image_dir = self.config_data.get('DEFAULT_IMAGE_DIR', str(DEFAULT_IMAGE_DIR))
        vol_threshold = self.config_data.get('VOL_THRESHOLD', DEFAULT_VOL_THRESHOLD)
        min_record_duration = self.config_data.get('MIN_RECORD_DURATION', DEFAULT_MIN_RECORD_DURATION)
        max_silence_duration = self.config_data.get('MAX_SILENCE_DURATION', DEFAULT_MAX_SILENCE_DURATION)

        openrouter_url = self.config_data.get('OPENROUTER_API_URL', 'https://openrouter.ai/api/v1/chat/completions')

        self.openrouter_url_var.set(openrouter_url)

        if openai_key:
            self.openai_key_var.set("******")
            self.openai_key = openai_key
        else:
            self.openai_key = ''

        if openrouter_key:
            self.openrouter_key_var.set("******")
            self.openrouter_key = openrouter_key
        else:
            self.openrouter_key = ''

        self.image_path_var.set(default_image_dir)

        # Load Audio Source 1
        input_device1_id = self.config_data.get('INPUT_DEVICE_1', None)
        if input_device1_id is not None and self.available_input_devices1:
            label1 = next((item[0] for item in self.available_input_devices1 if item[1] == input_device1_id), None)
            if label1:
                self.audio_source1_var.set(label1)
        elif self.available_input_devices1:
            self.audio_source1_combobox.current(0)

        # Load Audio Source 2
        input_device2_id = self.config_data.get('INPUT_DEVICE_2', None)
        if input_device2_id is not None and self.available_input_devices2:
            label2 = next((item[0] for item in self.available_input_devices2 if item[1] == input_device2_id), None)
            if label2:
                self.audio_source2_var.set(label2)
        elif self.available_input_devices2:
            self.audio_source2_combobox.current(0)

        self.models_list = self.config_data.get('MODELS_LIST', [
            "openai/gpt-4o-2024-11-20",
            "openai/gpt-4o-mini",
            "openai/o1-mini",
            "openai/o1-preview",
            "openai /gpt-3.5-turbo-instruct",
            "x-ai/grok-vision-beta",
            "meta-llama/llama-3.2-90b-vision-instruct",
            "anthropic/claude-3.5-sonnet-20240620"
        ])

        self.update_personalities_listbox()
        # In the load_existing_api_keys method, add near the end:

        # Load TTS voice if available
        if hasattr(self, 'tts_voice_var') and 'TTS_VOICE' in self.config_data:
            self.tts_voice_var.set(self.config_data.get('TTS_VOICE', 'alloy'))



    def save_api_keys(self):
        openai_key = self.openai_key_var.get().strip()
        openrouter_key = self.openrouter_key_var.get().strip()
        openrouter_url = self.openrouter_url_var.get().strip()  # Get URL
        image_path = self.image_path_var.get().strip()

        if not openai_key or not openrouter_key:
            messagebox.showerror("Error", "Both API keys must be provided.")
            return

        if not Path(image_path).exists() or not Path(image_path).is_dir():
            messagebox.showerror("Error", "Selected image path is not a valid directory.")
            return

        if not messagebox.askyesno("Confirm", "Are you sure you want to save the provided API keys and image path?"):
            return

        self.config_data['OPENAI_API_KEY'] = openai_key
        self.config_data['OPENROUTER_API_KEY'] = openrouter_key
        self.config_data['OPENROUTER_API_URL'] = openrouter_url  # Save URL
        self.config_data['DEFAULT_IMAGE_DIR'] = image_path

        # Save input devices
        selected_label1 = self.audio_source1_var.get()
        selected_device1 = next((item for item in self.available_input_devices1 if item[0] == selected_label1), None)
        if selected_device1:
            self.config_data['INPUT_DEVICE_1'] = selected_device1[1]  # device_id
        else:
            self.config_data['INPUT_DEVICE_1'] = None

        selected_label2 = self.audio_source2_var.get()
        selected_device2 = next((item for item in self.available_input_devices2 if item[0] == selected_label2), None)
        if selected_device2:
            self.config_data['INPUT_DEVICE_2'] = selected_device2[1]  # device_id
        else:
            self.config_data['INPUT_DEVICE_2'] = None

        save_config(self.config_data)
        messagebox.showinfo("Success", "API keys, image path, and audio sources saved successfully.")

        self.openai_key_var.set("******")
        self.openrouter_key_var.set("******")
        self.openai_key = openai_key
        self.openrouter_key = openrouter_key
        self.api_keys_provided = True
        self.append_message("API keys, image path, and audio sources loaded successfully.", sender="system")

    def save_image_path(self):
        image_path = self.image_path_var.get().strip()

        # Validate the selected path
        if not Path(image_path).exists() or not Path(image_path).is_dir():
            messagebox.showerror("Invalid Path", "The selected path does not exist or is not a directory.")
            return

        # Confirm saving the path
        if not messagebox.askyesno("Confirm Save", "Are you sure you want to save this image path?"):
            return

        # Save to config
        self.config_data['DEFAULT_IMAGE_DIR'] = image_path
        save_config(self.config_data)

        # Provide feedback to the user
        messagebox.showinfo("Success", "Image path saved successfully.")
        self.append_message("Image path saved successfully.", sender="system")

    def save_audio_parameters(self):
        vol_threshold = self.vol_threshold_var.get().strip()
        min_record_duration = self.min_record_duration_var.get().strip()
        max_silence_duration = self.max_silence_duration_var.get().strip()

        try:
            vol_threshold = float(vol_threshold)
            min_record_duration = float(min_record_duration)
            max_silence_duration = float(max_silence_duration)
        except ValueError:
            messagebox.showerror("Invalid Input", "All audio parameters must be valid numbers.")
            return

        if vol_threshold <= 0 or min_record_duration <= 0 or max_silence_duration <= 0:
            messagebox.showerror("Invalid Input", "Parameters must be greater than 0.")
            return

        if not messagebox.askyesno("Confirm", "Are you sure you want to save the audio parameters?"):
            return

        self.config_data['VOL_THRESHOLD'] = vol_threshold
        self.config_data['MIN_RECORD_DURATION'] = min_record_duration
        self.config_data['MAX_SILENCE_DURATION'] = max_silence_duration
        self.config_data['BLACKLIST'] = self.excluded_phrases

        save_config(self.config_data)
        messagebox.showinfo("Success", "Audio parameters and blacklist saved successfully.")
        self.append_message("Audio parameters and blacklist loaded successfully.", sender="system")

    def add_blacklist_phrase(self):
        new_phrase = self.new_phrase_var.get().strip()
        if not new_phrase:
            messagebox.showerror("Input Error", "Please enter a valid phrase to add.")
            return

        if new_phrase in self.excluded_phrases:
            messagebox.showwarning("Duplicate Entry", "This phrase is already in the blacklist.")
            return

        self.excluded_phrases.append(new_phrase)
        self.blacklist_listbox.insert(tk.END, new_phrase)
        self.new_phrase_var.set("")
        messagebox.showinfo("Success", f"Phrase '{new_phrase}' added to the blacklist.")

    def remove_blacklist_phrase(self):
        selected_indices = self.blacklist_listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Selection Error", "Please select a phrase to remove.")
            return

        selected_index = selected_indices[0]
        selected_phrase = self.blacklist_listbox.get(selected_index)

        if messagebox.askyesno("Confirm Deletion",
                               f"Are you sure you want to remove '{selected_phrase}' from the blacklist?"):
            self.blacklist_listbox.delete(selected_index)
            self.excluded_phrases.remove(selected_phrase)
            messagebox.showinfo("Success", f"Phrase '{selected_phrase}' removed from the blacklist.")

    def save_llm_personality(self):
        # Hae tekstilaatikosta käyttäjän syöttämä sisältö
        personality_text = self.llm_personality_text.get("1.0", tk.END).strip()
        personality_name = self.personality_name_var.get().strip()

        if not personality_text:
            messagebox.showerror("Error", "The personality description cannot be empty.")
            return

        if not personality_name:
            messagebox.showerror("Error", "You have to provide a name for the personality.")
            return

        # Tallenna konfiguraatioon sanakirjaan
        self.config_data['LLM_PERSONALITIES'][personality_name] = personality_text
        save_config(self.config_data)
        messagebox.showinfo("Success", f"Personality '{personality_name}' saved succesfully.")
        self.append_message(f"LLM personality '{personality_name}' saved to settings.", sender="system")

        # Päivitä listbox näyttämään kaikki tallennetut persoonallisuudet
        self.update_personalities_listbox()

    def update_personalities_listbox(self):
        self.saved_personalities_listbox.delete(0, tk.END)
        personalities = self.config_data.get('LLM_PERSONALITIES', {})
        for name in personalities.keys():
            self.saved_personalities_listbox.insert(tk.END, name)

    def load_selected_personality(self):
        selected_indices = self.saved_personalities_listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Error", "First pick saved personality from the list.")
            return
        index = selected_indices[0]
        name = self.saved_personalities_listbox.get(index)
        personalities = self.config_data.get('LLM_PERSONALITIES', {})
        personality_text = personalities.get(name, "")
        # Ladataan tallennettu teksti tekstikenttään
        self.llm_personality_text.delete("1.0", tk.END)
        self.llm_personality_text.insert("1.0", personality_text)
        # Asetetaan persoonallisuuden nimi omaan kenttään
        self.personality_name_var.set(name)
        # Päivitetään LLM_PERSONALITY -avain konfiguraatiossa
        self.config_data["LLM_PERSONALITY"] = personality_text
        save_config(self.config_data)
        messagebox.showinfo("Loaded", f"Personality '{name}' loaded and applied as the default AI personality.")

    def delete_selected_personality(self):
        selected_indices = self.saved_personalities_listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Error", "First pick personality to delete from the list.")
            return
        index = selected_indices[0]
        name = self.saved_personalities_listbox.get(index)
        if messagebox.askyesno("Confirm", f"Are you sure that you want to delete the personality '{name}'?"):
            # Poistetaan avain konfiguraatiosta
            if name in self.config_data.get('LLM_PERSONALITIES', {}):
                del self.config_data['LLM_PERSONALITIES'][name]
                save_config(self.config_data)
                messagebox.showinfo("Deleted", f"Personality '{name}' deleted.")
                self.update_personalities_listbox()


    def generate_personality_prompt(self):
        """
        Reads the short description from the text box and asks the LLM to expand it
        into a detailed personality prompt. The text box is then updated with the LLM's response.
        """
        # Hae lyhyt kuvaus tekstikentästä
        short_description = self.llm_personality_text.get("1.0", tk.END).strip()
        if not short_description:
            messagebox.showerror("Error", "Please enter a short description of the desired personality first!")
            return

        # Muodosta LLM:lle lähetettävä viesti.
        # Voit säätää viestiä tarpeesi mukaan; tässä esimerkissä annetaan ohje,
        # että laajennetaan lyhyt kuvaus kattavaksi persoonallisuuspromptiksi.
        prompt_message = (
            f"You are an AI that is given a personality. Expand the following short description "
            f"into a detailed personality prompt that can be used as guidelines for the AI:\n\n"
            f"\"{short_description}\"\n\n"
            "Respond with only the personality prompt."
        )

        # Luo viestirakenne LLM API:lle (käytetään samaa rakennetta kuin muissa kutsuissa)
        messages = [
            {"role": "system", "content": "You are an expert text writer."},
            {"role": "user", "content": prompt_message}
        ]

        # (Jos haluat lähettää myös kuvan, voit tehdä sen samalla tavalla kuin muissakin osioissa;
        # tässä oletetaan, että kuvaa ei tarvitse lähettää.)

        self.update_status("Sending request to the LLM...")

        # Käytetään jo olemassa olevaa send_message_to_openrouter-metodia.
        response_json = self.send_message(
            messages=messages,
            image_path=None,  # Ei lähetetä kuvaa tässä tapauksessa
            language=self.map_language_to_code(self.language_var.get()),
            openrouter_api_key=self.openrouter_key,
            model=self.model_var.get()
        )

        if not response_json:
            return

        try:
            # Oleta, että tekoäly vastaa viestissä 'choices[0]['message']['content']'
            personality_prompt = response_json['choices'][0]['message']['content']
            # Päivitä tekstikentän sisältö saaduilla tiedoilla
            self.llm_personality_text.delete("1.0", tk.END)
            self.llm_personality_text.insert("1.0", personality_prompt)
            self.update_status("Personality prompt updated.")
            self.append_message("Personality prompt updated.", sender="system")
        except (IndexError, KeyError) as e:
            self.handle_error(f"Error processing personality prompt: {e}")

    def check_api_keys(self):
        if hasattr(self, 'openai_key') and hasattr(self, 'openrouter_key'):
            if self.openai_key and self.openrouter_key:
                self.api_keys_provided = True
                self.append_message("API keys already loaded from config.", sender="system")
                return

        self.api_keys_provided = False
        self.append_message("Please enter your API keys and click 'Save API Keys' to start.", sender="system")

    def add_model(self):
        def save_new_model():
            new_model = new_model_var.get().strip()
            if not new_model:
                messagebox.showerror("Input Error", "Model name cannot be empty.")
                return
            if new_model in self.models_list:
                messagebox.showwarning("Duplicate Entry", "This model is already in the list.")
                return

            #  Prefix with "ollama/" if it's not already an API model.
            if not (new_model.startswith("ollama/") or "/" in new_model):
                new_model = "ollama/" + new_model

            self.models_list.append(new_model)
            self.model_combobox['values'] = self.models_list
            self.model_combobox.set(new_model)
            self.config_data['MODELS_LIST'] = self.models_list
            save_config(self.config_data)
            add_window.destroy()
            messagebox.showinfo("Success", f"Model '{new_model}' has been added.")

        add_window = tk.Toplevel(self)
        add_window.title("Add New Model")
        add_window.geometry("400x150")
        add_window.resizable(False, False)

        tk.Label(add_window, text="Enter Model Identifier:").pack(pady=10)
        new_model_var = tk.StringVar()
        new_model_entry = tk.Entry(add_window, textvariable=new_model_var, width=50)
        new_model_entry.pack(pady=5)
        new_model_entry.focus_set()

        tk.Button(add_window, text="Add Model", command=save_new_model, width=15).pack(pady=10)

    def remove_model(self):
        selected_model = self.model_var.get()
        if not selected_model:
            messagebox.showerror("Selection Error", "No model selected to remove.")
            return

        if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to remove the model '{selected_model}'?"):
            self.models_list.remove(selected_model)
            self.model_combobox['values'] = self.models_list
            if self.models_list:
                self.model_combobox.current(0)
            else:
                self.model_combobox.set('')
            self.config_data['MODELS_LIST'] = self.models_list
            save_config(self.config_data)
            messagebox.showinfo("Success", f"Model '{selected_model}' has been removed.")

    #_______________Image Generation________________________________
    def generate_image(self, prompt):
        """
        Call OpenAI's Image Generation API (e.g. DALL-E) and return the generated image URL.
        """
        # Make sure you have an OpenAI client instance.
        client = OpenAI(api_key=self.openai_key)
        
        try:
            print("generating image...")
            ## Call Dall-e
            response = client.images.generate(
                model="dall-e-2",
                prompt=prompt,
                size="256x256",
                n=1,  # number of images
            )
            # The returned URL for the image
            image_url = response.data[0].url
            return image_url
        except Exception as e:
            self.append_message(f"Error while generating image: {e}", sender="system")
            return None

    def download_image(self, url, filename):
        """
        Download an image from a URL and save it to a local file.
        
        Args:
            url (str): The URL of the image to download
            filename (str): The filename to save the image as
        
        Returns:
            str or None: The local file path if successful, None otherwise
        """
        try:
            # Create images directory if it doesn't exist
            import os
            images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
            os.makedirs(images_dir, exist_ok=True)
            
            # Download the image
            import requests
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Save to file
            filepath = os.path.join(images_dir, filename)
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return filepath
        except Exception as e:
            self.append_message(f"Error downloading image: {e}", sender="system")
            return None

    def display_generated_image(self, prompt):
        """
        Display a loading animation while generating an image,
        then display the generated image in the chat.
        """
        # Create a message frame for the loading animation
        loading_frame = tk.Frame(self.chat_inner_frame, bg="#ECE5DD", padx=10, pady=5)
        loading_frame.pack(fill=tk.BOTH, expand=True, anchor="w")
        self.last_message_frame = loading_frame
        
        # Create the loading message with prompt
        loading_msg = tk.Label(
            loading_frame,
            text=f"Generating image: {prompt}",
            bg="#FFFFFF",
            fg="black",
            padx=10,
            pady=5,
            wraplength=400,
            justify="left",
            relief="groove",
            borderwidth=1
        )
        loading_msg.pack(anchor="w")
        
        # Add timestamp
        timestamp = datetime.now().strftime('%H:%M')
        time_label = tk.Label(
            loading_frame,
            text=timestamp,
            bg="#ECE5DD",
            fg="gray",
            font=("Arial", 8)
        )
        time_label.pack(anchor="w")
        
        # Update scroll to show the loading message
        self.chat_canvas.update_idletasks()
        self.chat_canvas.yview_moveto(1.0)
        
        # Start pulsing animation
        self.pulsing_animation(loading_msg)
        
        # Start image generation in a separate thread to prevent UI freezing
        self.start_thread(
            target=self.generate_and_display_image,
            args=(prompt, loading_frame),
            thread_name=f"ImageGenThread-{datetime.now().strftime('%H%M%S')}"
        )

    def pulsing_animation(self, label, alpha=1.0, decreasing=True, interval=100):
        """
        Create a pulsing text effect by changing opacity gradually.
        """
        # Store animation state in the label to be able to stop it later
        if not hasattr(label, "animation_running"):
            label.animation_running = True
        
        # Stop the animation if it should no longer run
        if not hasattr(label, "animation_running") or not label.animation_running:
            return
        
        # Update alpha (opacity)
        if decreasing:
            alpha -= 0.1
            if alpha <= 0.3:
                decreasing = False
        else:
            alpha += 0.1
            if alpha >= 1.0:
                decreasing = True
        
        # Apply opacity to text color
        gray_value = int(alpha * 255)
        color = f"#{gray_value:02x}{gray_value:02x}{gray_value:02x}"
        label.config(fg=color)
        
        # Schedule the next animation frame
        self.after(interval, lambda: self.pulsing_animation(label, alpha, decreasing, interval))

    def generate_and_display_image(self, prompt, loading_frame):
        """
        Generate an image and replace the loading animation with the actual image.
        """
        # Call the image generation function
        image_url = self.generate_image(prompt)
        
        if image_url:
            # Download the image
            local_path = self.download_image(image_url, filename="generated_image.png")
            
            if local_path:
                # Stop the animation
                for child in loading_frame.winfo_children():
                    if isinstance(child, tk.Label) and hasattr(child, "animation_running"):
                        child.animation_running = False
                
                # Remove the loading frame
                loading_frame.destroy()
                
                # Display the actual image
                self.append_image_in_chat(local_path, sender="assistant")
        else:
            # If image generation failed, update the loading message
            for child in loading_frame.winfo_children():
                if isinstance(child, tk.Label) and child.cget("wraplength") == 400:
                    child.animation_running = False
                    child.config(text=f"Failed to generate image for: {prompt}", fg="red")

    



    #_________________________Music generation______________________________
    def call_mureka_music_api(self, prompt: str):
        """
        Calls the Mureka music creation API with the given prompt.
        Returns the list of generated songs (each is a dict with `mp3_url`, etc.).
        """
        apiUrl = self.config_data.get('MUREKA_API_URL',
                                      "https://api.useapi.net/v1/mureka/music/create")  # Provide default values
        account = self.config_data.get('MUREKA_ACCOUNT', "54543547170817")
        api_token = self.config_data.get('MUREKA_API_TOKEN', "user:1449-EaueMcx5pCxwSGF8mzoCQ")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_token}"
        }
        body = {
            "account": f"{account}",
            "prompt": f"{prompt}"
        }

        try:
            response = requests.post(apiUrl, headers=headers, json=body)
            response.raise_for_status()
            json_resp = response.json()

            # Mureka returns something like: { "songs": [ {...}, {...} ] }
            songs = json_resp.get("songs", [])
            return songs

        except Exception as e:
            #self.append_message(f"Error calling Mureka: {e}", sender="system")
            self.after(0, self.append_message, f"Error calling Mureka: {e}", "system")
            return []


    def play_music_from_url(self, mp3_url: str):
        try:
            self.update_status("Downloading MP3...")

            # 1) Download MP3 file
            response = requests.get(mp3_url)
            response.raise_for_status()

            # 2) Load MP3 into AudioSegment and ensure stereo
            audio = AudioSegment.from_file(BytesIO(response.content), format="mp3")
            audio = audio.set_channels(2)  # Ensure stereo

            self.update_status("Playing downloaded MP3...")

            # 3) Convert audio to NumPy array (preserve stereo)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

            # 4) Reshape for stereo playback
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))

            # 5) Normalize audio correctly (handle all bit depths)
            if audio.sample_width == 2:  # 16-bit PCM
                samples /= np.iinfo(np.int16).max
            elif audio.sample_width == 4:  # 32-bit PCM
                samples /= np.iinfo(np.int32).max
            else:
                samples /= np.finfo(np.float32).max  # Floating point normalization

            # 6) Ensure float32 format
            samples = samples.astype(np.float32)

            # 7) Upsample if needed
            target_sample_rate = max(audio.frame_rate, 192000)
            if audio.frame_rate < target_sample_rate:
                num_samples = int(len(samples) * target_sample_rate / audio.frame_rate)
                samples = resample(samples, num_samples)

            # 8) Play audio
            sd.play(samples, samplerate=target_sample_rate)
            sd.wait()

            self.update_status("Music playback finished.")


        except Exception as e:
            self.append_message(f"Failed to play music: {e}", sender="system")


    def wants_pop_song(self, user_input: str) -> bool:
        """
        Ask the LLM: "Does the user's text indicate that they want a pop song?"
        Return True if the LLM answers 'yes', otherwise False.
        """
        # A short prompt to classify the user's intention
        classification_prompt = (
            "You are a classifier. The user said:\n\n"
            f"\"{user_input}\"\n\n"
            "Answer only 'yes' or 'no'. Does the user want to create a song"
        )

        # We'll build a minimal messages array
        classify_messages = [
            {"role": "system", "content": "You are an expert classification AI."},
            {"role": "user", "content": classification_prompt},
        ]

        # Call your usual send_message_to_openrouter:
        response_json = self.send_message(
            messages=classify_messages,
            image_path=None,
            language="en",  # classification is short, language can be English
            openrouter_api_key=self.openrouter_key,
            model="openai/gpt-4o-mini"
        )

        if not response_json:
            return False  # Can't classify, default to 'no'

        try:
            # Extract the text, normalize case, and strip whitespace.
            assistant_response = response_json['choices'][0]['message']['content'].strip().lower()

            # Use regex to find all standalone instances of 'yes' or 'no'
            tokens = re.findall(r'\b(yes|no)\b', assistant_response)
            if not tokens:
                # No clear yes/no found, so default to 'no'
                return False

            # Return True if the first encountered token is 'yes'
            return tokens[0] == 'yes'
        except (IndexError, KeyError):
            return False
        


        
    def create_tasks_tab(self):
        """Create or update the tasks tab with enhanced features"""
        # Clear any existing content in the tasks tab
        for widget in self.tasks_tab.winfo_children():
            widget.destroy()
            
        # Create main frame for tasks
        tasks_frame = tk.Frame(self.tasks_tab)
        tasks_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Task filter options
        filter_frame = tk.Frame(tasks_frame)
        filter_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.task_filter_var = tk.StringVar(value="all")
        tk.Label(filter_frame, text="Show tasks:").pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Radiobutton(filter_frame, text="All", variable=self.task_filter_var, 
                    value="all", command=self.refresh_tasks_list).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(filter_frame, text="Active", variable=self.task_filter_var,
                    value="active", command=self.refresh_tasks_list).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(filter_frame, text="Completed", variable=self.task_filter_var,
                    value="completed", command=self.refresh_tasks_list).pack(side=tk.LEFT, padx=5)
        
        # Tasks list with scrollbar
        list_frame = tk.Frame(tasks_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.tasks_listbox = tk.Listbox(list_frame, selectmode=tk.SINGLE, font=("Arial", 11), height=15)
        self.tasks_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tasks_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tasks_listbox.config(yscrollcommand=scrollbar.set)
        
        # Action buttons
        buttons_frame = tk.Frame(tasks_frame)
        buttons_frame.pack(fill=tk.X, pady=10)
        
        # Add task button
        add_button = tk.Button(buttons_frame, text="Add Task", 
                            command=self.show_add_task_dialog, width=15)
        add_button.pack(side=tk.LEFT, padx=5)
        
        # Toggle completion button
        toggle_button = tk.Button(buttons_frame, text="Toggle Completion", 
                                command=self.toggle_selected_task, width=15)
        toggle_button.pack(side=tk.LEFT, padx=5)
        
        # Delete task button
        delete_button = tk.Button(buttons_frame, text="Delete Task", 
                                command=self.delete_selected_task, bg="#FF5733", fg="white", width=15)
        delete_button.pack(side=tk.LEFT, padx=5)

        # Schedule controls
        schedule_frame = tk.LabelFrame(self.tasks_tab, text="Work/Break Schedule", padx=10, pady=10)
        schedule_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(schedule_frame, text="Work duration (minutes):").grid(row=0, column=0, sticky="w")
        self.work_entry = tk.Entry(schedule_frame, width=10)
        self.work_entry.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(schedule_frame, text="Break duration (minutes):").grid(row=1, column=0, sticky="w")
        self.break_entry = tk.Entry(schedule_frame, width=10)
        self.break_entry.grid(row=1, column=1, padx=5, pady=5)
        
        self.start_button = tk.Button(schedule_frame, text="Start Schedule", command=self.start_schedule, width=15)
        self.start_button.grid(row=2, column=0, pady=5)
        
        self.stop_button = tk.Button(schedule_frame, text="Stop Schedule", command=self.stop_schedule, width=15, state=tk.DISABLED)
        self.stop_button.grid(row=2, column=1, pady=5)
        
        self.schedule_status_label = tk.Label(schedule_frame, text="Schedule not running", font=("Arial", 10))
        self.schedule_status_label.grid(row=3, column=0, columnspan=2, pady=5)
        
        self.work_entry.insert(0, str(self.config_data.get('WORK_DURATION', 25)))
        self.break_entry.insert(0, str(self.config_data.get('BREAK_DURATION', 5)))
        self.refresh_tasks_list()        
        
        # Initial load of tasks
        self.refresh_tasks_list()

    def start_schedule(self):
        try:
            work_minutes = int(self.work_entry.get())
            break_minutes = int(self.break_entry.get())
            if work_minutes <= 0 or break_minutes <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter positive integers for durations.")
            return
        
        self.work_duration = work_minutes * 60
        self.break_duration = break_minutes * 60
        self.schedule_running = True
        self.current_state = "working"
        
        # Add initial work task
        now = datetime.now()
        self.task_manager.add_task(
            description="[SCHEDULE] Start working",
            due_date=now,
            priority="High"
        )
        self.refresh_tasks_list()
        self.append_message("Schedule started.", sender="system")
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.update_schedule_display()
        self.check_due_tasks()  # Immediate check

    def stop_schedule(self):
        self.schedule_running = False
        self.current_state = None
        self.schedule_status_label.config(text="Schedule not running")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.append_message("Schedule stopped.", sender="system")

    def update_schedule_display(self):
        if self.schedule_running:
            state_str = self.current_state if self.current_state else "starting"
            self.schedule_status_label.config(text=f"Schedule running: {state_str}")
        else:
            self.schedule_status_label.config(text="Schedule not running")

    def refresh_tasks_list(self):
        """Refresh the tasks list based on the current filter"""
        self.tasks_listbox.delete(0, tk.END)
        
        # Get all tasks
        all_tasks = self.task_manager.get_all_tasks()
        
        # Apply filter
        filter_option = self.task_filter_var.get()
        for i, task in enumerate(all_tasks):
            # Check filter criteria
            if (filter_option == "all" or 
                (filter_option == "active" and not task["completed"]) or
                (filter_option == "completed" and task["completed"])):
                
                # Format the display string
                due_date = dateutil.parser.parse(task["due_date"]).strftime("%Y-%m-%d %H:%M")
                status = "✓" if task["completed"] else "○"
                priority_icons = {"High": "❗", "Medium": "⚠️", "Low": "ℹ️"}
                priority_icon = priority_icons.get(task["priority"], "⚠️")
                
                display_text = f"{status} {priority_icon} {task['description']} (Due: {due_date})"
                
                # Add to listbox with different colors for completed/non-completed
                self.tasks_listbox.insert(tk.END, display_text)
                
                # Set item color based on completion status
                if task["completed"]:
                    self.tasks_listbox.itemconfig(i, fg="gray")
                else:
                    # Check if overdue
                    if datetime.now() > dateutil.parser.parse(task["due_date"]):
                        self.tasks_listbox.itemconfig(i, fg="red")

    def toggle_selected_task(self):
        """Toggle the completion status of the selected task"""
        selected_index = self.tasks_listbox.curselection()
        if not selected_index:
            messagebox.showinfo("Selection Required", "Please select a task first.")
            return
            
        # Convert listbox index to task index based on filter
        task_index = self.get_actual_task_index(selected_index[0])
        if task_index is not None:
            if self.task_manager.toggle_task_completion(task_index):
                self.refresh_tasks_list()
                self.append_message("Task completion status toggled.", sender="system")

    def delete_selected_task(self):
        """Delete the selected task"""
        selected_index = self.tasks_listbox.curselection()
        if not selected_index:
            messagebox.showinfo("Selection Required", "Please select a task first.")
            return
        
        # Ask for confirmation
        if not messagebox.askyesno("Confirm Deletion", 
                                "Are you sure you want to delete this task permanently?"):
            return
            
        # Convert listbox index to task index based on filter
        task_index = self.get_actual_task_index(selected_index[0])
        if task_index is not None:
            if self.task_manager.delete_task(task_index):
                self.refresh_tasks_list()
                self.append_message("Task deleted successfully.", sender="system")

    def get_actual_task_index(self, listbox_index):
        """Convert listbox index to actual task index based on current filter"""
        filter_option = self.task_filter_var.get()
        all_tasks = self.task_manager.get_all_tasks()
        
        matched_indexes = []
        for i, task in enumerate(all_tasks):
            if (filter_option == "all" or 
                (filter_option == "active" and not task["completed"]) or
                (filter_option == "completed" and task["completed"])):
                matched_indexes.append(i)
        
        if 0 <= listbox_index < len(matched_indexes):
            return matched_indexes[listbox_index]
        return None
    
    def show_add_task_dialog(self):
        """Show a dialog to add a new task"""
        dialog = tk.Toplevel(self)
        dialog.title("Add New Task")
        dialog.geometry("400x250")
        dialog.resizable(False, False)
        dialog.grab_set()
        
        # Task description
        tk.Label(dialog, text="Description:").pack(anchor="w", padx=10, pady=(10, 0))
        description_var = tk.StringVar()
        description_entry = tk.Entry(dialog, textvariable=description_var, width=40)
        description_entry.pack(fill="x", padx=10, pady=5)
        
        # Due date frame
        date_frame = tk.Frame(dialog)
        date_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(date_frame, text="Due Date:").pack(side=tk.LEFT)
        
        # Date components
        year_var = tk.StringVar(value=str(datetime.now().year))
        month_var = tk.StringVar(value=str(datetime.now().month))
        day_var = tk.StringVar(value=str(datetime.now().day))
        hour_var = tk.StringVar(value="12")
        minute_var = tk.StringVar(value="00")
        
        # Create spinboxes for date components
        year_spin = tk.Spinbox(date_frame, from_=2023, to=2030, width=5, textvariable=year_var)
        month_spin = tk.Spinbox(date_frame, from_=1, to=12, width=3, textvariable=month_var)
        day_spin = tk.Spinbox(date_frame, from_=1, to=31, width=3, textvariable=day_var)
        hour_spin = tk.Spinbox(date_frame, from_=0, to=23, width=3, textvariable=hour_var)
        minute_spin = tk.Spinbox(date_frame, from_=0, to=59, width=3, textvariable=minute_var)
        
        year_spin.pack(side=tk.LEFT, padx=2)
        tk.Label(date_frame, text="-").pack(side=tk.LEFT)
        month_spin.pack(side=tk.LEFT, padx=2)
        tk.Label(date_frame, text="-").pack(side=tk.LEFT)
        day_spin.pack(side=tk.LEFT, padx=2)
        tk.Label(date_frame, text="  Time:").pack(side=tk.LEFT, padx=(10, 0))
        hour_spin.pack(side=tk.LEFT, padx=2)
        tk.Label(date_frame, text=":").pack(side=tk.LEFT)
        minute_spin.pack(side=tk.LEFT, padx=2)
        
        # Priority
        priority_frame = tk.Frame(dialog)
        priority_frame.pack(fill="x", padx=10, pady=10)
        
        tk.Label(priority_frame, text="Priority:").pack(side=tk.LEFT)
        priority_var = tk.StringVar(value="Medium")
        priorities = ["Low", "Medium", "High"]
        
        for p in priorities:
            tk.Radiobutton(priority_frame, text=p, variable=priority_var, value=p).pack(side=tk.LEFT, padx=10)
        
        # Buttons
        button_frame = tk.Frame(dialog)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        def save_task():
            # Validate inputs
            if not description_var.get().strip():
                messagebox.showwarning("Invalid Input", "Please enter a task description.")
                return
                
            try:
                # Create datetime object from components
                due_date = datetime(
                    int(year_var.get()), 
                    int(month_var.get()),
                    int(day_var.get()),
                    int(hour_var.get()),
                    int(minute_var.get())
                )
                
                # Add the task
                self.task_manager.add_task(
                    description_var.get().strip(),
                    due_date,
                    priority_var.get()
                )
                
                # Update the task list
                self.refresh_tasks_list()
                dialog.destroy()
                self.append_message(f"New task added: {description_var.get()}", sender="system")
                
            except ValueError as e:
                messagebox.showwarning("Invalid Date", f"Please enter a valid date: {e}")
        
        save_button = tk.Button(button_frame, text="Add Task", command=save_task, width=10)
        save_button.pack(side=tk.RIGHT, padx=5)
        
        cancel_button = tk.Button(button_frame, text="Cancel", command=dialog.destroy, width=10)
        cancel_button.pack(side=tk.RIGHT, padx=5)

    def add_new_task(self):
        description = self.task_entry.get()
        due_date_str = self.due_date_entry.get()
        priority = self.priority_var.get()
        
        try:
            due_date = datetime.strptime(due_date_str, "%Y-%m-%d %H:%M")
        except ValueError:
            messagebox.showerror("Invalid Date", "Please use YYYY-MM-DD HH:MM format")
            return
            
        self.task_manager.add_task(description, due_date, priority)
        self.refresh_tasks_list()
        self.task_entry.delete(0, tk.END)
        self.append_message(f"Added task: {description}", sender="system")

    def update_task_list(self):
        for item in self.task_list.get_children():
            self.task_list.delete(item)
            
        for task in self.task_manager.tasks:
            if not task['completed']:
                self.task_list.insert("", "end", values=(
                    task['description'],
                    dateutil.parser.parse(task['due_date']).strftime("%Y-%m-%d %H:%M"),
                    task['priority']
                ))

    def start_task_checker(self):
        def task_check_loop():
            while True:
                self.check_due_tasks()
                time.sleep(self.task_check_interval)
                
        self.start_thread(
            target=task_check_loop,
            thread_name="TaskCheckerThread"
        )

    def check_due_tasks(self):
        selected_language = self.language_var.get()
        due_tasks = self.task_manager.get_due_tasks()

        if due_tasks:
            logging.info(f"Found {len(due_tasks)} due tasks: {[task['description'] for task in due_tasks]}")
        else:
            logging.info("No due tasks found")
            return        
        
        for task in due_tasks:
            
            # Generate task reminder
            reminder_prompt = (
                f"Generate a creative reminder for this task: {task['description']}. The spoken language is {selected_language}. "
                f"It is due to start on {task['due_date']}. Make it playful and motivational!"
            )
            
            # Use streaming LLM to generate reminder
            messages = [{"role": "user", "content": reminder_prompt}]
            
            # Add placeholder message before streaming begins
            self.append_message("Generating task reminder...", sender="assistant")
            
            # Start thread for streaming request to avoid UI freeze
            self.start_thread(
                target=self._stream_task_reminder,
                args=(messages, task),
                thread_name=f"TaskReminderThread-{task['description'][:10]}"
            )

    def _stream_task_reminder(self, messages, task):
        """Process task reminder using the existing chat handling logic"""
        # Get current language, model and API key
        selected_language = self.map_language_to_code(self.language_var.get())
        selected_model = self.model_var.get()
        openrouter_api_key = self.openrouter_key
        
        # Extract the user message from the messages list, or create one if needed
        if messages and len(messages) > 0 and 'content' in messages[-1]:
            user_input = messages[-1]['content']
        else:
            # Create a reminder prompt if no message exists
            user_input = (
                f"Reminder: Task '{task['description']}' is due on {task['due_date']}. "
                f"Priority: {task['priority']}. Please remind me about this task."
            )

        task['completed'] = True
        self.task_manager.save_tasks()
        self.refresh_tasks_list()
        
        # Handle schedule tasks
        if task['description'].startswith("[SCHEDULE]") and self.schedule_running:
            if task['description'] == "[SCHEDULE] Start working":
                due_date = datetime.now() + timedelta(seconds=self.work_duration)
                self.task_manager.add_task(
                    description="[SCHEDULE] Take a break",
                    due_date=due_date,
                    priority="High"
                )
                self.current_state = "working"
            elif task['description'] == "[SCHEDULE] Take a break":
                due_date = datetime.now() + timedelta(seconds=self.break_duration)
                self.task_manager.add_task(
                    description="[SCHEDULE] Start working",
                    due_date=due_date,
                    priority="High"
                )
                self.current_state = "on break"
            self.refresh_tasks_list()
            self.update_schedule_display()
        
        # Handle as a normal chat interaction (no image)
        self._handle_normal_chat(
            user_input=user_input,
            selected_language=selected_language,
            openrouter_api_key=openrouter_api_key,
            selected_model=selected_model,
            image_path=None
        )
        

        # Mark task as completed and update UI
        def check_llm_tts_completion():
            if self.tts_queue_completed and self.current_tts_stream_completed:
                # Generate music as a completion reward after a short delay
                self.after(2000, lambda: self.generate_and_play_music(task['description'], is_task=True))
            else:
                self.after(2000, check_llm_tts_completion)                                 
        
        # Schedule task completion after a delay to ensure chat processing is complete
        self.after(2000, check_llm_tts_completion)
            

    def generate_and_play_music(self, base_prompt, is_task=False):
        """
        Unified function to generate and play music based on a prompt with streaming.
        
        Args:
            base_prompt: User input or task description to base music on
            is_task: Boolean indicating if this is for a task reminder (affects prompt)
        
        Returns:
            Boolean: True if music was successfully played
        """
        
        # Create appropriate prompt for the LLM based on source
        if is_task:
            system_prompt = (
                "Only a max 300 character prompt in english for music generator. "
                "The prompt should be based on the following task (the prompt will result in "
                "a music track that will be played right now while I am working on this task, "
                "i need the generated music to motivate me with the task), use your creativity "
                f"to make the prompt best it can be, here is my task: {base_prompt}. "
                "Respond only with the prompt."
            )
        else:
            system_prompt = (
                f"Only a max 300 character prompt in english for music generator. "
                f"This is about what the user wants, use your creativity to make it best it can be: {base_prompt}"
            )
        
        # Generate prompt for music using streaming LLM
        llm_messages = [{"role": "system", "content": system_prompt}]
        
        # Add placeholder message before streaming begins
        self.append_message("Thinking about your music request...", sender="assistant")
        
        # Start thread for streaming to avoid UI freeze
        self.start_thread(
            target=self._stream_music_prompt,
            args=(llm_messages, base_prompt, is_task),
            thread_name="MusicPromptThread"
        )
        
        # Return True as we've started the process (actual result will be handled in the thread)
        return True
    
    def _stream_music_prompt(self, messages, base_prompt, is_task):
        """Process music prompt generation using async pattern to avoid UI freezing"""
        # Get current language, model and API key
        selected_language = self.map_language_to_code(self.language_var.get())
        selected_model = self.model_var.get()
        openrouter_api_key = self.openrouter_key
        
        # Create appropriate prompt
        if is_task:
            user_input = (
                "Generate a brief, creative, English music prompt (max 300 characters) for a song that "
                f"would motivate someone working on this task: {base_prompt}. Just provide the prompt, "
                "no additional explanation."
            )
        else:
            user_input = (
                "Generate a brief, creative, English music prompt (max 300 characters) for a song based on "
                f"this request: {base_prompt}. Just provide the prompt, no additional explanation."
            )
        
        # Prepare messages
        prompt_messages = messages.copy() if messages else []
        prompt_messages.append({"role": "user", "content": user_input})
        
        # Use streaming API directly instead of _handle_normal_chat
        response_json = self.send_message(
            messages=prompt_messages,
            image_path=None,
            language=selected_language,
            openrouter_api_key=openrouter_api_key,
            model=selected_model
        )
        
        # Schedule processing response in main thread
        self.after(0, lambda: self._process_music_prompt_response(response_json, base_prompt))

    def _process_music_prompt_response(self, response_json, base_prompt):
        """Process the LLM response for music generation in the main thread"""
        if not response_json:
            self.append_message("Failed to generate music prompt.", sender="system")
            return
            
        # Extract text from response
        try:
            assistant_response = response_json['choices'][0]['message']['content'].strip()
            
            # Process image instructions if present
            pattern = r'\[CREATE_IMAGE:\s*(.*?)\]'
            match = re.search(pattern, assistant_response)
            if match:
                image_prompt = match.group(1).strip()
                self.display_generated_image(prompt=image_prompt)
            
            # Clean prompt and enforce length limit
            music_prompt = re.sub(r'\s*\[CREATE_IMAGE:\s*.*?\]', '', assistant_response).strip()
            if len(music_prompt) > 300:
                music_prompt = music_prompt[:300]
                last_space = music_prompt.rfind(' ')
                if last_space > 0:
                    music_prompt = music_prompt[:last_space]
            
            # Schedule the Mureka API call to run in a background thread
            self.append_message(f"Creating a song with prompt: {music_prompt}", sender="system")
            self.start_thread(
                target=self._call_mureka_and_play,
                args=(music_prompt,),
                thread_name="MusicGenThread"
            )
        except Exception as e:
            self.append_message(f"Error processing music prompt: {e}", sender="system")



    def _call_mureka_and_play(self, music_prompt):
        """Call Mureka API and play music in background thread"""
        try:
            songs = self.call_mureka_music_api(prompt=music_prompt)
            
            if not songs:
                self.after(0, lambda: self.append_message("Sorry, I couldn't generate music at this time.", sender="system"))
                return
            
            first_song = songs[0]
            mp3_url = first_song.get("mp3_url")
            title = first_song.get("title", "<untitled>")
            
            if mp3_url:
                self.after(0, lambda: self.append_message(f"Here's your generated song: {title}\nNow playing...", sender="system"))
                self.play_music_from_url(mp3_url)
            else:
                self.after(0, lambda: self.append_message("No MP3 URL found in the response.", sender="system"))
        except Exception as e:
            self.after(0, lambda: self.append_message(f"Error generating music: {e}", sender="system"))

# --------------------------- Main Entry Point ---------------------------
if __name__ == "__main__":
    app = ChatAudioApp5()
    app.mainloop()
