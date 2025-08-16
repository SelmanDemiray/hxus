import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import os
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Import your modules (assuming they're in the same directory)
try:
    from data_processor import DataProcessor
    from neural_network import TransformerEncoderDecoder
    from training import train_model
    from chat import chat_interface, test_model
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    print("Make sure all required files are in the same directory")

class ModernStyle:
    """Modern dark theme styling constants"""
    BG_PRIMARY = "#2b2b2b"
    BG_SECONDARY = "#3c3c3c"
    BG_ACCENT = "#4a4a4a"
    FG_PRIMARY = "#ffffff"
    FG_SECONDARY = "#cccccc"
    ACCENT_BLUE = "#007acc"
    ACCENT_GREEN = "#28a745"
    ACCENT_RED = "#dc3545"
    ACCENT_ORANGE = "#fd7e14"
    FONT_MAIN = ("Segoe UI", 10)
    FONT_HEADER = ("Segoe UI", 12, "bold")
    FONT_SMALL = ("Segoe UI", 9)

class ChatBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Chat Bot - Advanced Interface")
        self.root.geometry("1200x800")
        self.root.configure(bg=ModernStyle.BG_PRIMARY)
        
        # Configure style
        self.setup_styles()
        
        # Initialize variables
        self.processor = None
        self.model = None
        self.training_thread = None
        self.is_training = False
        
        # Create main interface
        self.create_interface()
        
    def setup_styles(self):
        """Configure modern dark theme styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure notebook (tabs)
        style.configure('TNotebook', 
                       background=ModernStyle.BG_PRIMARY,
                       borderwidth=0)
        style.configure('TNotebook.Tab',
                       background=ModernStyle.BG_SECONDARY,
                       foreground=ModernStyle.FG_PRIMARY,
                       padding=[20, 10],
                       font=ModernStyle.FONT_MAIN)
        style.map('TNotebook.Tab',
                  background=[('selected', ModernStyle.ACCENT_BLUE),
                            ('active', ModernStyle.BG_ACCENT)])
        
        # Configure frames
        style.configure('Card.TFrame',
                       background=ModernStyle.BG_SECONDARY,
                       relief='flat',
                       borderwidth=1)
        
        # Configure buttons
        style.configure('Modern.TButton',
                       background=ModernStyle.ACCENT_BLUE,
                       foreground=ModernStyle.FG_PRIMARY,
                       font=ModernStyle.FONT_MAIN,
                       padding=[15, 8])
        style.map('Modern.TButton',
                  background=[('active', '#0056b3'),
                            ('pressed', '#004085')])
        
        # Configure labels
        style.configure('Heading.TLabel',
                       background=ModernStyle.BG_SECONDARY,
                       foreground=ModernStyle.FG_PRIMARY,
                       font=ModernStyle.FONT_HEADER)
        style.configure('Modern.TLabel',
                       background=ModernStyle.BG_SECONDARY,
                       foreground=ModernStyle.FG_SECONDARY,
                       font=ModernStyle.FONT_MAIN)
        
    def create_interface(self):
        """Create the main tabbed interface"""
        # Main container
        main_frame = tk.Frame(self.root, bg=ModernStyle.BG_PRIMARY)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_frame, 
                              text="🤖 Neural Network Chat Bot",
                              font=("Segoe UI", 16, "bold"),
                              bg=ModernStyle.BG_PRIMARY,
                              fg=ModernStyle.FG_PRIMARY)
        title_label.pack(pady=(0, 20))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill='both', expand=True)
        
        # Create tabs
        self.create_data_tab()
        self.create_training_tab()
        self.create_model_tab()
        self.create_test_tab()
        
    def create_data_tab(self):
        """Create data management tab"""
        data_frame = ttk.Frame(self.notebook, style='Card.TFrame')
        self.notebook.add(data_frame, text='📊 Data')
        
        # Main container with padding
        container = tk.Frame(data_frame, bg=ModernStyle.BG_SECONDARY)
        container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header
        header = tk.Label(container, text="Data Management",
                         font=ModernStyle.FONT_HEADER,
                         bg=ModernStyle.BG_SECONDARY,
                         fg=ModernStyle.FG_PRIMARY)
        header.pack(anchor='w', pady=(0, 20))
        
        # Dataset selection frame
        dataset_frame = tk.Frame(container, bg=ModernStyle.BG_SECONDARY)
        dataset_frame.pack(fill='x', pady=(0, 15))
        
        tk.Label(dataset_frame, text="Dataset Type:",
                font=ModernStyle.FONT_MAIN,
                bg=ModernStyle.BG_SECONDARY,
                fg=ModernStyle.FG_SECONDARY).pack(anchor='w')
        
        self.dataset_var = tk.StringVar(value="all")
        dataset_options = [
            ("All Available Datasets (Recommended)", "all"),
            ("Simple Q&A Patterns", "simple"),
            ("Cornell Movie Dialogs", "cornell"),
            ("Daily Dialog (Hugging Face)", "daily_dialog"),
            ("PersonaChat (Hugging Face)", "persona_chat")
        ]
        
        for text, value in dataset_options:
            tk.Radiobutton(dataset_frame, text=text, variable=self.dataset_var, value=value,
                          bg=ModernStyle.BG_SECONDARY, fg=ModernStyle.FG_SECONDARY,
                          selectcolor=ModernStyle.BG_ACCENT,
                          font=ModernStyle.FONT_MAIN).pack(anchor='w', padx=20)
        
        # Settings frame
        settings_frame = tk.Frame(container, bg=ModernStyle.BG_SECONDARY)
        settings_frame.pack(fill='x', pady=(0, 15))
        
        # Max conversations
        conv_frame = tk.Frame(settings_frame, bg=ModernStyle.BG_SECONDARY)
        conv_frame.pack(fill='x', pady=5)
        
        tk.Label(conv_frame, text="Max Conversations per Dataset:",
                font=ModernStyle.FONT_MAIN,
                bg=ModernStyle.BG_SECONDARY,
                fg=ModernStyle.FG_SECONDARY).pack(side='left')
        
        self.max_conversations_var = tk.StringVar(value="3000")
        conv_entry = tk.Entry(conv_frame, textvariable=self.max_conversations_var,
                             font=ModernStyle.FONT_MAIN, width=10,
                             bg=ModernStyle.BG_ACCENT, fg=ModernStyle.FG_PRIMARY,
                             insertbackground=ModernStyle.FG_PRIMARY)
        conv_entry.pack(side='right')
        
        # Max sequence length
        seq_frame = tk.Frame(settings_frame, bg=ModernStyle.BG_SECONDARY)
        seq_frame.pack(fill='x', pady=5)
        
        tk.Label(seq_frame, text="Max Sequence Length:",
                font=ModernStyle.FONT_MAIN,
                bg=ModernStyle.BG_SECONDARY,
                fg=ModernStyle.FG_SECONDARY).pack(side='left')
        
        self.seq_length_var = tk.StringVar(value="20")
        seq_entry = tk.Entry(seq_frame, textvariable=self.seq_length_var,
                            font=ModernStyle.FONT_MAIN, width=10,
                            bg=ModernStyle.BG_ACCENT, fg=ModernStyle.FG_PRIMARY,
                            insertbackground=ModernStyle.FG_PRIMARY)
        seq_entry.pack(side='right')
        
        # Buttons frame
        button_frame = tk.Frame(container, bg=ModernStyle.BG_SECONDARY)
        button_frame.pack(fill='x', pady=15)
        
        load_button = tk.Button(button_frame, text="🔄 Load Dataset",
                               command=self.load_dataset,
                               bg=ModernStyle.ACCENT_BLUE, fg=ModernStyle.FG_PRIMARY,
                               font=ModernStyle.FONT_MAIN, relief='flat',
                               padx=20, pady=8)
        load_button.pack(side='left', padx=(0, 10))
        
        # Status and info area
        info_frame = tk.Frame(container, bg=ModernStyle.BG_ACCENT, relief='solid', bd=1)
        info_frame.pack(fill='both', expand=True, pady=(15, 0))
        
        tk.Label(info_frame, text="Dataset Information",
                font=ModernStyle.FONT_HEADER,
                bg=ModernStyle.BG_ACCENT,
                fg=ModernStyle.FG_PRIMARY).pack(anchor='w', padx=15, pady=(10, 5))
        
        self.data_info_text = scrolledtext.ScrolledText(info_frame,
                                                       height=12,
                                                       bg=ModernStyle.BG_PRIMARY,
                                                       fg=ModernStyle.FG_SECONDARY,
                                                       font=ModernStyle.FONT_SMALL,
                                                       insertbackground=ModernStyle.FG_PRIMARY)
        self.data_info_text.pack(fill='both', expand=True, padx=15, pady=(0, 15))
        
        # Initial info
        self.data_info_text.insert('end', "Welcome to the Neural Network Chat Bot!\n\n")
        self.data_info_text.insert('end', "Click 'Load Dataset' to start loading training data.\n")
        self.data_info_text.insert('end', "You can choose from multiple dataset sources:\n\n")
        self.data_info_text.insert('end', "• All Available: Loads all copyright-free datasets\n")
        self.data_info_text.insert('end', "• Simple Q&A: Basic conversational patterns\n")
        self.data_info_text.insert('end', "• Cornell Movie Dialogs: Movie conversation dataset\n")
        self.data_info_text.insert('end', "• Daily Dialog: Daily conversation dataset\n")
        self.data_info_text.insert('end', "• PersonaChat: Personality-based conversations\n")
        
    def create_training_tab(self):
        """Create model training tab"""
        train_frame = ttk.Frame(self.notebook, style='Card.TFrame')
        self.notebook.add(train_frame, text='🎯 Training')
        
        container = tk.Frame(train_frame, bg=ModernStyle.BG_SECONDARY)
        container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header
        header = tk.Label(container, text="Model Training",
                         font=ModernStyle.FONT_HEADER,
                         bg=ModernStyle.BG_SECONDARY,
                         fg=ModernStyle.FG_PRIMARY)
        header.pack(anchor='w', pady=(0, 20))
        
        # Training parameters
        params_frame = tk.LabelFrame(container, text="Training Parameters",
                                   bg=ModernStyle.BG_SECONDARY,
                                   fg=ModernStyle.FG_PRIMARY,
                                   font=ModernStyle.FONT_MAIN)
        params_frame.pack(fill='x', pady=(0, 15))
        
        # Create parameter inputs
        param_vars = {}
        param_defaults = {
            "Epochs": "50",
            "Batch Size": "32",
            "Learning Rate": "0.01",
            "Embedding Dimension": "128",
            "Hidden Dimension": "256",
            "Number of Heads": "4"
        }
        
        for i, (param, default) in enumerate(param_defaults.items()):
            row = i // 2
            col = i % 2
            
            param_frame = tk.Frame(params_frame, bg=ModernStyle.BG_SECONDARY)
            param_frame.grid(row=row, column=col, sticky='ew', padx=10, pady=5)
            params_frame.columnconfigure(col, weight=1)
            
            tk.Label(param_frame, text=f"{param}:",
                    font=ModernStyle.FONT_MAIN,
                    bg=ModernStyle.BG_SECONDARY,
                    fg=ModernStyle.FG_SECONDARY).pack(side='left')
            
            var = tk.StringVar(value=default)
            param_vars[param.lower().replace(" ", "_")] = var
            entry = tk.Entry(param_frame, textvariable=var,
                            font=ModernStyle.FONT_MAIN, width=10,
                            bg=ModernStyle.BG_ACCENT, fg=ModernStyle.FG_PRIMARY)
            entry.pack(side='right')
        
        self.param_vars = param_vars
        
        # GPU option
        gpu_frame = tk.Frame(params_frame, bg=ModernStyle.BG_SECONDARY)
        gpu_frame.grid(row=3, column=0, columnspan=2, sticky='w', padx=10, pady=5)
        
        self.use_gpu_var = tk.BooleanVar()
        gpu_check = tk.Checkbutton(gpu_frame, text="Use GPU Acceleration (CuPy)",
                                  variable=self.use_gpu_var,
                                  bg=ModernStyle.BG_SECONDARY,
                                  fg=ModernStyle.FG_SECONDARY,
                                  selectcolor=ModernStyle.BG_ACCENT,
                                  font=ModernStyle.FONT_MAIN)
        gpu_check.pack(side='left')
        
        # Training controls
        control_frame = tk.Frame(container, bg=ModernStyle.BG_SECONDARY)
        control_frame.pack(fill='x', pady=15)
        
        self.train_button = tk.Button(control_frame, text="🚀 Start Training",
                                     command=self.start_training,
                                     bg=ModernStyle.ACCENT_GREEN, fg=ModernStyle.FG_PRIMARY,
                                     font=ModernStyle.FONT_MAIN, relief='flat',
                                     padx=20, pady=8)
        self.train_button.pack(side='left', padx=(0, 10))
        
        self.stop_button = tk.Button(control_frame, text="⏹ Stop Training",
                                    command=self.stop_training,
                                    bg=ModernStyle.ACCENT_RED, fg=ModernStyle.FG_PRIMARY,
                                    font=ModernStyle.FONT_MAIN, relief='flat',
                                    padx=20, pady=8, state='disabled')
        self.stop_button.pack(side='left')
        
        # Progress and visualization
        progress_frame = tk.Frame(container, bg=ModernStyle.BG_ACCENT, relief='solid', bd=1)
        progress_frame.pack(fill='both', expand=True, pady=(15, 0))
        
        tk.Label(progress_frame, text="Training Progress",
                font=ModernStyle.FONT_HEADER,
                bg=ModernStyle.BG_ACCENT,
                fg=ModernStyle.FG_PRIMARY).pack(anchor='w', padx=15, pady=(10, 5))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                           maximum=100, length=300)
        self.progress_bar.pack(padx=15, pady=5, fill='x')
        
        # Training log
        self.training_log = scrolledtext.ScrolledText(progress_frame,
                                                     height=10,
                                                     bg=ModernStyle.BG_PRIMARY,
                                                     fg=ModernStyle.FG_SECONDARY,
                                                     font=('Courier', 9),
                                                     insertbackground=ModernStyle.FG_PRIMARY)
        self.training_log.pack(fill='both', expand=True, padx=15, pady=(0, 15))
        
    def create_model_tab(self):
        """Create model management tab"""
        model_frame = ttk.Frame(self.notebook, style='Card.TFrame')
        self.notebook.add(model_frame, text='🧠 Model')
        
        container = tk.Frame(model_frame, bg=ModernStyle.BG_SECONDARY)
        container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header
        header = tk.Label(container, text="Model Management",
                         font=ModernStyle.FONT_HEADER,
                         bg=ModernStyle.BG_SECONDARY,
                         fg=ModernStyle.FG_PRIMARY)
        header.pack(anchor='w', pady=(0, 20))
        
        # Model paths
        paths_frame = tk.LabelFrame(container, text="Model Paths",
                                  bg=ModernStyle.BG_SECONDARY,
                                  fg=ModernStyle.FG_PRIMARY,
                                  font=ModernStyle.FONT_MAIN)
        paths_frame.pack(fill='x', pady=(0, 15))
        
        # Model path
        model_path_frame = tk.Frame(paths_frame, bg=ModernStyle.BG_SECONDARY)
        model_path_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(model_path_frame, text="Model Path:",
                font=ModernStyle.FONT_MAIN,
                bg=ModernStyle.BG_SECONDARY,
                fg=ModernStyle.FG_SECONDARY).pack(side='left')
        
        self.model_path_var = tk.StringVar(value="models/model.pkl")
        model_path_entry = tk.Entry(model_path_frame, textvariable=self.model_path_var,
                                   font=ModernStyle.FONT_MAIN,
                                   bg=ModernStyle.BG_ACCENT, fg=ModernStyle.FG_PRIMARY)
        model_path_entry.pack(side='left', fill='x', expand=True, padx=(10, 5))
        
        tk.Button(model_path_frame, text="Browse",
                 command=lambda: self.browse_file(self.model_path_var, "model"),
                 bg=ModernStyle.BG_ACCENT, fg=ModernStyle.FG_PRIMARY,
                 font=ModernStyle.FONT_SMALL, relief='flat').pack(side='right')
        
        # Processor path
        proc_path_frame = tk.Frame(paths_frame, bg=ModernStyle.BG_SECONDARY)
        proc_path_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(proc_path_frame, text="Processor Path:",
                font=ModernStyle.FONT_MAIN,
                bg=ModernStyle.BG_SECONDARY,
                fg=ModernStyle.FG_SECONDARY).pack(side='left')
        
        self.processor_path_var = tk.StringVar(value="models/processor.pkl")
        proc_path_entry = tk.Entry(proc_path_frame, textvariable=self.processor_path_var,
                                  font=ModernStyle.FONT_MAIN,
                                  bg=ModernStyle.BG_ACCENT, fg=ModernStyle.FG_PRIMARY)
        proc_path_entry.pack(side='left', fill='x', expand=True, padx=(10, 5))
        
        tk.Button(proc_path_frame, text="Browse",
                 command=lambda: self.browse_file(self.processor_path_var, "processor"),
                 bg=ModernStyle.BG_ACCENT, fg=ModernStyle.FG_PRIMARY,
                 font=ModernStyle.FONT_SMALL, relief='flat').pack(side='right')
        
        # Model operations
        ops_frame = tk.Frame(container, bg=ModernStyle.BG_SECONDARY)
        ops_frame.pack(fill='x', pady=15)
        
        load_model_btn = tk.Button(ops_frame, text="📁 Load Model",
                                  command=self.load_model,
                                  bg=ModernStyle.ACCENT_BLUE, fg=ModernStyle.FG_PRIMARY,
                                  font=ModernStyle.FONT_MAIN, relief='flat',
                                  padx=20, pady=8)
        load_model_btn.pack(side='left', padx=(0, 10))
        
        save_model_btn = tk.Button(ops_frame, text="💾 Save Model",
                                  command=self.save_model,
                                  bg=ModernStyle.ACCENT_ORANGE, fg=ModernStyle.FG_PRIMARY,
                                  font=ModernStyle.FONT_MAIN, relief='flat',
                                  padx=20, pady=8)
        save_model_btn.pack(side='left')
        
        # Model information
        info_frame = tk.Frame(container, bg=ModernStyle.BG_ACCENT, relief='solid', bd=1)
        info_frame.pack(fill='both', expand=True, pady=(15, 0))
        
        tk.Label(info_frame, text="Model Information",
                font=ModernStyle.FONT_HEADER,
                bg=ModernStyle.BG_ACCENT,
                fg=ModernStyle.FG_PRIMARY).pack(anchor='w', padx=15, pady=(10, 5))
        
        self.model_info_text = scrolledtext.ScrolledText(info_frame,
                                                        height=12,
                                                        bg=ModernStyle.BG_PRIMARY,
                                                        fg=ModernStyle.FG_SECONDARY,
                                                        font=ModernStyle.FONT_SMALL,
                                                        insertbackground=ModernStyle.FG_PRIMARY)
        self.model_info_text.pack(fill='both', expand=True, padx=15, pady=(0, 15))
        
    def create_test_tab(self):
        """Create testing and inference tab"""
        test_frame = ttk.Frame(self.notebook, style='Card.TFrame')
        self.notebook.add(test_frame, text='🧪 Test Inference')
        
        container = tk.Frame(test_frame, bg=ModernStyle.BG_SECONDARY)
        container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header
        header = tk.Label(container, text="Model Testing & Chat Interface",
                         font=ModernStyle.FONT_HEADER,
                         bg=ModernStyle.BG_SECONDARY,
                         fg=ModernStyle.FG_PRIMARY)
        header.pack(anchor='w', pady=(0, 20))
        
        # Test controls
        controls_frame = tk.Frame(container, bg=ModernStyle.BG_SECONDARY)
        controls_frame.pack(fill='x', pady=(0, 15))
        
        predefined_test_btn = tk.Button(controls_frame, text="🎯 Run Predefined Tests",
                                       command=self.run_predefined_tests,
                                       bg=ModernStyle.ACCENT_BLUE, fg=ModernStyle.FG_PRIMARY,
                                       font=ModernStyle.FONT_MAIN, relief='flat',
                                       padx=20, pady=8)
        predefined_test_btn.pack(side='left', padx=(0, 10))
        
        # Chat interface
        chat_frame = tk.LabelFrame(container, text="Interactive Chat",
                                 bg=ModernStyle.BG_SECONDARY,
                                 fg=ModernStyle.FG_PRIMARY,
                                 font=ModernStyle.FONT_MAIN)
        chat_frame.pack(fill='both', expand=True, pady=(0, 15))
        
        # Chat display
        self.chat_display = scrolledtext.ScrolledText(chat_frame,
                                                     height=15,
                                                     bg=ModernStyle.BG_PRIMARY,
                                                     fg=ModernStyle.FG_SECONDARY,
                                                     font=ModernStyle.FONT_MAIN,
                                                     insertbackground=ModernStyle.FG_PRIMARY,
                                                     state='disabled')
        self.chat_display.pack(fill='both', expand=True, padx=10, pady=(10, 5))
        
        # Chat input
        input_frame = tk.Frame(chat_frame, bg=ModernStyle.BG_SECONDARY)
        input_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        tk.Label(input_frame, text="You:",
                font=ModernStyle.FONT_MAIN,
                bg=ModernStyle.BG_SECONDARY,
                fg=ModernStyle.FG_SECONDARY).pack(side='left')
        
        self.chat_input_var = tk.StringVar()
        self.chat_input = tk.Entry(input_frame, textvariable=self.chat_input_var,
                                  font=ModernStyle.FONT_MAIN,
                                  bg=ModernStyle.BG_ACCENT, fg=ModernStyle.FG_PRIMARY,
                                  insertbackground=ModernStyle.FG_PRIMARY)
        self.chat_input.pack(side='left', fill='x', expand=True, padx=(10, 5))
        self.chat_input.bind('<Return>', self.send_message)
        
        send_btn = tk.Button(input_frame, text="Send",
                            command=self.send_message,
                            bg=ModernStyle.ACCENT_GREEN, fg=ModernStyle.FG_PRIMARY,
                            font=ModernStyle.FONT_MAIN, relief='flat',
                            padx=15, pady=5)
        send_btn.pack(side='right')
        
        # Clear chat button
        clear_btn = tk.Button(input_frame, text="Clear",
                             command=self.clear_chat,
                             bg=ModernStyle.ACCENT_RED, fg=ModernStyle.FG_PRIMARY,
                             font=ModernStyle.FONT_MAIN, relief='flat',
                             padx=15, pady=5)
        clear_btn.pack(side='right', padx=(5, 0))
        
        # Initial chat message
        self.add_to_chat("Bot", "Hello! I'm your neural network chat bot. Load a trained model to start chatting!")
        
    def add_to_chat(self, sender, message):
        """Add message to chat display"""
        self.chat_display.config(state='normal')
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_display.insert('end', f"[{timestamp}] {sender}: {message}\n")
        self.chat_display.config(state='disabled')
        self.chat_display.see('end')
        
    def send_message(self, event=None):
        """Send user message and get bot response"""
        message = self.chat_input_var.get().strip()
        if not message:
            return
            
        if not self.model or not self.processor:
            self.add_to_chat("System", "Please load a trained model first!")
            return
            
        # Add user message
        self.add_to_chat("You", message)
        self.chat_input_var.set("")
        
        # Get bot response (this is a placeholder - you'll need to implement the actual prediction)
        try:
            # For now, just echo back - replace with actual model prediction
            response = f"Echo: {message} (Model prediction not yet implemented in GUI)"
            self.add_to_chat("Bot", response)
        except Exception as e:
            self.add_to_chat("System", f"Error generating response: {str(e)}")
            
    def clear_chat(self):
        """Clear the chat display"""
        self.chat_display.config(state='normal')
        self.chat_display.delete(1.0, 'end')
        self.chat_display.config(state='disabled')
        self.add_to_chat("System", "Chat cleared.")
        
    def browse_file(self, var, file_type):
        """Browse for file path"""
        if file_type == "model":
            filetypes = [("Pickle files", "*.pkl"), ("All files", "*.*")]
        else:
            filetypes = [("Pickle files", "*.pkl"), ("All files", "*.*")]
            
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            var.set(filename)
            
    def log_message(self, message):
        """Add message to training log"""
        self.training_log.insert('end', f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
        self.training_log.see('end')
        self.root.update()
        
    def load_dataset(self):
        """Load dataset in a separate thread"""
        def load_thread():
            try:
                self.log_data_message("Starting dataset loading...")
                
                max_conv = int(self.max_conversations_var.get())
                seq_len = int(self.seq_length_var.get())
                dataset_type = self.dataset_var.get()
                
                # Initialize processor
                self.processor = DataProcessor(max_sequence_length=seq_len)
                
                # Load dataset
                self.log_data_message(f"Loading {dataset_type} dataset with max {max_conv} conversations...")
                questions, answers = self.processor.load_dataset(dataset_type, max_conv)
                
                # Prepare data
                self.log_data_message("Preparing training data...")
                self.encoder_inputs, self.decoder_inputs, self.decoder_targets = \
                    self.processor.prepare_data(questions, answers)
                
                # Update info
                info = f"Dataset loaded successfully!\n\n"
                info += f"Dataset Type: {dataset_type}\n"
                info += f"Total Question-Answer Pairs: {len(questions)}\n"
                info += f"Vocabulary Size: {self.processor.vocab_size}\n"
                info += f"Max Sequence Length: {seq_len}\n"
                info += f"Encoder Input Shape: {self.encoder_inputs.shape}\n"
                info += f"Decoder Input Shape: {self.decoder_inputs.shape}\n"
                info += f"Decoder Target Shape: {self.decoder_targets.shape}\n\n"
                info += "Sample conversations:\n"
                for i in range(min(3, len(questions))):
                    info += f"Q: {questions[i][:50]}...\n"
                    info += f"A: {answers[i][:50]}...\n\n"
                
                self.update_data_info(info)
                self.log_data_message("Dataset loading completed!")
                
                # Save processor
                os.makedirs("models", exist_ok=True)
                self.processor.save_processor("models/processor.pkl")
                self.log_data_message("Data processor saved to models/processor.pkl")
                
            except Exception as e:
                error_msg = f"Error loading dataset: {str(e)}"
                self.log_data_message(error_msg)
                messagebox.showerror("Dataset Loading Error", error_msg)
        
        threading.Thread(target=load_thread, daemon=True).start()
        
    def log_data_message(self, message):
        """Add message to data info text"""
        def update_ui():
            self.data_info_text.insert('end', f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
            self.data_info_text.see('end')
        
        self.root.after(0, update_ui)
        
    def update_data_info(self, info):
        """Update data info text"""
        def update_ui():
            self.data_info_text.delete(1.0, 'end')
            self.data_info_text.insert(1.0, info)
        
        self.root.after(0, update_ui)
        
    def start_training(self):
        """Start model training in a separate thread"""
        if not hasattr(self, 'processor') or not self.processor:
            messagebox.showerror("Error", "Please load a dataset first!")
            return
            
        if self.is_training:
            messagebox.showwarning("Warning", "Training is already in progress!")
            return
            
        def training_thread():
            try:
                self.is_training = True
                self.train_button.config(state='disabled')
                self.stop_button.config(state='normal')
                
                # Get parameters
                epochs = int(self.param_vars["epochs"].get())
                batch_size = int(self.param_vars["batch_size"].get())
                learning_rate = float(self.param_vars["learning_rate"].get())
                embedding_dim = int(self.param_vars["embedding_dimension"].get())
                hidden_dim = int(self.param_vars["hidden_dimension"].get())
                num_heads = int(self.param_vars["number_of_heads"].get())
                use_gpu = self.use_gpu_var.get()
                
                # Select array module
                if use_gpu:
                    try:
                        import cupy as cp
                        xp = cp
                        self.log_message("Using CuPy for GPU acceleration.")
                    except ImportError:
                        self.log_message("CuPy not installed. Falling back to NumPy.")
                        xp = np
                else:
                    xp = np
                    
                self.log_message(f"Starting training with parameters:")
                self.log_message(f"Epochs: {epochs}, Batch Size: {batch_size}")
                self.log_message(f"Learning Rate: {learning_rate}")
                self.log_message(f"Model Architecture: {embedding_dim}d embedding, {hidden_dim}d hidden, {num_heads} heads")
                
                # Initialize model
                self.model = TransformerEncoderDecoder(
                    vocab_size=self.processor.vocab_size,
                    embedding_dim=embedding_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    xp=xp
                )
                
                # Convert data to appropriate arrays
                encoder_inputs = self.encoder_inputs
                decoder_inputs = self.decoder_inputs
                decoder_targets = self.decoder_targets
                
                if use_gpu and xp.__name__ == "cupy":
                    encoder_inputs = xp.array(encoder_inputs)
                    decoder_inputs = xp.array(decoder_inputs)
                    decoder_targets = xp.array(decoder_targets)
                
                # Custom training loop with progress updates
                self.train_model_with_progress(
                    encoder_inputs, decoder_inputs, decoder_targets,
                    epochs, batch_size, learning_rate
                )
                
            except Exception as e:
                error_msg = f"Training error: {str(e)}"
                self.log_message(error_msg)
                messagebox.showerror("Training Error", error_msg)
            finally:
                self.is_training = False
                self.root.after(0, lambda: self.train_button.config(state='normal'))
                self.root.after(0, lambda: self.stop_button.config(state='disabled'))
                
        self.training_thread = threading.Thread(target=training_thread, daemon=True)
        self.training_thread.start()
        
    def train_model_with_progress(self, encoder_inputs, decoder_inputs, decoder_targets, 
                                 epochs, batch_size, learning_rate):
        """Custom training loop with GUI progress updates"""
        xp = self.model.xp
        num_samples = encoder_inputs.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))
        
        epoch_losses = []
        
        for epoch in range(epochs):
            if not self.is_training:  # Check for stop signal
                break
                
            start_time = time.time()
            total_loss = 0
            
            # Update progress
            progress = (epoch / epochs) * 100
            self.root.after(0, lambda p=progress: self.progress_var.set(p))
            
            # Shuffle data
            indices = np.random.permutation(num_samples)
            if xp.__name__ == "cupy":
                indices = xp.array(indices)
                
            shuffled_encoder = encoder_inputs[indices]
            shuffled_decoder = decoder_inputs[indices]
            shuffled_targets = decoder_targets[indices]
            
            for batch in range(num_batches):
                if not self.is_training:
                    break
                    
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, num_samples)
                
                batch_encoder = shuffled_encoder[start_idx:end_idx]
                batch_decoder = shuffled_decoder[start_idx:end_idx]
                batch_targets = shuffled_targets[start_idx:end_idx]
                
                # Training step
                batch_loss, _ = self.model.train_step(
                    batch_encoder, batch_decoder, batch_targets, learning_rate
                )
                
                # Convert loss to scalar if needed
                if xp.__name__ == "cupy":
                    batch_loss_scalar = float(batch_loss)
                else:
                    batch_loss_scalar = batch_loss
                    
                total_loss += batch_loss_scalar * (end_idx - start_idx)
                
                # Log progress every 10 batches
                if batch % 10 == 0:
                    msg = f"Epoch {epoch+1}/{epochs}, Batch {batch+1}/{num_batches}, Loss: {batch_loss_scalar:.4f}"
                    self.root.after(0, lambda m=msg: self.log_message(m))
            
            avg_loss = total_loss / num_samples
            epoch_time = time.time() - start_time
            epoch_losses.append(avg_loss)
            
            msg = f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}"
            self.root.after(0, lambda m=msg: self.log_message(m))
            
            # Save model periodically
            if (epoch + 1) % 5 == 0:
                os.makedirs("models", exist_ok=True)
                self.model.save_model("models")
                msg = f"Checkpoint saved at epoch {epoch+1}"
                self.root.after(0, lambda m=msg: self.log_message(m))
        
        # Final save
        if self.is_training:  # Only if not stopped
            self.model.save_model("models")
            self.root.after(0, lambda: self.log_message("Training completed! Final model saved."))
            self.root.after(0, lambda: self.progress_var.set(100))
        else:
            self.root.after(0, lambda: self.log_message("Training stopped by user."))
            
    def stop_training(self):
        """Stop the training process"""
        self.is_training = False
        self.log_message("Stopping training...")
        
    def load_model(self):
        """Load a trained model"""
        try:
            model_path = self.model_path_var.get()
            processor_path = self.processor_path_var.get()
            
            if not os.path.exists(model_path):
                messagebox.showerror("Error", f"Model file not found: {model_path}")
                return
                
            if not os.path.exists(processor_path):
                messagebox.showerror("Error", f"Processor file not found: {processor_path}")
                return
            
            # Load processor
            self.processor = DataProcessor.load_processor(processor_path)
            
            # Load model
            self.model = TransformerEncoderDecoder.load_model(model_path)
            
            # Update model info
            info = f"Model loaded successfully!\n\n"
            info += f"Model Path: {model_path}\n"
            info += f"Processor Path: {processor_path}\n"
            info += f"Vocabulary Size: {self.processor.vocab_size}\n"
            info += f"Max Sequence Length: {self.processor.max_sequence_length}\n"
            info += f"Model Architecture:\n"
            info += f"  - Embedding Dimension: {self.model.embedding_dim}\n"
            info += f"  - Hidden Dimension: {self.model.hidden_dim}\n"
            info += f"  - Number of Heads: {self.model.num_heads}\n"
            info += f"  - Array Library: {self.model.xp.__name__}\n"
            
            self.model_info_text.delete(1.0, 'end')
            self.model_info_text.insert(1.0, info)
            
            messagebox.showinfo("Success", "Model and processor loaded successfully!")
            
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            messagebox.showerror("Model Loading Error", error_msg)
            
    def save_model(self):
        """Save the current model"""
        if not self.model:
            messagebox.showerror("Error", "No model to save!")
            return
            
        try:
            # Create models directory
            os.makedirs("models", exist_ok=True)
            
            # Save model
            self.model.save_model("models")
            
            # Save processor if available
            if self.processor:
                self.processor.save_processor("models/processor.pkl")
            
            messagebox.showinfo("Success", "Model saved successfully to models/ directory!")
            
        except Exception as e:
            error_msg = f"Error saving model: {str(e)}"
            messagebox.showerror("Model Saving Error", error_msg)
            
    def run_predefined_tests(self):
        """Run predefined test questions"""
        if not self.model or not self.processor:
            messagebox.showerror("Error", "Please load a trained model first!")
            return
            
        test_questions = [
            "hello how are you",
            "what is your name",
            "how does this work",
            "tell me a joke",
            "what time is it",
            "goodbye",
            "thank you",
            "help me"
        ]
        
        self.add_to_chat("System", "Running predefined tests...")
        
        for i, question in enumerate(test_questions):
            try:
                # For now, just echo - replace with actual model prediction
                response = f"Test response {i+1}: {question} (Prediction not implemented)"
                self.add_to_chat(f"Test Q{i+1}", question)
                self.add_to_chat(f"Test A{i+1}", response)
            except Exception as e:
                self.add_to_chat("System", f"Error testing question '{question}': {str(e)}")
                
        self.add_to_chat("System", "Predefined tests completed!")

def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    
    # Set window icon (if available)
    try:
        # You can add an icon file here
        # root.iconbitmap('icon.ico')
        pass
    except:
        pass
    
    app = ChatBotGUI(root)
    
    # Center the window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    # Make window resizable
    root.minsize(800, 600)
    
    # Start the GUI
    root.mainloop()

if __name__ == "__main__":
    main()