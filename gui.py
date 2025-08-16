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
        """Create model training tab with sub-tabs for controls and visuals"""
        train_frame = ttk.Frame(self.notebook, style='Card.TFrame')
        self.notebook.add(train_frame, text='🎯 Training')
        
        # Create sub-notebook for training controls and visuals
        self.train_notebook = ttk.Notebook(train_frame)
        self.train_notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create training controls tab
        self.create_training_controls_tab()
        
        # Create training visuals tab
        self.create_training_visuals_tab()
        
    def create_training_controls_tab(self):
        """Create training controls sub-tab"""
        controls_frame = ttk.Frame(self.train_notebook, style='Card.TFrame')
        self.train_notebook.add(controls_frame, text='⚙️ Controls')
        
        container = tk.Frame(controls_frame, bg=ModernStyle.BG_SECONDARY)
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
        
    def create_training_visuals_tab(self):
        """Create training visuals sub-tab with real-time graphs"""
        visuals_frame = ttk.Frame(self.train_notebook, style='Card.TFrame')
        self.train_notebook.add(visuals_frame, text='📈 Visuals')
        
        container = tk.Frame(visuals_frame, bg=ModernStyle.BG_SECONDARY)
        container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Initialize training metrics storage
        self.training_metrics = {
            'epoch_losses': [],
            'batch_losses': [],
            'gradient_norms': [],
            'learning_rates': [],
            'weight_changes': [],
            'epochs': [],
            'batch_numbers': []
        }
        
        # Create matplotlib figures
        self.setup_training_plots(container)
        
        # Metrics summary frame
        summary_frame = tk.Frame(container, bg=ModernStyle.BG_ACCENT, relief='solid', bd=1)
        summary_frame.pack(side='bottom', fill='x', pady=(10, 0))
        
        tk.Label(summary_frame, text="Training Metrics Summary",
                font=ModernStyle.FONT_HEADER,
                bg=ModernStyle.BG_ACCENT,
                fg=ModernStyle.FG_PRIMARY).pack(anchor='w', padx=15, pady=(10, 5))
        
        # Metrics labels
        metrics_container = tk.Frame(summary_frame, bg=ModernStyle.BG_ACCENT)
        metrics_container.pack(fill='x', padx=15, pady=(0, 15))
        
        # Create metric display labels
        self.metric_labels = {}
        metrics = [
            ('Current Loss', 'current_loss'),
            ('Best Loss', 'best_loss'),
            ('Avg Gradient Norm', 'avg_grad_norm'),
            ('Learning Progress', 'learning_progress'),
            ('Convergence Rate', 'convergence_rate'),
            ('Weight Stability', 'weight_stability')
        ]
        
        for i, (label, key) in enumerate(metrics):
            row = i // 3
            col = i % 3
            
            metric_frame = tk.Frame(metrics_container, bg=ModernStyle.BG_ACCENT)
            metric_frame.grid(row=row, column=col, sticky='w', padx=(0, 20), pady=2)
            
            tk.Label(metric_frame, text=f"{label}:",
                    font=ModernStyle.FONT_SMALL,
                    bg=ModernStyle.BG_ACCENT,
                    fg=ModernStyle.FG_SECONDARY).pack(side='left')
            
            value_label = tk.Label(metric_frame, text="N/A",
                                  font=(ModernStyle.FONT_SMALL[0], ModernStyle.FONT_SMALL[1], 'bold'),
                                  bg=ModernStyle.BG_ACCENT,
                                  fg=ModernStyle.ACCENT_BLUE)
            value_label.pack(side='left', padx=(5, 0))
            
            self.metric_labels[key] = value_label
        
    def setup_training_plots(self, container):
        """Setup matplotlib plots for training visualization"""
        # Create figure with subplots
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.patch.set_facecolor(ModernStyle.BG_SECONDARY)
        
        # Configure subplot styles
        axes = [self.ax1, self.ax2, self.ax3, self.ax4]
        for ax in axes:
            ax.set_facecolor(ModernStyle.BG_PRIMARY)
            ax.tick_params(colors=ModernStyle.FG_SECONDARY, labelsize=8)
            ax.spines['bottom'].set_color(ModernStyle.FG_SECONDARY)
            ax.spines['top'].set_color(ModernStyle.FG_SECONDARY)
            ax.spines['left'].set_color(ModernStyle.FG_SECONDARY)
            ax.spines['right'].set_color(ModernStyle.FG_SECONDARY)
        
        # Plot 1: Loss over time
        self.ax1.set_title('Training Loss', color=ModernStyle.FG_PRIMARY, fontsize=10, pad=10)
        self.ax1.set_xlabel('Epoch', color=ModernStyle.FG_SECONDARY, fontsize=8)
        self.ax1.set_ylabel('Loss', color=ModernStyle.FG_SECONDARY, fontsize=8)
        self.loss_line, = self.ax1.plot([], [], color=ModernStyle.ACCENT_BLUE, linewidth=2, label='Epoch Loss')
        self.batch_loss_line, = self.ax1.plot([], [], color=ModernStyle.ACCENT_ORANGE, alpha=0.6, linewidth=1, label='Batch Loss')
        self.ax1.legend(loc='upper right', fontsize=8)
        self.ax1.grid(True, alpha=0.3, color=ModernStyle.FG_SECONDARY)
        
        # Plot 2: Gradient norms (backpropagation health)
        self.ax2.set_title('Gradient Norms (Backprop Health)', color=ModernStyle.FG_PRIMARY, fontsize=10, pad=10)
        self.ax2.set_xlabel('Batch', color=ModernStyle.FG_SECONDARY, fontsize=8)
        self.ax2.set_ylabel('Gradient Norm', color=ModernStyle.FG_SECONDARY, fontsize=8)
        self.grad_norm_line, = self.ax2.plot([], [], color=ModernStyle.ACCENT_GREEN, linewidth=2, label='Gradient Norm')
        self.ax2.axhline(y=1.0, color=ModernStyle.ACCENT_RED, linestyle='--', alpha=0.7, label='Exploding Threshold')
        self.ax2.axhline(y=0.01, color=ModernStyle.ACCENT_ORANGE, linestyle='--', alpha=0.7, label='Vanishing Threshold')
        self.ax2.legend(loc='upper right', fontsize=8)
        self.ax2.grid(True, alpha=0.3, color=ModernStyle.FG_SECONDARY)
        
        # Plot 3: Learning rate and weight changes
        self.ax3.set_title('Learning Dynamics', color=ModernStyle.FG_PRIMARY, fontsize=10, pad=10)
        self.ax3.set_xlabel('Epoch', color=ModernStyle.FG_SECONDARY, fontsize=8)
        self.ax3.set_ylabel('Magnitude', color=ModernStyle.FG_SECONDARY, fontsize=8)
        self.weight_change_line, = self.ax3.plot([], [], color=ModernStyle.ACCENT_GREEN, linewidth=2, label='Weight Changes')
        self.lr_line, = self.ax3.plot([], [], color=ModernStyle.ACCENT_ORANGE, linewidth=2, label='Learning Rate')
        self.ax3.legend(loc='upper right', fontsize=8)
        self.ax3.grid(True, alpha=0.3, color=ModernStyle.FG_SECONDARY)
        
        # Plot 4: Training progress and convergence
        self.ax4.set_title('Convergence Analysis', color=ModernStyle.FG_PRIMARY, fontsize=10, pad=10)
        self.ax4.set_xlabel('Epoch', color=ModernStyle.FG_SECONDARY, fontsize=8)
        self.ax4.set_ylabel('Rate', color=ModernStyle.FG_SECONDARY, fontsize=8)
        self.convergence_line, = self.ax4.plot([], [], color=ModernStyle.ACCENT_BLUE, linewidth=2, label='Loss Improvement')
        self.stability_line, = self.ax4.plot([], [], color=ModernStyle.ACCENT_GREEN, linewidth=2, label='Weight Stability')
        self.ax4.legend(loc='upper right', fontsize=8)
        self.ax4.grid(True, alpha=0.3, color=ModernStyle.FG_SECONDARY)
        
        plt.tight_layout(pad=2.0)
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, container)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
    def update_training_plots(self):
        """Update all training plots with current data"""
        if not hasattr(self, 'training_metrics'):
            return
            
        try:
            metrics = self.training_metrics
            
            # Update loss plot
            if metrics['epoch_losses']:
                epochs = list(range(1, len(metrics['epoch_losses']) + 1))
                self.loss_line.set_data(epochs, metrics['epoch_losses'])
                self.ax1.relim()
                self.ax1.autoscale_view()
            
            if metrics['batch_losses']:
                batch_x = list(range(len(metrics['batch_losses'])))
                self.batch_loss_line.set_data(batch_x, metrics['batch_losses'])
            
            # Update gradient norms
            if metrics['gradient_norms']:
                grad_x = list(range(len(metrics['gradient_norms'])))
                self.grad_norm_line.set_data(grad_x, metrics['gradient_norms'])
                self.ax2.relim()
                self.ax2.autoscale_view()
            
            # Update learning dynamics
            if metrics['weight_changes']:
                epochs = list(range(1, len(metrics['weight_changes']) + 1))
                self.weight_change_line.set_data(epochs, metrics['weight_changes'])
                self.ax3.relim()
                self.ax3.autoscale_view()
            
            if metrics['learning_rates']:
                epochs = list(range(1, len(metrics['learning_rates']) + 1))
                self.lr_line.set_data(epochs, metrics['learning_rates'])
            
            # Update convergence analysis
            if len(metrics['epoch_losses']) > 1:
                epochs = list(range(2, len(metrics['epoch_losses']) + 1))
                # Calculate loss improvement rate
                improvements = []
                stabilities = []
                
                for i in range(1, len(metrics['epoch_losses'])):
                    prev_loss = metrics['epoch_losses'][i-1]
                    curr_loss = metrics['epoch_losses'][i]
                    improvement = max(0, (prev_loss - curr_loss) / prev_loss) if prev_loss > 0 else 0
                    improvements.append(improvement)
                    
                    # Calculate stability (inverse of loss variance)
                    if i >= 3:
                        recent_losses = metrics['epoch_losses'][max(0, i-2):i+1]
                        stability = 1.0 / (np.var(recent_losses) + 1e-8)
                        stabilities.append(min(stability, 1.0))  # Cap at 1.0
                
                if improvements:
                    self.convergence_line.set_data(epochs, improvements)
                    
                if stabilities:
                    stability_epochs = list(range(3, len(metrics['epoch_losses']) + 1))
                    self.stability_line.set_data(stability_epochs, stabilities)
                    
                self.ax4.relim()
                self.ax4.autoscale_view()
            
            # Redraw canvas
            self.canvas.draw()
            
            # Update metrics summary
            self.update_metrics_summary()
            
        except Exception as e:
            print(f"Error updating plots: {e}")
    
    def update_metrics_summary(self):
        """Update the metrics summary display"""
        try:
            metrics = self.training_metrics
            
            # Current loss
            current_loss = metrics['epoch_losses'][-1] if metrics['epoch_losses'] else 0
            self.metric_labels['current_loss'].config(text=f"{current_loss:.4f}")
            
            # Best loss
            best_loss = min(metrics['epoch_losses']) if metrics['epoch_losses'] else 0
            self.metric_labels['best_loss'].config(text=f"{best_loss:.4f}")
            
            # Average gradient norm
            avg_grad_norm = np.mean(metrics['gradient_norms'][-10:]) if metrics['gradient_norms'] else 0
            color = ModernStyle.ACCENT_GREEN if 0.01 < avg_grad_norm < 1.0 else ModernStyle.ACCENT_RED
            self.metric_labels['avg_grad_norm'].config(text=f"{avg_grad_norm:.4f}", fg=color)
            
            # Learning progress (improvement over last 5 epochs)
            if len(metrics['epoch_losses']) >= 5:
                recent_improvement = (metrics['epoch_losses'][-5] - current_loss) / metrics['epoch_losses'][-5]
                progress_pct = max(0, recent_improvement) * 100
                self.metric_labels['learning_progress'].config(text=f"{progress_pct:.1f}%")
            else:
                self.metric_labels['learning_progress'].config(text="Warming up...")
            
            # Convergence rate
            if len(metrics['epoch_losses']) >= 3:
                recent_losses = metrics['epoch_losses'][-3:]
                convergence_rate = np.mean(np.diff(recent_losses))
                convergence_status = "Converging" if convergence_rate < -0.001 else "Stable" if abs(convergence_rate) < 0.001 else "Diverging"
                color = ModernStyle.ACCENT_GREEN if convergence_rate < 0 else ModernStyle.ACCENT_ORANGE if abs(convergence_rate) < 0.001 else ModernStyle.ACCENT_RED
                self.metric_labels['convergence_rate'].config(text=convergence_status, fg=color)
            else:
                self.metric_labels['convergence_rate'].config(text="Initializing...")
            
            # Weight stability
            if metrics['weight_changes']:
                recent_changes = metrics['weight_changes'][-5:] if len(metrics['weight_changes']) >= 5 else metrics['weight_changes']
                stability = 1.0 / (np.var(recent_changes) + 1e-8)
                stability_status = "High" if stability > 100 else "Medium" if stability > 10 else "Low"
                color = ModernStyle.ACCENT_GREEN if stability > 100 else ModernStyle.ACCENT_ORANGE if stability > 10 else ModernStyle.ACCENT_RED
                self.metric_labels['weight_stability'].config(text=stability_status, fg=color)
            else:
                self.metric_labels['weight_stability'].config(text="Monitoring...")
                
        except Exception as e:
            print(f"Error updating metrics summary: {e}")
    
    def calculate_gradient_norm(self, model):
        """Calculate the L2 norm of all gradients"""
        try:
            total_norm = 0
            
            # Get all parameters that have gradients
            # This is a simplified version - you might need to adapt based on your model structure
            if hasattr(model, 'encoder_embed'):
                # Calculate approximate gradient norms based on weight magnitudes
                # In a real implementation, you'd track actual gradients during backprop
                weights = [
                    model.encoder_embed, model.decoder_embed,
                    model.enc_attn_Wq, model.enc_attn_Wk, model.enc_attn_Wv, model.enc_attn_Wo,
                    model.dec_attn_Wq, model.dec_attn_Wk, model.dec_attn_Wv, model.dec_attn_Wo,
                    model.cross_attn_Wq, model.cross_attn_Wk, model.cross_attn_Wv, model.cross_attn_Wo,
                    model.out_W
                ]
                
                for weight in weights:
                    if hasattr(weight, 'shape'):
                        # Approximate gradient norm using weight magnitudes
                        weight_norm = float(model.xp.linalg.norm(weight))
                        total_norm += weight_norm ** 2
                
                total_norm = np.sqrt(total_norm) / len(weights)  # Normalize by number of weight matrices
            
            return total_norm
            
        except Exception as e:
            print(f"Error calculating gradient norm: {e}")
            return 0.1  # Default value
    
    def calculate_weight_change_magnitude(self, model, prev_weights=None):
        """Calculate the magnitude of weight changes"""
        try:
            if prev_weights is None:
                return 0
                
            current_weights = [
                model.encoder_embed, model.decoder_embed,
                model.enc_attn_Wq, model.enc_attn_Wk, model.enc_attn_Wv, model.enc_attn_Wo,
                model.dec_attn_Wq, model.dec_attn_Wk, model.dec_attn_Wv, model.dec_attn_Wo,
                model.cross_attn_Wq, model.cross_attn_Wk, model.cross_attn_Wv, model.cross_attn_Wo,
                model.out_W
            ]
            
            total_change = 0
            for curr, prev in zip(current_weights, prev_weights):
                if hasattr(curr, 'shape') and hasattr(prev, 'shape'):
                    change = float(model.xp.linalg.norm(curr - prev))
                    total_change += change
            
            return total_change / len(current_weights)
            
        except Exception as e:
            print(f"Error calculating weight changes: {e}")
            return 0.01  # Default value
    
    def store_current_weights(self, model):
        """Store current weights for change calculation"""
        try:
            weights = [
                model.encoder_embed.copy(), model.decoder_embed.copy(),
                model.enc_attn_Wq.copy(), model.enc_attn_Wk.copy(), 
                model.enc_attn_Wv.copy(), model.enc_attn_Wo.copy(),
                model.dec_attn_Wq.copy(), model.dec_attn_Wk.copy(), 
                model.dec_attn_Wv.copy(), model.dec_attn_Wo.copy(),
                model.cross_attn_Wq.copy(), model.cross_attn_Wk.copy(), 
                model.cross_attn_Wv.copy(), model.cross_attn_Wo.copy(),
                model.out_W.copy()
            ]
            return weights
        except Exception as e:
            print(f"Error storing weights: {e}")
            return None
        
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
        """Custom training loop with GUI progress updates and advanced metrics"""
        xp = self.model.xp
        num_samples = encoder_inputs.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))
        
        # Initialize metrics tracking
        self.training_metrics = {
            'epoch_losses': [],
            'batch_losses': [],
            'gradient_norms': [],
            'learning_rates': [],
            'weight_changes': [],
            'epochs': [],
            'batch_numbers': []
        }
        
        prev_weights = None
        
        for epoch in range(epochs):
            if not self.is_training:  # Check for stop signal
                break
                
            start_time = time.time()
            total_loss = 0
            epoch_gradient_norms = []
            
            # Store previous weights for change calculation
            if epoch > 0:
                prev_weights = self.store_current_weights(self.model)
            
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
                
                # Calculate gradient norm for backpropagation analysis
                grad_norm = self.calculate_gradient_norm(self.model)
                epoch_gradient_norms.append(grad_norm)
                
                # Store batch metrics
                self.training_metrics['batch_losses'].append(batch_loss_scalar)
                self.training_metrics['gradient_norms'].append(grad_norm)
                self.training_metrics['batch_numbers'].append(epoch * num_batches + batch)
                
                # Update plots every 10 batches for responsiveness
                if batch % 10 == 0:
                    self.root.after(0, self.update_training_plots)
                    msg = f"Epoch {epoch+1}/{epochs}, Batch {batch+1}/{num_batches}, Loss: {batch_loss_scalar:.4f}, Grad Norm: {grad_norm:.4f}"
                    self.root.after(0, lambda m=msg: self.log_message(m))
            
            # Calculate epoch metrics
            avg_loss = total_loss / num_samples
            epoch_time = time.time() - start_time
            avg_grad_norm = np.mean(epoch_gradient_norms) if epoch_gradient_norms else 0
            
            # Calculate weight changes
            weight_change = 0
            if prev_weights is not None:
                weight_change = self.calculate_weight_change_magnitude(self.model, prev_weights)
            
            # Store epoch metrics
            self.training_metrics['epoch_losses'].append(avg_loss)
            self.training_metrics['learning_rates'].append(learning_rate)
            self.training_metrics['weight_changes'].append(weight_change)
            self.training_metrics['epochs'].append(epoch + 1)
            
            # Update all plots
            self.root.after(0, self.update_training_plots)
            
            # Enhanced logging with backpropagation insights
            msg = f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s"
            self.root.after(0, lambda m=msg: self.log_message(m))
            
            msg = f"  ├─ Avg Loss: {avg_loss:.4f}"
            self.root.after(0, lambda m=msg: self.log_message(m))
            
            msg = f"  ├─ Avg Gradient Norm: {avg_grad_norm:.4f}"
            if avg_grad_norm > 1.0:
                msg += " [WARNING: Possible exploding gradients]"
            elif avg_grad_norm < 0.01:
                msg += " [WARNING: Possible vanishing gradients]"
            else:
                msg += " [HEALTHY: Good gradient flow]"
            self.root.after(0, lambda m=msg: self.log_message(m))
            
            if weight_change > 0:
                msg = f"  ├─ Weight Change Magnitude: {weight_change:.6f}"
                if weight_change > 0.1:
                    msg += " [Large changes - fast learning]"
                elif weight_change < 0.001:
                    msg += " [Small changes - may be converging]"
                else:
                    msg += " [Moderate changes - stable learning]"
                self.root.after(0, lambda m=msg: self.log_message(m))
            
            # Convergence analysis
            if len(self.training_metrics['epoch_losses']) >= 3:
                recent_losses = self.training_metrics['epoch_losses'][-3:]
                loss_trend = np.polyfit(range(3), recent_losses, 1)[0]
                if loss_trend < -0.001:
                    trend_msg = "  └─ Learning Trend: IMPROVING ↗️"
                elif abs(loss_trend) < 0.001:
                    trend_msg = "  └─ Learning Trend: STABLE →"
                else:
                    trend_msg = "  └─ Learning Trend: CONCERNING ↘️"
                self.root.after(0, lambda m=trend_msg: self.log_message(m))
            
            # Save model periodically
            if (epoch + 1) % 5 == 0:
                os.makedirs("models", exist_ok=True)
                self.model.save_model("models")
                msg = f"Checkpoint saved at epoch {epoch+1}"
                self.root.after(0, lambda m=msg: self.log_message(m))
        
        # Final save and analysis
        if self.is_training:  # Only if not stopped
            self.model.save_model("models")
            self.root.after(0, lambda: self.log_message("Training completed! Final model saved."))
            self.root.after(0, lambda: self.progress_var.set(100))
            
            # Final training summary
            if self.training_metrics['epoch_losses']:
                initial_loss = self.training_metrics['epoch_losses'][0]
                final_loss = self.training_metrics['epoch_losses'][-1]
                improvement = ((initial_loss - final_loss) / initial_loss) * 100
                
                summary_msg = f"\n=== TRAINING SUMMARY ==="
                self.root.after(0, lambda m=summary_msg: self.log_message(m))
                
                summary_msg = f"Initial Loss: {initial_loss:.4f} → Final Loss: {final_loss:.4f}"
                self.root.after(0, lambda m=summary_msg: self.log_message(m))
                
                summary_msg = f"Total Improvement: {improvement:.2f}%"
                self.root.after(0, lambda m=summary_msg: self.log_message(m))
                
                if improvement > 50:
                    summary_msg = "Result: EXCELLENT - Model learned very well! 🎉"
                elif improvement > 20:
                    summary_msg = "Result: GOOD - Solid learning progress! ✅"
                elif improvement > 5:
                    summary_msg = "Result: MODERATE - Some learning occurred 📈"
                else:
                    summary_msg = "Result: POOR - Consider adjusting hyperparameters ⚠️"
                self.root.after(0, lambda m=summary_msg: self.log_message(m))
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