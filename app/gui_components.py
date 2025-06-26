import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
from model_loader import ModelLoader
from data_processor import DataProcessor


class MLModelApp:
    """Main application class for the ML Model Loader GUI"""
    
    def __init__(self, root):
        self.root = root
        self.model_loader = ModelLoader()
        
        # Configure styles
        self.setup_styles()
        
        # Create GUI components
        self.create_widgets()
        
        # Initialize state
        self.update_ui_state()
    
    def setup_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        
        # Configure button styles
        style.configure('Action.TButton', padding=(10, 5))
        style.configure('Success.TLabel', foreground='green')
        style.configure('Error.TLabel', foreground='red')
        style.configure('Info.TLabel', foreground='blue')
    
    def create_widgets(self):
        """Create and layout all GUI widgets"""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="ML Model Loader - PKL Executor", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Model loading section
        self.create_model_section(main_frame)
        
        # Model information section
        self.create_info_section(main_frame)
        
        # Input section
        self.create_input_section(main_frame)
        
        # Results section
        self.create_results_section(main_frame)
        
        # Status bar
        self.create_status_bar(main_frame)
    
    def create_model_section(self, parent):
        """Create model loading section"""
        # Model file selection
        model_frame = ttk.LabelFrame(parent, text="Model Selection", padding="10")
        model_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        model_frame.columnconfigure(1, weight=1)
        
        ttk.Label(model_frame, text="Model File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        
        self.model_path_var = tk.StringVar()
        self.model_path_entry = ttk.Entry(model_frame, textvariable=self.model_path_var, state='readonly')
        self.model_path_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        self.browse_button = ttk.Button(model_frame, text="Browse", command=self.browse_model_file,
                                       style='Action.TButton')
        self.browse_button.grid(row=0, column=2, padx=(0, 10))
        
        self.load_button = ttk.Button(model_frame, text="Load Model", command=self.load_model,
                                     style='Action.TButton', state='disabled')
        self.load_button.grid(row=0, column=3)
        
        self.clear_button = ttk.Button(model_frame, text="Clear", command=self.clear_model,
                                      style='Action.TButton', state='disabled')
        self.clear_button.grid(row=0, column=4, padx=(10, 0))
    
    def create_info_section(self, parent):
        """Create model information section"""
        info_frame = ttk.LabelFrame(parent, text="Model Information", padding="10")
        info_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        info_frame.columnconfigure(0, weight=1)
        
        # Create treeview for model info
        self.info_tree = ttk.Treeview(info_frame, columns=('Value',), show='tree headings', height=4)
        self.info_tree.heading('#0', text='Property')
        self.info_tree.heading('Value', text='Value')
        self.info_tree.column('#0', width=200, minwidth=150)
        self.info_tree.column('Value', width=300, minwidth=200)
        
        # Scrollbar for info tree
        info_scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_tree.yview)
        self.info_tree.configure(yscrollcommand=info_scrollbar.set)
        
        self.info_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        info_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
    
    def create_input_section(self, parent):
        """Create input data section"""
        input_frame = ttk.LabelFrame(parent, text="Input Data", padding="10")
        input_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(0, weight=1)
        
        # Input text area
        self.input_text = scrolledtext.ScrolledText(input_frame, height=6, wrap=tk.WORD)
        self.input_text.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Buttons frame
        buttons_frame = ttk.Frame(input_frame)
        buttons_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        self.predict_button = ttk.Button(buttons_frame, text="Predict", command=self.make_prediction,
                                        style='Action.TButton', state='disabled')
        self.predict_button.grid(row=0, column=0, padx=(0, 10))
        
        self.predict_proba_button = ttk.Button(buttons_frame, text="Predict Probabilities", 
                                              command=self.make_probability_prediction,
                                              style='Action.TButton', state='disabled')
        self.predict_proba_button.grid(row=0, column=1, padx=(0, 10))
        
        self.clear_input_button = ttk.Button(buttons_frame, text="Clear Input", 
                                            command=self.clear_input)
        self.clear_input_button.grid(row=0, column=2, padx=(0, 10))
        
        self.examples_button = ttk.Button(buttons_frame, text="Show Examples", 
                                         command=self.show_examples)
        self.examples_button.grid(row=0, column=3)
    
    def create_results_section(self, parent):
        """Create results display section"""
        results_frame = ttk.LabelFrame(parent, text="Results", padding="10")
        results_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Create treeview for results
        self.results_tree = ttk.Treeview(results_frame, show='headings')
        
        # Scrollbars for results tree
        v_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        h_scrollbar = ttk.Scrollbar(results_frame, orient=tk.HORIZONTAL, command=self.results_tree.xview)
        
        self.results_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.results_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Export button
        self.export_button = ttk.Button(results_frame, text="Export Results", 
                                       command=self.export_results, state='disabled')
        self.export_button.grid(row=2, column=0, pady=(10, 0), sticky=tk.W)
    
    def create_status_bar(self, parent):
        """Create status bar"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        status_frame.columnconfigure(0, weight=1)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Select a PKL model file to begin")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                     relief=tk.SUNKEN, padding="5")
        self.status_label.grid(row=0, column=0, sticky=(tk.W, tk.E))
    
    def browse_model_file(self):
        """Open file dialog to select model file"""
        file_path = filedialog.askopenfilename(
            title="Select PKL Model File",
            filetypes=[
                ("Pickle files", "*.pkl"),
                ("Pickle files", "*.pickle"),
                ("Joblib files", "*.joblib"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.model_path_var.set(file_path)
            self.load_button.config(state='normal')
            self.status_var.set(f"Selected: {os.path.basename(file_path)}")
    
    def load_model(self):
        """Load the selected model"""
        model_path = self.model_path_var.get()
        
        if not model_path:
            messagebox.showerror("Error", "Please select a model file first")
            return
        
        self.status_var.set("Loading model...")
        self.root.update()
        
        success, message = self.model_loader.load_model(model_path)
        
        if success:
            self.status_var.set(f"Success: {message}")
            self.update_model_info()
            messagebox.showinfo("Success", message)
        else:
            self.status_var.set(f"Error: {message}")
            messagebox.showerror("Error", message)
        
        self.update_ui_state()
    
    def clear_model(self):
        """Clear the loaded model"""
        if messagebox.askyesno("Confirm", "Clear the loaded model?"):
            self.model_loader.clear_model()
            self.model_path_var.set("")
            self.clear_model_info()
            self.clear_results()
            self.status_var.set("Model cleared - Select a new model file")
            self.update_ui_state()
    
    def update_model_info(self):
        """Update the model information display"""
        # Clear existing items
        for item in self.info_tree.get_children():
            self.info_tree.delete(item)
        
        if not self.model_loader.is_model_loaded():
            return
        
        model_info = self.model_loader.get_model_info()
        
        for key, value in model_info.items():
            if key == 'feature_names' and isinstance(value, list):
                # Create a parent node for features
                features_node = self.info_tree.insert('', 'end', text='Feature Names', values=('',))
                for i, feature in enumerate(value):
                    self.info_tree.insert(features_node, 'end', text=f'  {i+1}', values=(feature,))
            else:
                self.info_tree.insert('', 'end', text=key.replace('_', ' ').title(), values=(str(value),))
    
    def clear_model_info(self):
        """Clear the model information display"""
        for item in self.info_tree.get_children():
            self.info_tree.delete(item)
    
    def make_prediction(self):
        """Make prediction using the loaded model"""
        if not self.model_loader.is_model_loaded():
            messagebox.showerror("Error", "No model loaded")
            return
        
        input_text = self.input_text.get("1.0", tk.END).strip()
        if not input_text:
            messagebox.showerror("Error", "Please enter input data")
            return
        
        self.status_var.set("Processing input data...")
        self.root.update()
        
        # Get expected features if available
        model_info = self.model_loader.get_model_info()
        expected_features = model_info.get('feature_names')
        
        # Parse input data
        success, input_data, message = DataProcessor.parse_input_data(input_text, expected_features)
        
        if not success:
            self.status_var.set(f"Input Error: {message}")
            messagebox.showerror("Input Error", message)
            return
        
        self.status_var.set("Making prediction...")
        self.root.update()
        
        # Make prediction
        success, predictions, message = self.model_loader.predict(input_data)
        
        if success:
            self.status_var.set("Prediction completed successfully")
            self.display_results(predictions, None)
            messagebox.showinfo("Success", "Prediction completed successfully")
        else:
            self.status_var.set(f"Prediction Error: {message}")
            messagebox.showerror("Prediction Error", message)
    
    def make_probability_prediction(self):
        """Make probability prediction using the loaded model"""
        if not self.model_loader.is_model_loaded():
            messagebox.showerror("Error", "No model loaded")
            return
        
        input_text = self.input_text.get("1.0", tk.END).strip()
        if not input_text:
            messagebox.showerror("Error", "Please enter input data")
            return
        
        self.status_var.set("Processing input data...")
        self.root.update()
        
        # Get expected features if available
        model_info = self.model_loader.get_model_info()
        expected_features = model_info.get('feature_names')
        
        # Parse input data
        success, input_data, message = DataProcessor.parse_input_data(input_text, expected_features)
        
        if not success:
            self.status_var.set(f"Input Error: {message}")
            messagebox.showerror("Input Error", message)
            return
        
        self.status_var.set("Making probability prediction...")
        self.root.update()
        
        # Make predictions
        pred_success, predictions, pred_message = self.model_loader.predict(input_data)
        prob_success, probabilities, prob_message = self.model_loader.predict_proba(input_data)
        
        if pred_success:
            if prob_success:
                self.status_var.set("Probability prediction completed successfully")
                self.display_results(predictions, probabilities)
                messagebox.showinfo("Success", "Probability prediction completed successfully")
            else:
                self.status_var.set("Regular prediction completed (probabilities not available)")
                self.display_results(predictions, None)
                messagebox.showwarning("Partial Success", 
                                     f"Regular prediction completed, but {prob_message}")
        else:
            self.status_var.set(f"Prediction Error: {pred_message}")
            messagebox.showerror("Prediction Error", pred_message)
    
    def display_results(self, predictions, probabilities=None):
        """Display prediction results in the table"""
        # Clear existing results
        self.clear_results()
        
        # Format data for display
        formatted_data = DataProcessor.format_predictions_for_display(predictions, probabilities)
        
        if not formatted_data:
            return
        
        # Set up columns
        columns = list(formatted_data[0].keys())
        self.results_tree['columns'] = columns
        
        # Configure column headings and widths
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=120, minwidth=80)
        
        # Insert data
        for row_data in formatted_data:
            values = [str(row_data[col]) for col in columns]
            self.results_tree.insert('', 'end', values=values)
        
        # Enable export button
        self.export_button.config(state='normal')
    
    def clear_results(self):
        """Clear the results table"""
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.export_button.config(state='disabled')
    
    def clear_input(self):
        """Clear the input text area"""
        self.input_text.delete("1.0", tk.END)
    
    def show_examples(self):
        """Show input format examples"""
        examples = DataProcessor.get_example_formats()
        
        # Create example window
        example_window = tk.Toplevel(self.root)
        example_window.title("Input Format Examples")
        example_window.geometry("600x400")
        example_window.transient(self.root)
        example_window.grab_set()
        
        # Center the window
        example_window.update_idletasks()
        x = (example_window.winfo_screenwidth() // 2) - (600 // 2)
        y = (example_window.winfo_screenheight() // 2) - (400 // 2)
        example_window.geometry(f"600x400+{x}+{y}")
        
        # Create text widget with examples
        text_widget = scrolledtext.ScrolledText(example_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget.insert("1.0", examples)
        text_widget.config(state='disabled')
        
        # Close button
        close_button = ttk.Button(example_window, text="Close", 
                                 command=example_window.destroy)
        close_button.pack(pady=(0, 10))
    
    def export_results(self):
        """Export results to CSV file"""
        if not self.results_tree.get_children():
            messagebox.showwarning("Warning", "No results to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            import csv
            
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                # Get column names
                columns = self.results_tree['columns']
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow(columns)
                
                # Write data
                for item in self.results_tree.get_children():
                    values = self.results_tree.item(item)['values']
                    writer.writerow(values)
            
            self.status_var.set(f"Results exported to: {os.path.basename(file_path)}")
            messagebox.showinfo("Success", f"Results exported successfully to:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results:\n{str(e)}")
    
    def update_ui_state(self):
        """Update UI state based on current model status"""
        model_loaded = self.model_loader.is_model_loaded()
        
        # Update button states
        self.load_button.config(state='normal' if self.model_path_var.get() else 'disabled')
        self.clear_button.config(state='normal' if model_loaded else 'disabled')
        self.predict_button.config(state='normal' if model_loaded else 'disabled')
        self.predict_proba_button.config(state='normal' if model_loaded else 'disabled')
