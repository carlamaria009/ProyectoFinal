import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
from gui_components import MLModelApp

def main():
    """Main function to run the ML Model Loader application"""
    try:
        # Create the main window
        root = tk.Tk()
        
        # Set window properties
        root.title("ML Model Loader - PKL Executor")
        root.geometry("1000x700")
        root.minsize(800, 600)
        
        # Center the window
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (1000 // 2)
        y = (root.winfo_screenheight() // 2) - (700 // 2)
        root.geometry(f"1000x700+{x}+{y}")
        
        # Create and run the application
        app = MLModelApp(root)
        
        # Start the main loop
        root.mainloop()
        
    except Exception as e:
        messagebox.showerror("Application Error", f"Failed to start application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()