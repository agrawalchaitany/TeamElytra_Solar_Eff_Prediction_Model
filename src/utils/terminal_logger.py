import os
import sys
from datetime import datetime
from pathlib import Path

class TerminalLogger:
    """
    Simple logger that captures all terminal output during model training
    and saves it to a timestamped file.
    """
    
    def __init__(self, log_dir=None):
        """
        Initialize the terminal logger
        
        Parameters:
        -----------
        log_dir : str, optional
            Directory to save log files. If None, creates 'logs' in project root.
        """
        # Set up log directory
        base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
        self.log_dir = log_dir if log_dir else os.path.join(base_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create timestamped log file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = os.path.join(self.log_dir, f"training_log_{timestamp}.txt")
        
        # Store original stdout and stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Open log file
        self.log_file = None
    
    def start(self):
        """Start capturing terminal output"""
        # Open log file for writing
        self.log_file = open(self.log_file_path, 'w')
        
        # Create a custom writer that writes to both console and file
        class DualWriter:
            def __init__(self, file, original_stream):
                self.file = file
                self.original_stream = original_stream
            
            def write(self, text):
                self.original_stream.write(text)
                self.file.write(text)
                self.file.flush()  # Ensure immediate writing
            
            def flush(self):
                self.original_stream.flush()
                self.file.flush()
        
        # Redirect stdout and stderr to our custom writer
        sys.stdout = DualWriter(self.log_file, self.original_stdout)
        sys.stderr = DualWriter(self.log_file, self.original_stderr)
        
        print(f"[{datetime.now()}] Terminal logging started - Saving to: {self.log_file_path}")
        return self
    
    def stop(self):
        """Stop capturing terminal output"""
        if self.log_file:
            print(f"[{datetime.now()}] Terminal logging stopped")
            
            # Restore original stdout and stderr
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            
            # Close log file
            self.log_file.close()
            self.log_file = None
            
            print(f"Training log saved to: {self.log_file_path}")
    
    def __enter__(self):
        """Support for 'with' statement"""
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support for 'with' statement"""
        self.stop()
        # Don't suppress exceptions
        return False

# Example usage
if __name__ == "__main__":
    with TerminalLogger() as logger:
        print("This is a test message")
        print("All of this output is being captured")
        print("Including any errors or warnings")