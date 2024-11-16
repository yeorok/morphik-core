import os
from pathlib import Path

def should_ignore_directory(dirname):
    """
    Check if directory should be ignored.
    
    Args:
        dirname (str): Name of the directory
    
    Returns:
        bool: True if directory should be ignored, False otherwise
    """
    # List of directory names to ignore
    ignore_dirs = {
        'venv',
        'env',
        '.venv',
        'virtualenv',
        '__pycache__',
        '.pytest_cache',
        '.mypy_cache',
        '.tox'
    }
    return dirname in ignore_dirs

def aggregate_python_files(root_dir, output_file, script_name):
    """
    Recursively search through directories starting from root_dir,
    find all Python files, and write their contents to a single output file.
    Ignores virtual environment directories, __init__.py files and the script itself.
    
    Args:
        root_dir (str): The root directory to start the search from
        output_file (str): The name of the output file to create
        script_name (str): Name of this script to ignore
    """
    # Convert root_dir to absolute path
    root_dir = os.path.abspath(root_dir)
    
    # Use with statement to properly handle file opening/closing
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Walk through all directories
        for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True):
            # Modify dirnames in place to skip ignored directories
            dirnames[:] = [d for d in dirnames if not should_ignore_directory(d)]
            
            # Filter for Python files, excluding __init__.py and this script
            python_files = [
                f for f in filenames 
                if f.endswith('.py') 
                and f != '__init__.py'
                and f != script_name
                and f != output_file
            ]
            
            for py_file in python_files:
                # Get the full file path
                file_path = os.path.join(dirpath, py_file)
                # Get relative path from root_dir
                rel_path = os.path.relpath(file_path, root_dir)
                
                try:
                    # Read the content of the Python file
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                    
                    # Write the file path and contents to the output file
                    outfile.write(f"{rel_path}\n")
                    outfile.write(content)
                    outfile.write("\n\n" + "="*80 + "\n\n")  # Separator between files
                    
                except Exception as e:
                    outfile.write(f"Error reading {rel_path}: {str(e)}\n\n")

if __name__ == "__main__":
    # Get the name of this script
    script_name = os.path.basename(__file__)
    
    # Get current directory as default root
    current_dir = os.getcwd()
    
    # Output file name
    output_file = "python_files_contents.txt"
    
    print(f"Starting to process Python files from: {current_dir}")
    print(f"Ignoring {script_name}, all __init__.py files, and virtual environment directories")
    aggregate_python_files(current_dir, output_file, script_name)
    print(f"Finished! Results written to: {output_file}")
