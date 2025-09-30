"""
Script to run all notebooks in sequence.
"""
import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter
import sys

def run_notebook(notebook_path):
    """Execute a notebook and save the output."""
    print(f"\n{'='*50}")
    print(f"Running {notebook_path}")
    print(f"{'='*50}")
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Configure the notebook execution
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    try:
        # Execute the notebook
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
        
        # Save the executed notebook
        output_path = notebook_path.replace('.ipynb', '.output.ipynb')
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        # Export to HTML for easier viewing
        html_exporter = HTMLExporter()
        html_output, _ = html_exporter.from_notebook_node(nb)
        html_path = notebook_path.replace('.ipynb', '.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_output)
            
        print(f"Successfully executed {notebook_path}")
        print(f"Output saved to {output_path}")
        print(f"HTML version saved to {html_path}")
        return True
        
    except Exception as e:
        print(f"Error executing {notebook_path}: {str(e)}")
        return False

def main():
    # List of notebooks to run in order
    notebooks = [
        'notebooks/1_Data_Preparation.ipynb',
        'notebooks/2_Feature_Engineering.ipynb',
        'notebooks/3_Model_Training.ipynb'
    ]
    
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Run each notebook
    for notebook in notebooks:
        if not os.path.exists(notebook):
            print(f"Error: {notebook} not found")
            continue
            
        success = run_notebook(notebook)
        if not success:
            print(f"Stopping execution due to error in {notebook}")
            sys.exit(1)
    
    print("\nAll notebooks executed successfully!")

if __name__ == "__main__":
    main()
