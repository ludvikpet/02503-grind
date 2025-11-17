import jupytext
from glob import glob 
import os 

for file in glob("**/readme.md"):
    parent = file.split("/")[0]
    outdir = f"{parent}/src"
    os.makedirs(outdir,exist_ok=True)
    notebook = jupytext.read(file)
    output_name = file.rsplit("-",1)[0].lower()
    output_file = f"{outdir}/{output_name}"
    jupytext.write(notebook, output_file+".ipynb")
    jupytext.write(notebook, output_file+".py",fmt='py:percent')
    print(f"Converted {file} → {output_file}.ipynb and .py")

