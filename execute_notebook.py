"""
Executes the notebook top-to-bottom and saves executed cell outputs.
"""

import nbformat
from nbclient import NotebookClient

notebook_path = "notebooks/en_de_seq2seq_attention.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb, timeout=600, kernel_name="python3")
print("[*] Executing notebook...")
client.execute()

with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("[*] Notebook successfully executed and saved with all outputs intact!")
