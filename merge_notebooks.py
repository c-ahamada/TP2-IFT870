import nbformat as nbf
def merge_notebooks(filenames):
    merged = nbf.v4.new_notebook()
    for fname in filenames:
        with open(fname) as f:
            nb = nbf.read(f, as_version=4)
            merged.cells.extend(nb.cells)
    return merged

notebooks_to_merge = ["Groupe11_tp2.ipynb", "RNN.ipynb", "caro_notebook.ipynb"]
merged_notebook = merge_notebooks(notebooks_to_merge)
with open("merged_notebook.ipynb", "w") as f:
    nbf.write(merged_notebook, f)