# TinyGPT Studies

A learn-by-building record of the math and Python experiments leading toward a small GPT-style language model. Notes are Obsidian-compatible Markdown; notebooks run in Jupyter.

**Study progress:** Stages 1–5 completed. Stage 6 is next. File numbers represent *document topics*, not separate study stages: documents 05 and 06 both cover Stage 5.

## Study index

| Topic | Note | Notebook |
| --- | --- | --- |
| Derivatives and finite differences | [01 — Derivatives](notes/01-derivatives.md) | — |
| Gradient descent and partial derivatives | [02 — Gradient descent](notes/02-gradient-descent.md) | — |
| Chain rule and backpropagation | [03 — Chain rule](notes/03-chain-rule-backpropagation.md) | — |
| Implementing gradient descent in Python | [04 — Python gradient descent](notes/04-gradient-descent-python.md) | [Gradient descent](notebooks/04-gradient-descent.ipynb) |
| Training with multiple examples | [05 — Multiple examples](notes/05-training-multiple-examples.md) | [Multiple examples](notebooks/05-training-multiple-examples.ipynb) |
| Mini-batch gradient descent | [06 — Mini-batches](notes/06-mini-batch-gradient-descent.md) | [Mini-batch training](notebooks/05-mini-batch-training.ipynb) |

## Repository layout

```text
tinygpt/
├── README.md
├── notes/           # Obsidian learning notes and worked exercises
├── notebooks/       # Python and Jupyter experiments
├── assets/images/   # Diagrams used in the notes
└── .obsidian/        # Obsidian configuration
```

## Run the notebooks

Open the repository in Jupyter Notebook, JupyterLab, or VS Code with a Python 3 kernel. Run the cells from top to bottom. Some cells build on earlier variables. Matplotlib is needed for plots.

The mini-batch notebook prints *full-dataset MSE after each epoch*, which is different from each mini-batch's loss calculated before its update. The first epoch should give approximately `w = 2.27`, `b = 0.795`, and MSE `2.252025`.

These notes preserve the original manual calculations and learning process, with corrected mathematical notation. They are study material, not production ML code.
