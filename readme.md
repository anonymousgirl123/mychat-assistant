# Sequential Question Answering with Flan-T5 
Abstract

This project implements a sequential, context-aware question answering pipeline using the Flan-T5 language model. The system processes three related queries in a chained manner, where each generated response serves as contextual input for the subsequent query. The final output is a strictly deterministic binary response (YES or NO).

## Methodology

The pipeline operates in three stages:

Stage 1: Generate an answer to Question 1 using Flan-T5.

Stage 2: Generate an answer to Question 2 conditioned on the output of Stage 1 (one-shot contextual prompting).

Stage 3: Produce a deterministic binary classification (YES or NO) for Question 3 using the output of Stage 2 as context.

Generation randomness is disabled to ensure reproducibility.

### Model

Model: google/flan-t5-xl

Framework: PyTorch

Library: HuggingFace Transformers

### Execution
python template.py "<Question 1>" "<Question 2>" "<Question 3>"

Example:

python template.py \
"Who is Rabindranath Tagore?" \
"Where was he born?" \
"Is it in India?"

Output:

YES
Requirements

Python 3.8+

torch

transformers

sentencepiece

### Reproducibility

torch.manual_seed(42) is used for deterministic behavior.

Sampling is disabled (do_sample=False).

Output is strictly constrained to uppercase YES or NO.


python template.py "<Question 1>" "<Question 2>" "<Question 3>"

ex: 
python template.py \
"Who is Rabindranath Tagore?" \
"Where was he born?" \
"Is it in India?"

