🔍 Sequential Question Answering with Flan-T5

This project implements a multi-step, context-aware question answering system using Google’s Flan-T5 model.

The system processes three related questions sequentially and produces a final deterministic YES or NO output.


python template.py "<Question 1>" "<Question 2>" "<Question 3>"

ex: 
python template.py \
"Who is Rabindranath Tagore?" \
"Where was he born?" \
"Is it in India?"

