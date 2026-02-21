"""
Sequential Question Answering using Flan-T5
Final output is strictly YES or NO (uppercase, no extra text).
"""

import sys
import torch
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

# Suppress transformer warnings (required by assignment)
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()


def llm_function(model, tokenizer, questions):
    """
    1. Generate answer for Question 1.
    2. Generate answer for Question 2 using Answer 1 as context.
    3. Generate deterministic YES/NO for Question 3 using Answer 2 as context.
    4. Return strictly 'YES' or 'NO'.
    """

    q1, q2, q3 = questions

    # ---------------- Step 1 ----------------
    input_ids = tokenizer(q1, return_tensors="pt").input_ids
    output_ids = model.generate(
        input_ids,
        max_new_tokens=64,
        do_sample=False
    )
    answer1 = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    # ---------------- Step 2 ----------------
    prompt_q2 = (
        f"Question: {q1}\n"
        f"Answer: {answer1}\n\n"
        f"Question: {q2}\n"
        f"Answer:"
    )

    input_ids = tokenizer(prompt_q2, return_tensors="pt").input_ids
    output_ids = model.generate(
        input_ids,
        max_new_tokens=64,
        do_sample=False
    )
    answer2 = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    # ---------------- Step 3 (Deterministic YES/NO) ----------------
    prompt_q3 = (
        f"Context: {answer2}\n"
        f"Question: {q3}\n"
        f"Answer only YES or NO."
    )

    input_ids = tokenizer(prompt_q3, return_tensors="pt").input_ids
    output_ids = model.generate(
        input_ids,
        max_new_tokens=3,
        do_sample=False
    )

    answer3 = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip().upper()

    # Strict enforcement
    if "YES" in answer3:
        final_output = "YES"
    else:
        final_output = "NO"

    return final_output


if __name__ == '__main__':

    question_a = sys.argv[1].strip()
    question_b = sys.argv[2].strip()
    question_c = sys.argv[3].strip()

    questions = [question_a, question_b, question_c]

    # Load Model and Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")

    torch.manual_seed(42)

    result = llm_function(model, tokenizer, questions)

    # STRICT OUTPUT (no extra text)
    print(result.strip())