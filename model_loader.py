# model_loader.py
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_model(model_name="nsi319/legal-led-base-16384"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        gradient_checkpointing=True,
        use_cache=False
    )
    return tokenizer, model