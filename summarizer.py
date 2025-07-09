# summarizer.py

import torch


def summarize_text(text, tokenizer, model, encoder_max_len=4096, decoder_max_len=512):
    """
    Summarizes legal text using a pretrained LED model with safer limits for memory.
    """
    # Limit very long texts to avoid freezing
    text = text[:5000]  # character limit ~1000–2000 tokens

    # Tokenize input
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=encoder_max_len,
        return_tensors="pt",
    )

    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    # Set global attention on first token (required for LED)
    global_attention_mask = torch.zeros_like(attention_mask)
    global_attention_mask[:, 0] = 1

    print("📏 Tokenized input shape:", input_ids.shape)

    # Generate summary
    summary_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        global_attention_mask=global_attention_mask,
        num_beams=2,
        max_length=decoder_max_len,
        min_length=128,
        no_repeat_ngram_size=4,
        early_stopping=True,
        repetition_penalty=2.5
    )

    # Decode and return summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
    return summary