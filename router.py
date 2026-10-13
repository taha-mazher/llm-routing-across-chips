"""Routing rules: RouteLLM's, and the same decision extended to any pool of (model, chip) backends."""
import numpy as np

CKPT = "routellm/bert_gpt4_augmented"  # RouteLLM's published BERT router, used unchanged


def embed(prompts):
    """For each prompt: RouteLLM's score P(strong model needed), and the encoder's mean-pooled features."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForSequenceClassification.from_pretrained(CKPT, num_labels=3).eval()
    scores, vecs = [], []
    with torch.no_grad():
        for i in range(0, len(prompts), 16):
            batch = tok(prompts[i:i + 16], return_tensors="pt", padding=True, truncation=True)
            out = model(**batch, output_hidden_states=True)
            # Same maths as RouteLLM's BERTRouter.calculate_strong_win_rate.
            scores += (1 - out.logits.softmax(-1)[:, -2:].sum(-1)).tolist()
            mask = batch["attention_mask"].unsqueeze(-1)
            vecs.append(((out.hidden_states[-1] * mask).sum(1) / mask.sum(1)).numpy())
    return np.array(scores), np.concatenate(vecs)


def routellm_quality(p, strength):
    """Expected quality per backend as RouteLLM sees it.

    The strongest backend (strength 1) always answers well, the weakest (strength 0)
    fails with probability p, and the rest sit in between.
    """
    return 1 - np.multiply.outer(p, 1 - np.asarray(strength))


def choose(quality, cost, latency, alpha, beta):
    """Index of the backend with the best expected quality minus weighted cost and latency.

    quality, cost and latency hold one value per backend, each on a 0..1 scale.
    With beta=0 and RouteLLM's quality for two backends this is exactly RouteLLM's
    rule: strong when p >= alpha * (cost_strong - cost_weak).
    """
    # ponytail: latency is one constant per backend, so it cannot react to a query
    # that needs a long answer. Predict output length per query if that matters.
    return np.argmax(np.asarray(quality) - alpha * np.asarray(cost) - beta * np.asarray(latency), axis=-1)
