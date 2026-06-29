"""ESM2-150M-ReactionT5: fine-tune ReactionT5 to predict products, conditioned on
a **trainable** ESM-2 150M embedding of the enzyme sequence.

Unlike the frozen ESM2-650M sibling, ESM is part of the training graph and is
fine-tuned end-to-end with ReactionT5; there is no precomputed embedding cache.
"""
