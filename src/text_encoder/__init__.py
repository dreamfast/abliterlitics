# Text-encoder conditioning fidelity analysis (tebench).
#
# Measures how much an abliterated/quantized text encoder perturbs the
# conditioning tensor that enters a diffusion transformer's cross-attention.
# Answers the Reddit PSA debate: "does a heretic text encoder help or hurt?"

TE_RESULTS_VERSION = 1
