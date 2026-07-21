# Third-party notices

ChunkFlow is derived from Physical Intelligence's
[openpi](https://github.com/Physical-Intelligence/openpi) and retains its
Apache-2.0 license and source attribution. The following embedded or adapted
files carry additional notices in their file headers. This summary does not
replace those headers or the repository [LICENSE](LICENSE).

## Big Vision Authors

The following JAX model adaptations retain `Copyright 2024 Big Vision Authors`
headers and are licensed under Apache-2.0:

- `src/openpi/models/gemma.py`
- `src/openpi/models/gemma_fast.py`
- `src/openpi/models/siglip.py`

## Google LLC

`src/openpi/models/vit.py` retains its `Copyright 2024 Google LLC` header and is
licensed under Apache-2.0. The file identifies its adaptation source in the
header.

## Hugging Face and Google transformer replacements

Files below `src/openpi/models_pytorch/transformers_replace/` are replacements
adapted from Hugging Face Transformers sources. Their file headers attribute,
as applicable, the Hugging Face team, Google Inc., Google AI, and The HuggingFace
Team. They are licensed under Apache-2.0:

- `src/openpi/models_pytorch/transformers_replace/models/gemma/configuration_gemma.py`
- `src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py`
- `src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py`
- `src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py`

When copying or modifying any of these files, retain their original file headers,
attribution statements, generation warnings, and Apache-2.0 notices. Attribution
names above reproduce the source headers; this document makes no additional
ownership claim.
