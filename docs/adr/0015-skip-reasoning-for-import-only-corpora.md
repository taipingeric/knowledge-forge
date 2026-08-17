# Skip reasoning for import-only corpora

An Input Corpus containing only Imported Concepts is published through a deterministic zero-model path and does not require model credentials. For a mixed corpus, planning receives the imported manifest and content as canonical existing knowledge, creates only missing Concepts from ordinary PDF and Markdown evidence, and validates model configuration only when reasoning is actually required; this avoids needless re-synthesis, provider coupling, and model cost while still supporting cross-format planning.
