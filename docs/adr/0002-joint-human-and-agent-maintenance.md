# Joint human and agent maintenance

The OKF Bundle is a shared write model rather than a disposable generated artifact. Knowledge Forge stores the previous agent output as an explicit baseline and performs deterministic three-way merging at Markdown heading-section and frontmatter-key granularity; non-overlapping human edits survive automatically, while overlapping changes fail atomically and require an auditable reconciliation decision. This costs private state and merge complexity but prevents later source updates from silently overwriting authoritative human curation.
