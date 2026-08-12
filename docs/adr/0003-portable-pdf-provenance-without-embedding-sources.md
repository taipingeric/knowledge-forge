# Portable PDF provenance without embedding sources

PDF Sources are not copied into the OKF Bundle. Each source is represented by a stable logical URN derived from its source-root-relative identity, while its SHA-256 content version and normalized 1-based page ranges are recorded directly in Concept provenance. This keeps the Bundle portable across repositories without persisting source binaries or runtime filesystem paths, at the cost of requiring the source corpus separately for full evidence verification.
