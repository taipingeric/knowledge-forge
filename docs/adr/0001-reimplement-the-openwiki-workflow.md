# Reimplement the OpenWiki workflow

We will preserve OpenWiki's core progression from deterministic ingestion through agent synthesis and validation to a knowledge bundle, but implement a focused workflow directly with LangGraph and LangChain instead of forking or wrapping OpenWiki. OpenWiki currently targets OKF 0.1 and carries broader connector, interface, provider, and visualization concerns; a clean implementation lets the product target OKF 0.2 without inheriting that coupling or relying on a lossy version-conversion layer.
