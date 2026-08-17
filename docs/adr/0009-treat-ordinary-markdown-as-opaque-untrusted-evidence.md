# Treat ordinary Markdown as opaque untrusted evidence

Ordinary Markdown Sources must be non-empty UTF-8 and are indexed as opaque, untrusted evidence, including their frontmatter, HTML, and comments. Knowledge Forge does not execute includes, dereference links, or read linked local images, because doing so would let source content alter system behavior or expand the authorized Input Corpus; frontmatter becomes authoritative only after the document qualifies as an Imported Concept.
