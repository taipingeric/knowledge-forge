# Use structural Markdown provenance

Markdown evidence is identified by Source Identity and document SHA-256 plus a parser-derived heading path, same-level occurrence, and structural-block hash; content before the first heading belongs to a synthetic root block, and line ranges are retained only as human navigation hints. Unlike PDF pages, line numbers are not durable evidence locators because unrelated edits shift them, while structural parsing avoids false headings inside constructs such as code fences and the block hash makes changed evidence detectable.
