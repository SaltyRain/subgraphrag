CONTEXT_FROM_SUBGRAPH_PROMPT = """You are an expert knowledge-graph interpreter and fact-based writer.

Write a concise, factual narrative that summarizes the subgraph. Focus on the HEAD entity and facts directly connected to it; include multi-hop connections only when they clarify the HEAD. Do not use any knowledge outside the subgraph.

Requirements:
1) Describe only what is present in EDGES (and relevant PATHS), preserving subject→predicate→object direction.
2) Prioritize facts where the subject is the HEAD. Mention other entities only to explain relationships to the HEAD.
3) Group related facts (e.g., languages, political relations, historical changes). Merge duplicates or near-duplicates; do not repeat the same fact.
4) If any subject/predicate/object equals "unknown", state that the information is unspecified.
5) Avoid meta commentary about graphs. Use the given labels verbatim; prefer nouns over pronouns to avoid ambiguity.
6) Keep it brief and readable: 3–6 sentences, max ~120–150 words. Neutral, encyclopedic tone.

Input Subgraph:
HEAD: {head}

NODES:
{nodes}

EDGES (subject, predicate, object):
{edges}

PATHS (if relevant):
{paths}

Output:
A compact paragraph (or two) in fluent English that integrates the facts into a cohesive narrative centered on the HEAD. Do not add external facts.
"""