import re

MAX_QUERY_LENGTH = 500

PROMPT_INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"system prompt",
    r"act as",
    r"jailbreak",
]

def validate_query(query: str) -> str:
    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError("Query too long.")

    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            raise ValueError("Prompt injection detected.")

    return query.strip()