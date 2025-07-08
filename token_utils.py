import tiktoken

# Use tiktoken for OpenAI-compatible token counting
tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4/ChatGPT tokenizer

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))
