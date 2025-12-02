import dspy


LM_CONFIG = {
    "model": "openai/unsloth/Llama-3.2-3B-Instruct",
    "api_base": "http://127.0.0.1:8000/v1",
    "api_key": "local",
    "max_tokens": 1024, # short max tokens for testing
    "seed": 42,
}

lm=dspy.LM(
        **LM_CONFIG,
        cache=False # disable caching
    )

print(
    lm("Say this is a test!", temperature=0.7)
)
