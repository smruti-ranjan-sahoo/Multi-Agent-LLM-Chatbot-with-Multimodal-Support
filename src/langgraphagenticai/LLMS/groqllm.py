from langchain_groq import ChatGroq

# Map decommissioned vision model IDs to current Llama-4 Vision models.
_DECOMMISSIONED = {
    "llama-3.2-11b-vision-preview": "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.2-90b-vision-preview": "meta-llama/llama-4-maverick-17b-128e-instruct",
}

def maybe_remap_model(name: str) -> str:
    key = (name or "").strip()
    return _DECOMMISSIONED.get(key, key)

class GroqLLM:
    def __init__(self, user_contols_input):
        self.user_input = user_contols_input

    def get_llm_model(self):
        api_key = self.user_input.get("GROQ_API_KEY")
        selected = self.user_input.get("selected_groq_model")
        selected = maybe_remap_model(selected)

        # IMPORTANT: ChatGroq expects `model=...`, not `model_name=...`
        llm = ChatGroq(
            api_key=api_key,
            model=selected,
            temperature=0.2,
            max_retries=2,
        )

        # Debug: confirm the exact Groq model being used
        print(f"[GroqLLM] Using Groq model: {getattr(llm, 'model', None)}")
        return llm