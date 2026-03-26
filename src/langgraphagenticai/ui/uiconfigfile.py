from configparser import ConfigParser


class Config:
    def __init__(self, config_file="./src/langgraphagenticai/ui/uiconfigfile.ini"):
        self.config = ConfigParser()
        self.config.read(config_file)

    def get_llm_options(self):
        raw = self.config["DEFAULT"].get("LLM_OPTIONS", "")
        return [x.strip() for x in raw.split(",") if x.strip()]

    def get_usecase_options(self):
        raw = self.config["DEFAULT"].get("USECASE_OPTIONS", "")
        return [u.strip() for u in raw.split(",") if u.strip()]

    def get_groq_model_options(self):
        raw = self.config["DEFAULT"].get("GROQ_MODEL_OPTIONS", "")
        return [x.strip() for x in raw.split(",") if x.strip()]

    def get_page_title(self):
        return self.config["DEFAULT"].get("PAGE_TITLE", "")
