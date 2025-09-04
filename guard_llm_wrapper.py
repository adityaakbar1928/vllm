from typing import Any, List, Optional
from langchain.llms.base import LLM

class VLLMAsLangChain(LLM):
    def __init__(self, core_llm, sampling_params=None):
        super().__init__()
        self.core_llm = core_llm
        self.sampling_params = sampling_params

    @property
    def _llm_type(self) -> str:
        return "vllm-local"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        out = self.core_llm.generate(prompt, self.sampling_params)
        text = out[0].outputs[0].text
        if stop:
            for s in stop:
                if s in text:
                    text = text.split(s)[0]
        return text

    @property
    def _identifying_params(self) -> dict:
        return {"engine": "vllm"}
