import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict
import os

# Swap model name to whichever fine-tuned checkpoint you have locally or on HuggingFace Hub
# e.g. "your-hf-username/nl-to-bash-qwen" or a local path like "./checkpoints/qwen-finetuned"
DEFAULT_MODEL = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-0.5B")


class BashTranslator:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._ready = False
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load tokenizer and model. Fails gracefully if model unavailable."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            self.model.eval()
            self._ready = True
            print(f"[INFO] Model '{self.model_name}' loaded successfully.")
        except Exception as e:
            print(f"[WARNING] Could not load model: {e}. Running in mock mode.")
            self._ready = False

    def is_ready(self) -> bool:
        return self._ready

    def translate(self, instruction: str, max_new_tokens: int = 128) -> Dict:
        """
        Translate a natural language instruction to a Bash command.
        Falls back to mock response if model not loaded (useful for testing).
        """
        if not self._ready:
            return self._mock_translate(instruction)

        prompt = f"### Instruction:\n{instruction}\n\n### Bash Command:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        bash_command = decoded.split("### Bash Command:")[-1].strip().split("\n")[0]
        confidence = self._compute_confidence(outputs, inputs)

        return {
            "bash_command": bash_command,
            "confidence": confidence
        }

    def _compute_confidence(self, outputs, inputs) -> float:
        """
        Approximate confidence score using output token count ratio.
        Replace with log-prob scoring for more accuracy.
        """
        input_len = inputs["input_ids"].shape[1]
        output_len = outputs.shape[1]
        generated_len = output_len - input_len
        # Heuristic: penalize very short or very long outputs
        score = min(1.0, generated_len / 20) * 0.9
        return round(score, 4)

    def _mock_translate(self, instruction: str) -> Dict:
        """Mock response for testing without a loaded model."""
        return {
            "bash_command": f"echo 'Mock output for: {instruction}'",
            "confidence": 0.0
        }
