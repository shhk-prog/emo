import torch
import numpy as np
from typing import Dict, List, Any, Optional

class InterventionEvaluator:
    """
    Evaluates the effect of interventions on the model's output.
    Implements a dual-evaluation system:
    1. Direct Logit Evaluation (fast, for patching/ablation sweeping)
    2. Generative Evaluation (free response, for ecological validity)
    """
    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        
        # Determine tokens for scale 1-9 for direct logit evaluation
        self.scale_tokens = []
        for i in range(1, 10):
            # Attempt to find the correct token ID for '1' to '9'
            # This depends heavily on the tokenizer. Usually it's just the string "1".
            tok_id = self.tokenizer.encode(str(i), add_special_tokens=False)
            if len(tok_id) > 0:
                self.scale_tokens.append((i, tok_id[0]))
                
    def evaluate_direct_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, float]:
        """
        Directly evaluates the probability distribution over the 1-9 scale tokens 
        at the next generated position.
        
        Args:
            input_ids: The prompt just before generating the number for V or A.
            attention_mask: Attention mask for the input.
            
        Returns:
            Dict containing expected_value and max_prob_value.
        """
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            next_token_logits = outputs.logits[0, -1, :]
            
        # Extract logits for scale tokens 1-9
        scale_logits = []
        scale_values = []
        for val, tok_id in self.scale_tokens:
            scale_logits.append(next_token_logits[tok_id].item())
            scale_values.append(val)
            
        scale_logits = torch.tensor(scale_logits)
        probs = torch.nn.functional.softmax(scale_logits, dim=0).numpy()
        
        expected_value = float(np.sum(probs * np.array(scale_values)))
        max_prob_idx = int(np.argmax(probs))
        max_prob_value = scale_values[max_prob_idx]
        
        return {
            "expected_value": expected_value,
            "max_prob_value": float(max_prob_value)
        }

    def evaluate_generative(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int = 50) -> str:
        """
        Generates free text response after intervention.
        """
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False, # greedy decoding for deterministic evaluation
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        # Extract only the generated part
        generated_ids = output_ids[0][len(input_ids[0]):]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()
