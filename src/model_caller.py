"""
Model caller using LiteLLM for unified API access.
"""
import os
from typing import Optional, Dict, Any
import litellm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure LiteLLM
litellm.drop_params = True  # Drop unsupported params for different providers


class ModelCaller:
    """Unified interface for calling LLMs via LiteLLM."""

    def __init__(
        self,
        model: str,
        reasoning_effort: Optional[str] = None
    ):
        """
        Initialize the model caller.

        Args:
            model: Model identifier (e.g., 'gpt-4o', 'claude-3-5-sonnet-20241022')
            reasoning_effort: For reasoning models (o1, o3): 'low', 'medium', 'high'
        """
        self.model = model
        self.reasoning_effort = reasoning_effort

        # Determine provider from model name
        self.provider = self._detect_provider()

        # Validate API keys
        self._validate_api_keys()

    def _detect_provider(self) -> str:
        """Detect the API provider from the model name."""
        model_lower = self.model.lower()
        if 'gpt' in model_lower or 'o1' in model_lower or 'o3' in model_lower:
            return 'openai'
        elif 'claude' in model_lower:
            return 'anthropic'
        elif 'gemini' in model_lower:
            return 'google'
        else:
            return 'unknown'

    def _validate_api_keys(self):
        """Validate that necessary API keys are set."""
        if self.provider == 'openai':
            if not os.getenv('OPENAI_API_KEY'):
                raise ValueError("OPENAI_API_KEY not set in environment")
        elif self.provider == 'anthropic':
            if not os.getenv('ANTHROPIC_API_KEY'):
                raise ValueError("ANTHROPIC_API_KEY not set in environment")
        elif self.provider == 'google':
            if not os.getenv('GOOGLE_API_KEY'):
                raise ValueError("GOOGLE_API_KEY not set in environment")

    def call(
        self,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call the model with a prompt.

        Args:
            prompt: The user prompt
            **kwargs: Additional parameters to pass to the API

        Returns:
            Dictionary with 'content' and 'usage' keys
        """
        try:
            messages = [
                {'role': 'user', 'content': prompt}
            ]

            completion_kwargs = {
                'model': self.model,
                'messages': messages,
                **kwargs
            }

            # Add reasoning effort for o1/o3 models
            if self.reasoning_effort and ('gpt-5' in self.model.lower() or 'o1' in self.model.lower() or 'o3' in self.model.lower()):
                completion_kwargs['reasoning_effort'] = self.reasoning_effort

            response = litellm.completion(**completion_kwargs)

            return {
                'content': response.choices[0].message.content,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                'model': response.model,
                'finish_reason': response.choices[0].finish_reason
            }

        except Exception as e:
            print(f"Error calling model {self.model}: {str(e)}")
            raise


if __name__ == "__main__":
    # Example usage
    caller = ModelCaller(model='gpt-4o')

    result = caller.call(
        prompt="What is the capital of France?"
    )

    print(f"Response: {result['content']}")
    print(f"Tokens used: {result['usage']['total_tokens']}")
