from typing import Type, TypeVar, Any
from pydantic import BaseModel
from ollama import chat


# Placeholder type for the different schemas
T = TypeVar("T", bound=BaseModel)


def call_ollama(*, model: str, system_prompt: str, user_prompt: str, schema: Type[T]) -> T:

    # Parse the json schema and make sure that no extra fields can be added
    json_schema: dict[str, Any] = schema.model_json_schema()
    json_schema["additionalProperties"] = False

    # LLM call
    response = chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        format=json_schema,
        stream=False,
    )

    # Get reply and validate on schema
    content: str = response["message"]["content"]
    output = schema.model_validate_json(content)
    
    return output