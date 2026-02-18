"""
LLM Prompt Engineering Assignment: Clinical Entity Extraction

Complete the functions below to extract structured data from clinical notes
using LLM APIs.
"""

import json
import os
from typing import Optional


def get_client():
    """
    Initialize the LLM client based on available API keys.

    Checks for OpenRouter first (preferred), then falls back to OpenAI.
    Both use the same openai SDK.

    Returns
    -------
    tuple
        (client, provider) where provider is 'openrouter' or 'openai'
    """
    from openai import OpenAI

    # Check for OpenRouter API key (preferred)
    if os.environ.get("OPENROUTER_API_KEY"):
        client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
        return client, "openrouter"

    # Fallback to OpenAI API key
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAI(), "openai"

    raise ValueError(
        "No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY"
    )


def build_prompt(note: str, few_shot: bool = False) -> str:
    """
    Build a prompt for clinical entity extraction.

    Parameters
    ----------
    note : str
        The clinical note to process
    few_shot : bool
        Whether to include few-shot examples

    Returns
    -------
    str
        The complete prompt
    """
    # TODO: Implement your prompt here
    #
    # Your prompt should:
    # 1. Describe the task clearly
    # 2. Specify the output format (JSON schema)
    # 3. Optionally include few-shot examples
    #
    # Example structure:
    # prompt = f"""
    # [TASK DESCRIPTION]
    #
    # [OUTPUT FORMAT]
    # {{
    #   "diagnosis": "...",
    #   "medications": [...],
    #   "lab_values": {{...}},
    #   "confidence": 0.0
    # }}
    #
    # [FEW-SHOT EXAMPLES if few_shot=True]
    #
    # [INPUT]
    # {note}
    # """

    pass


def call_llm(prompt: str, provider: str, client=None) -> str:
    """
    Call the LLM API with the given prompt.

    Both OpenRouter and OpenAI use the same openai SDK interface.

    Parameters
    ----------
    prompt : str
        The prompt to send
    provider : str
        'openrouter' or 'openai'
    client : optional
        The OpenAI-compatible client

    Returns
    -------
    str
        The raw response text
    """
    # Select model name based on provider
    model = "openai/gpt-4o-mini" if provider == "openrouter" else "gpt-4o-mini"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a medical information extraction assistant.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=500,
    )
    return response.choices[0].message.content


def extract_entities(note: str, few_shot: bool = False) -> dict:
    """
    Extract structured entities from a clinical note.

    Parameters
    ----------
    note : str
        The clinical note to process
    few_shot : bool
        Whether to use few-shot prompting

    Returns
    -------
    dict
        Extracted entities with keys: diagnosis, medications, lab_values, confidence
    """
    # TODO: Implement this function
    #
    # Steps:
    # 1. Get the LLM client
    # 2. Build the prompt
    # 3. Call the LLM
    # 4. Parse the JSON response
    # 5. Validate the response
    # 6. Return the structured data

    pass


def validate_response(response: dict) -> bool:
    """
    Validate that the response has the required structure.

    Parameters
    ----------
    response : dict
        The parsed response from the LLM

    Returns
    -------
    bool
        True if valid, False otherwise
    """
    # TODO: Implement validation
    #
    # Required fields:
    # - diagnosis (string)
    # - medications (list)
    # - lab_values (dict)
    # - confidence (float, 0-1)

    pass


def parse_json_response(response_text: str) -> Optional[dict]:
    """
    Parse JSON from the LLM response text.

    The response might contain additional text around the JSON,
    so we need to extract just the JSON portion.

    Parameters
    ----------
    response_text : str
        Raw response from the LLM

    Returns
    -------
    dict or None
        Parsed JSON or None if parsing fails
    """
    # TODO: Implement JSON parsing
    #
    # Hints:
    # - The JSON might be wrapped in ```json ... ```
    # - Try to find the { and } boundaries
    # - Handle parsing errors gracefully

    pass


if __name__ == "__main__":
    # Example usage
    sample_note = """
    The patient, a 55-year-old female with a history of type 2 diabetes mellitus,
    was admitted to the hospital after reporting nausea, vomiting, and polyuria for
    the past 24 hours. Lab tests revealed acidosis with an anion gap of 32 mmol/L
    and acute kidney injury. Serum glucose concentration of 366 mg/dL. The patient
    received a continuous insulin infusion and was transitioned to metformin and
    glipizide. Diagnosis: Diabetic ketoacidosis (DKA).
    """

    print("Testing entity extraction...")
    print("-" * 50)

    # TODO: Test your implementation
    # result = extract_entities(sample_note, few_shot=False)
    # print("Zero-shot result:")
    # print(json.dumps(result, indent=2))
    #
    # result_few_shot = extract_entities(sample_note, few_shot=True)
    # print("\nFew-shot result:")
    # print(json.dumps(result_few_shot, indent=2))
