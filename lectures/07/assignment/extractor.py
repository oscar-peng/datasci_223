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

    Returns
    -------
    tuple
        (client, provider) where provider is 'openai' or 'huggingface'
    """
    # Check for OpenAI API key
    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI

        return OpenAI(), "openai"

    # Check for Hugging Face API key
    if os.environ.get("HUGGINGFACE_API_KEY"):
        import requests

        return None, "huggingface"

    raise ValueError(
        "No API key found. Set OPENAI_API_KEY or HUGGINGFACE_API_KEY"
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

    Parameters
    ----------
    prompt : str
        The prompt to send
    provider : str
        'openai' or 'huggingface'
    client : optional
        The OpenAI client if using OpenAI

    Returns
    -------
    str
        The raw response text
    """
    if provider == "openai":
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use a cost-effective model
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical information extraction assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,  # Deterministic for extraction
            max_tokens=500,
        )
        return response.choices[0].message.content

    elif provider == "huggingface":
        import requests

        api_key = os.environ.get("HUGGINGFACE_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"}

        # Using a capable open model
        api_url = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"

        response = requests.post(
            api_url, headers=headers, json={"inputs": prompt}, timeout=60
        )

        if response.status_code == 200:
            return response.json()[0]["generated_text"]
        else:
            raise Exception(f"API error: {response.status_code}")

    raise ValueError(f"Unknown provider: {provider}")


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
    Patient is a 58-year-old male presenting with chest pain radiating to the left arm. 
    Blood pressure 145/92 mmHg, heart rate 88 bpm. Troponin elevated at 0.8 ng/mL.
    ECG shows ST elevation in leads V1-V4. Patient started on aspirin 325mg and 
    heparin drip. Diagnosis: Acute ST-elevation myocardial infarction (STEMI).
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
