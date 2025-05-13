# Demo 3: API Usage and Prompt Engineering for Health Data

This demo explores how to effectively use language model APIs for healthcare applications, focusing on prompt engineering techniques to improve reliability and reduce hallucination.

## Setup

First, let's install the necessary packages:

```python
# Install required packages
%pip install -q openai python-dotenv pandas numpy matplotlib seaborn

%reset -f

# Import packages
import os
import json
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import openai

# Set model to use
MODEL_NAME = "gpt-4o-mini"  # Using a smaller, more cost-effective model
```

## Getting Your API Key

To use the OpenAI API, you'll need an API key:

1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up or log in to your account
3. Navigate to "API Keys" in the left sidebar
4. Click "Create new secret key"
5. Copy your API key and store it securely

For this demo, you can create a `.env` file in your working directory with: `OPENAI_API_KEY=your_api_key_here`

```python
# Configure OpenAI API key
if os.path.exists('.env'):
    # Load from .env file if it exists
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')
else:
    # Otherwise, prompt for API key
    print("No .env file found. Please enter your OpenAI API key:")
    openai.api_key = "PUT YOUR API KEY HERE IF NOT USING .env FILE"
    print("\nTo avoid entering your API key each time, create a .env file with:")
    print("OPENAI_API_KEY=your_api_key_here")
```

## Zero-Shot Learning

Let's start with zero-shot learning for medical text classification:

```python
def classify_medical_text(text: str, categories: List[str]) -> Dict[str, Any]:
    """
    Classify medical text into predefined categories using zero-shot learning.
    """
    prompt = f"""Classify the following medical text into one of these categories: {', '.join(categories)}.
    
Text: {text}

Provide the classification in JSON format with the following structure:
{{
    "category": "chosen_category",
    "confidence": confidence_score,
    "explanation": "brief explanation"
}}"""

    response = openai.ChatCompletion.create(
        model=MODEL_NAME,  # Using our configured model
        messages=[
            {"role": "system", "content": "You are a medical text classification expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    return json.loads(response.choices[0].message.content)

# Example usage
categories = ["Diagnosis", "Treatment", "Prognosis", "Medical History"]
text = "Patient presents with persistent cough and fever for 3 days. Chest X-ray shows right lower lobe infiltrate. Started on azithromycin 500mg daily."

result = classify_medical_text(text, categories)
print(json.dumps(result, indent=2))
```

## One-Shot Learning

Now let's try one-shot learning for medical report generation:

```python
def generate_medical_report(patient_data: Dict[str, Any]) -> str:
    """
    Generate a medical report using one-shot learning.
    """
    example = {
        "patient_id": "P12345",
        "age": 45,
        "symptoms": ["fever", "cough", "fatigue"],
        "vitals": {"temperature": 38.5, "heart_rate": 95, "blood_pressure": "120/80"},
        "diagnosis": "Acute bronchitis"
    }
    
    example_report = """MEDICAL REPORT
Patient ID: P12345
Age: 45

SYMPTOMS:
- Fever
- Cough
- Fatigue

VITAL SIGNS:
- Temperature: 38.5°C
- Heart Rate: 95 bpm
- Blood Pressure: 120/80 mmHg

DIAGNOSIS:
Acute bronchitis

RECOMMENDATIONS:
1. Rest and adequate hydration
2. Over-the-counter antipyretics for fever
3. Follow-up in 1 week if symptoms persist"""

    prompt = f"""Generate a medical report following this exact format:

{example_report}

Now generate a report for this patient data:
{json.dumps(patient_data, indent=2)}"""

    response = openai.ChatCompletion.create(
        model=MODEL_NAME,  # Using our configured model
        messages=[
            {"role": "system", "content": "You are a medical report generation expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    return response.choices[0].message.content

# Example usage
patient_data = {
    "patient_id": "P67890",
    "age": 62,
    "symptoms": ["chest pain", "shortness of breath", "sweating"],
    "vitals": {"temperature": 37.2, "heart_rate": 110, "blood_pressure": "145/90"},
    "diagnosis": "Suspected angina"
}

report = generate_medical_report(patient_data)
print(report)
```

## Few-Shot Learning

Let's implement few-shot learning for medical coding:

```python
def assign_icd_codes(clinical_note: str, num_examples: int = 3) -> List[Dict[str, Any]]:
    """
    Assign ICD-10 codes to a clinical note using few-shot learning.
    """
    examples = [
        {
            "note": "Patient presents with type 2 diabetes mellitus, uncontrolled. HbA1c 9.2%. Also reports diabetic retinopathy.",
            "codes": [
                {"code": "E11.9", "description": "Type 2 diabetes mellitus without complications"},
                {"code": "E11.3", "description": "Type 2 diabetes mellitus with ophthalmic complications"}
            ]
        },
        {
            "note": "Acute appendicitis with localized peritonitis. Patient taken to OR for appendectomy.",
            "codes": [
                {"code": "K35.2", "description": "Acute appendicitis with localized peritonitis"},
                {"code": "47.01", "description": "Laparoscopic appendectomy"}
            ]
        },
        {
            "note": "Hypertensive heart disease with heart failure. Patient on ACE inhibitor and diuretic.",
            "codes": [
                {"code": "I11.0", "description": "Hypertensive heart disease with heart failure"},
                {"code": "I50.9", "description": "Heart failure, unspecified"}
            ]
        }
    ]
    
    # Select random examples
    selected_examples = examples[:num_examples]
    
    prompt = f"""Assign ICD-10 codes to the following clinical notes. Here are some examples:

{json.dumps(selected_examples, indent=2)}

Now assign codes to this note:
{clinical_note}

Provide the codes in JSON format with the following structure:
[
    {{
        "code": "ICD-10 code",
        "description": "code description"
    }}
]"""

    response = openai.ChatCompletion.create(
        model=MODEL_NAME,  # Using our configured model
        messages=[
            {"role": "system", "content": "You are a medical coding expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    return json.loads(response.choices[0].message.content)

# Example usage
note = "Patient with chronic obstructive pulmonary disease, severe. Presents with acute exacerbation. Started on prednisone and antibiotics."
codes = assign_icd_codes(note)
print(json.dumps(codes, indent=2))
```

## Prompt Engineering Techniques

Let's implement some advanced prompt engineering techniques:

```python
def extract_structured_data(text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract structured data from medical text using schema-based prompting.
    """
    prompt = f"""Extract information from the following medical text according to this schema:
{json.dumps(schema, indent=2)}

Text: {text}

Provide the extracted data in JSON format matching the schema exactly.
If a field is not found in the text, use null as the value."""

    response = openai.ChatCompletion.create(
        model=MODEL_NAME,  # Using our configured model
        messages=[
            {"role": "system", "content": "You are a medical information extraction expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    return json.loads(response.choices[0].message.content)

# Example usage
schema = {
    "patient_demographics": {
        "age": "number",
        "gender": "string",
        "race": "string"
    },
    "vital_signs": {
        "temperature": "number",
        "heart_rate": "number",
        "blood_pressure": "string",
        "oxygen_saturation": "number"
    },
    "medications": {
        "current_meds": ["string"],
        "allergies": ["string"]
    }
}

text = """45-year-old African American male presents to ED. 
Vitals: T 38.2, HR 110, BP 145/90, O2 sat 96% on room air.
Current medications: Lisinopril 10mg daily, Metformin 1000mg BID.
Allergies: Penicillin, Sulfa drugs."""

structured_data = extract_structured_data(text, schema)
print(json.dumps(structured_data, indent=2))
```

## Chain-of-Thought Prompting

Let's implement chain-of-thought prompting for medical reasoning:

```python
def analyze_medical_case(case: str) -> Dict[str, Any]:
    """
    Analyze a medical case using chain-of-thought prompting.
    """
    prompt = f"""Analyze this medical case step by step:

{case}

Follow these steps:
1. List the key symptoms and findings
2. Identify potential differential diagnoses
3. Explain your reasoning for each diagnosis
4. Recommend next steps for diagnosis
5. Suggest initial treatment approach

Provide your analysis in JSON format with the following structure:
{{
    "symptoms": ["list of symptoms"],
    "findings": ["list of findings"],
    "differential_diagnoses": [
        {{
            "diagnosis": "diagnosis name",
            "probability": "high/medium/low",
            "reasoning": "explanation"
        }}
    ],
    "next_steps": ["list of recommended steps"],
    "treatment": ["list of treatment suggestions"]
}}"""

    response = openai.ChatCompletion.create(
        model=MODEL_NAME,  # Using our configured model
        messages=[
            {"role": "system", "content": "You are a medical case analysis expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    return json.loads(response.choices[0].message.content)

# Example usage
case = """A 35-year-old woman presents with 3 days of right upper quadrant pain, 
fever to 38.5°C, and nausea. Physical exam shows right upper quadrant tenderness 
and positive Murphy's sign. WBC is 15,000 with 85% neutrophils. 
Ultrasound shows gallbladder wall thickening and pericholecystic fluid."""

analysis = analyze_medical_case(case)
print(json.dumps(analysis, indent=2))
```

## Key Takeaways

1. **Prompt Engineering Techniques**
   - Zero-shot learning for classification
   - One-shot learning for report generation
   - Few-shot learning for coding
   - Chain-of-thought for complex reasoning

2. **Structured Output**
   - JSON schema for consistent formatting
   - Clear instructions for data extraction
   - Validation of extracted information

3. **Healthcare-Specific Considerations**
   - Medical terminology accuracy
   - Clinical reasoning transparency
   - Ethical and privacy concerns

4. **Best Practices**
   - Clear and specific prompts
   - Appropriate temperature settings
   - Error handling and validation
   - Documentation of prompt templates 