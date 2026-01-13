#!/usr/bin/env python3
"""
Assignment data generator for EHR encounter data.

Creates realistic EHR patient encounter data with proper medical coding,
aligned to the Nature dataset schema. Generates all required tables for
the Polars data cleaning assignment.

Usage:
    python generate_assignment_data.py --size small
    python generate_assignment_data.py --size large
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import polars as pl
import numpy as np
import yaml

# Imports when faker is available
from faker import Faker

# Load data dictionary for schema reference
DATA_DICT_PATH = Path(__file__).parent / "data_dictionary.yaml"
with open(DATA_DICT_PATH) as f:
    DATA_DICT = yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate EHR encounter assignment data"
    )
    parser.add_argument(
        "--size",
        choices=["small", "large"],
        default="small",
        help="Dataset size (small=500K encounters, large=5M encounters)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for generated Parquet files",
    )
    return parser.parse_args()


def generate_patients(num_patients: int, faker: Faker) -> pl.DataFrame:
    """Generate patient demographic data."""

    patients = []
    for i in range(num_patients):
        # Generate realistic patient ages (skewed older for medical data)
        age = np.random.lognormal(mean=3.5, sigma=0.5)  # Mean ~33, median ~31
        age = int(np.clip(age, 18, 95))

        # Generate realistic patient data
        patient_id = f"PAT-{i:06d}"

        patients.append(
            {
                "patient_id": patient_id,
                "age": age,
                # Add other demographics as needed based on data dictionary
            }
        )

    return pl.DataFrame(patients)


def generate_encounters(
    patients_df: pl.DataFrame, target_rows: int, faker: Faker
) -> pl.DataFrame:
    """Generate encounter data with proper medical coding."""
    data = DATA_DICT["tables"]["encounters"]["columns"]

    # Generate multiple encounters per patient
    num_patients = len(patients_df)
    encounters_per_patient = max(1, target_rows // num_patients)

    encounters = []
    encounter_counter = 1

    for patient_row in patients_df.iter_rows(named=True):
        patient_id = patient_row["patient_id"]
        age = patient_row["age"]

        # Age-appropriate encounter frequency
        base_encounters = 1 + (age // 20)  # More encounters for older patients

        for enc_num in range(max(1, encounters_per_patient)):
            # Random date within last 2 years
            days_ago = faker.random_int(0, 730)  # 2 years
            encounter_date = datetime.now() - timedelta(days=days_ago)

            # Encounter type by age groups
            if age < 30:
                encounter_type = faker.random_element(
                    ["Outpatient", "Emergency", "Urgent Care", "Telehealth"]
                )
            elif age < 65:
                encounter_type = faker.random_element(
                    ["Outpatient", "Inpatient", "Emergency", "Telehealth"]
                )
            else:
                encounter_type = faker.random_element(
                    ["Inpatient", "Outpatient", "Emergency", "Skilled Nursing"]
                )

            # Facility
            facility = faker.random_element(data["facility"]["values"])

            # Primary diagnosis (diabetes-related for this dataset)
            icd10_codes = [
                "E11.9",
                "E11.8",
                "E11.65",
                "E11.21",
                "E11.22",
                "E11.40",
                "E11.41",
            ]
            icd10_code = faker.random_element(icd10_codes)

            icd10_descriptions = {
                "E11.9": "Type 2 diabetes mellitus without complications",
                "E11.8": "Type 2 diabetes mellitus with unspecified complications",
                "E11.65": "Type 2 diabetes mellitus with hyperglycemia",
                "E11.21": "Type 2 diabetes mellitus with diabetic nephropathy",
                "E11.22": "Type 2 diabetes mellitus with diabetic chronic kidney disease",
                "E11.40": "Type 2 diabetes mellitus with diabetic neuropathy",
                "E11.41": "Type 2 diabetes mellitus with diabetic neuropathy with ulcer",
            }
            icd10_desc = icd10_descriptions[icd10_code]

            # Admission and discharge timestamps
            admit_ts = encounter_date.replace(
                hour=faker.random_int(8, 16), minute=0, second=0
            )

            if encounter_type == "Inpatient":
                los_days = faker.random_int(1, 7)
                discharge_ts = admit_ts + timedelta(days=los_days)
            else:
                los_days = 0
                discharge_ts = admit_ts + timedelta(hours=faker.random_int(1, 6))

            # Disposition
            if encounter_type == "Inpatient":
                disposition = faker.random_element(["Home", "Rehab", "SNF", "Transfer"])
            else:
                disposition = "Home"

            encounters.append(
                {
                    "encounter_id": f"ENC-{encounter_date.year}-{encounter_counter:08d}",
                    "patient_id": patient_id,
                    "facility": facility,
                    "encounter_type": encounter_type,
                    "admit_ts": admit_ts,
                    "discharge_ts": discharge_ts,
                    "primary_diagnosis_icd10": icd10_code,
                    "primary_diagnosis_desc": icd10_desc,
                    "disposition": disposition,
                }
            )

            encounter_counter += 1

            # Stop if we've reached target
            if len(encounters) >= target_rows:
                break

        if len(encounters) >= target_rows:
            break

    return pl.DataFrame(encounters[:target_rows])


def generate_procedures(
    encounters_df: pl.DataFrame, target_rows: int, faker: Faker
) -> pl.DataFrame:
    """Generate procedure data with HCPCS codes."""
    data = DATA_DICT["tables"]["procedures"]["columns"]

    procedures = []
    procedure_counter = 1

    # Common diabetes-related HCPCS codes
    hcpcs_codes = {
        "83036": "Hemoglobin A1c test",
        "82947": "Glucose test",
        "82948": "Glucose test using blood reagent strip",
        "82951": "Glucose tolerance test",
        "83036": "Glycated hemoglobin test",
        "99214": "Office visit for established patient",
        "99213": "Office visit for established patient",
        "99215": "Office visit for established patient",
        "G0463": "Hospital outpatient clinic visit",
        "G0479": "Laboratory tests",
    }

    for encounter_row in encounters_df.iter_rows(named=True):
        encounter_id = encounter_row["encounter_id"]
        encounter_type = encounter_row["encounter_type"]
        encounter_date = encounter_row["admit_ts"]

        # Number of procedures per encounter
        if encounter_type == "Inpatient":
            if faker.random_int(0, 9) < 8:  # 80% have procedures
                num_procedures = faker.random_int(2, 6)
            else:
                num_procedures = 1  # At least one lab
        elif encounter_type == "Emergency":
            if faker.random_int(0, 9) < 6:  # 60% have procedures
                num_procedures = faker.random_int(1, 3)
            else:
                num_procedures = 1  # At least one lab
        else:
            if faker.random_int(0, 9) < 4:  # 40% have procedures
                num_procedures = faker.random_int(1, 2)
            else:
                num_procedures = 1  # At least one lab

        for proc_num in range(num_procedures):
            # Select HCPCS code
            hcpcs_code = faker.random_element(list(hcpcs_codes.keys()))
            hcpcs_desc = hcpcs_codes[hcpcs_code]

            # Procedure timestamp (during encounter)
            proc_offset = faker.random_int(
                0,
                int(
                    (encounter_row["discharge_ts"] - encounter_date).total_seconds()
                    // 60
                ),
            )
            procedure_ts = encounter_date + timedelta(minutes=proc_offset)

            # Ordering provider
            provider_id = f"DR-{faker.random_int(1000, 9999):04d}"

            # Result value and unit for lab tests
            if hcpcs_code in ["83036"]:  # HbA1c
                result_value = round(np.random.uniform(5.0, 12.0), 1)
                result_unit = "%"
            elif hcpcs_code.startswith("829"):  # Glucose
                result_value = round(np.random.uniform(80, 300), 0)
                result_unit = "mg/dL"
            else:
                result_value = None
                result_unit = None

            procedures.append(
                {
                    "procedure_id": f"PROC-{procedure_ts.year}-{procedure_counter:08d}",
                    "encounter_id": encounter_id,
                    "hcpcs_code": hcpcs_code,
                    "hcpcs_desc": hcpcs_desc,
                    "procedure_ts": procedure_ts,
                    "ordering_provider": provider_id,
                    "result_value": result_value,
                    "result_unit": result_unit,
                }
            )

            procedure_counter += 1

            # Stop if we've reached target
            if len(procedures) >= target_rows:
                break

        if len(procedures) >= target_rows:
            break

    return pl.DataFrame(procedures[:target_rows])


def generate_vitals(
    patients_df: pl.DataFrame, target_rows: int, faker: Faker
) -> pl.DataFrame:
    """Generate vital signs measurements."""
    data = DATA_DICT["tables"]["vitals"]["columns"]

    vitals = []
    vital_counter = 1

    # Calculate measurements needed per patient
    num_patients = len(patients_df)
    measurements_per_patient = max(1, target_rows // num_patients)

    for patient_row in patients_df.iter_rows(named=True):
        patient_id = patient_row["patient_id"]
        age = patient_row["age"]

        for meas_num in range(measurements_per_patient):
            # Random measurement time within last 6 months
            days_ago = faker.random_int(0, 180)
            measurement_ts = datetime.now() - timedelta(
                days=days_ago, hours=faker.random_int(0, 23)
            )

            # Measurement source
            source = faker.random_element(data["source"]["values"])

            # Age-adjusted vitals
            # Heart rate
            hr = 70 + (age - 40) * 0.2 + faker.random.uniform(-20, 30)
            hr = int(np.clip(hr, 40, 180))

            # Blood pressure
            systolic = 120 + (age - 40) * 0.5 + faker.random.uniform(-15, 25)
            diastolic = 80 + (age - 40) * 0.3 + faker.random.uniform(-10, 15)
            systolic = int(np.clip(systolic, 80, 200))
            diastolic = int(np.clip(diastolic, 40, 120))

            # Blood glucose (higher for diabetic patients)
            glucose = np.random.lognormal(mean=4.8, sigma=0.3)  # Skewed higher
            glucose = np.clip(glucose, 70, 400)

            # Temperature
            temp = faker.random.uniform(36.5, 38.0)

            # SpO2
            spo2 = faker.random.uniform(94, 100)

            vitals.append(
                {
                    "vital_id": f"VIT-{measurement_ts.year}-{vital_counter:08d}",
                    "patient_id": patient_id,
                    "measurement_ts": measurement_ts,
                    "source": source,
                    "heart_rate": hr,
                    "systolic_bp": systolic,
                    "diastolic_bp": diastolic,
                    "blood_glucose": round(glucose, 1),
                    "temperature": round(temp, 1),
                    "spo2": round(spo2, 1),
                }
            )

            vital_counter += 1

            # Stop if we've reached target
            if len(vitals) >= target_rows:
                break

        if len(vitals) >= target_rows:
            break

    return pl.DataFrame(vitals[:target_rows])


def main() -> None:
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    faker = Faker()
    faker.seed_instance(args.seed)
    np.random.seed(args.seed)

    # Get row count targets from data dictionary
    encounter_target = DATA_DICT["tables"]["encounters"]["row_count_target"][args.size]
    procedure_target = DATA_DICT["tables"]["procedures"]["row_count_target"][args.size]
    vitals_target = DATA_DICT["tables"]["vitals"]["row_count_target"][args.size]
    patient_target = DATA_DICT["tables"]["patients"]["row_count_target"][args.size]

    logger.info(f"Generating {args.size} EHR dataset:")
    logger.info(f"  - Target patients: {patient_target:,}")
    logger.info(f"  - Target encounters: {encounter_target:,}")
    logger.info(f"  - Target procedures: {procedure_target:,}")
    logger.info(f"  - Target vitals: {vitals_target:,}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    logger.info("Generating patients...")
    patients_df = generate_patients(patient_target, faker)

    logger.info("Generating encounters...")
    encounters_df = generate_encounters(patients_df, encounter_target, faker)

    logger.info("Generating procedures...")
    procedures_df = generate_procedures(encounters_df, procedure_target, faker)

    logger.info("Generating vitals...")
    vitals_df = generate_vitals(patients_df, vitals_target, faker)

    logger.info("Writing Parquet files...")

    # Use row groups for better streaming performance
    encounters_write_path = output_dir / "encounters.parquet"
    encounters_df.write_parquet(
        encounters_write_path, compression="snappy", row_group_size=100_000
    )

    procedures_write_path = output_dir / "procedures.parquet"
    procedures_df.write_parquet(
        procedures_write_path, compression="snappy", row_group_size=100_000
    )

    vitals_write_path = output_dir / "vitals.parquet"
    vitals_df.write_parquet(
        vitals_write_path, compression="snappy", row_group_size=100_000
    )

    patients_write_path = output_dir / "patients.parquet"
    patients_df.write_parquet(patients_write_path, compression="snappy")

    # Report statistics
    encounter_size_mb = encounters_write_path.stat().st_size / (1024 * 1024)
    procedure_size_mb = procedures_write_path.stat().st_size / (1024 * 1024)
    vitals_size_mb = vitals_write_path.stat().st_size / (1024 * 1024)
    patient_size_mb = patients_write_path.stat().st_size / (1024 * 1024)

    logger.info("Generation complete!")
    logger.info(f"Files written to: {output_dir}")
    logger.info(
        f" - encounters.parquet: {encounters_df.height:,} rows, {encounter_size_mb:.1f} MB"
    )
    logger.info(
        f" - procedures.parquet: {procedures_df.height:,} rows, {procedure_size_mb:.1f} MB"
    )
    logger.info(
        f" - vitals.parquet: {vitals_df.height:,} rows, {vitals_size_mb:.1f} MB"
    )
    logger.info(
        f" - patients.parquet: {patients_df.height:,} rows, {patient_size_mb:.1f} MB"
    )

    # Write generation metadata
    metadata = {
        "generation_timestamp": datetime.now().isoformat(),
        "size": args.size,
        "seed": args.seed,
        "targets": {
            "encounters": encounter_target,
            "procedures": procedure_target,
            "vitals": vitals_target,
            "patients": patient_target,
        },
        "actuals": {
            "encounters": encounters_df.height,
            "procedures": procedures_df.height,
            "vitals": vitals_df.height,
            "patients": patients_df.height,
        },
        "file_sizes_mb": {
            "encounters": round(encounter_size_mb, 2),
            "procedures": round(procedure_size_mb, 2),
            "vitals": round(vitals_size_mb, 2),
            "patients": round(patient_size_mb, 2),
        },
    }

    metadata_path = output_dir / "generation_metadata.yaml"
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)

    logger.info(f"Metadata written to: {metadata_path}")


if __name__ == "__main__":
    main()
