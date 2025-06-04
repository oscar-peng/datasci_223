#!/usr/bin/env python3
"""
Neonatal Feeding Study Pipeline Orchestrator

This script demonstrates how to build an automated clinical research pipeline
that generates interactive reports and can be deployed to GitHub Pages.

Based on real neonatal feeding research but simplified for educational purposes.
This pipeline studies optimal feeding times for premature babies - a critical
clinical question that affects thousands of infants worldwide.

Pipeline Steps:
1. Generate realistic clinical data (neonatal feeding study)
2. Perform statistical analysis (correlations, t-tests, regression)
3. Create interactive visualizations (Altair/Vega-Lite charts)
"""

import sys
from pathlib import Path
import json

# Ensure the project modules can be found
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

try:
    from utils import data_generator
    from utils import report_generator
    from utils import visualization
except ImportError as e:
    print(f"❌ Error importing required modules: {e}")
    print(f"   Ensure all required modules exist in: {project_root}/utils/")
    sys.exit(1)


def print_pipeline_header():
    """Print an informative header about what this pipeline does."""
    print("🏥 Neonatal Feeding Study Pipeline")
    print("=" * 50)
    print(
        "📋 Research Question: What factors predict time to full oral feeding"
    )
    print("   in premature infants?")
    print()
    print("🎯 Pipeline Goals:")
    print("   • Generate realistic clinical data")
    print("   • Perform statistical analysis")
    print("   • Create interactive visualizations")
    print()


def print_data_summary(summary_file):
    """Print key findings from the analysis."""
    try:
        with open(summary_file, "r") as f:
            summary = json.load(f)

        study = summary["study_overview"]
        stats = summary["statistical_analysis"]

        print("📊 Key Findings:")
        print(f"   • Analyzed {study['total_patients']} premature infants")
        print(
            f"   • Mean gestational age: {study['mean_gestational_age']:.1f} weeks"
        )
        print(
            f"   • Mean time to full oral feeding: {study['mean_time_to_fof']:.1f} days"
        )
        print(
            f"   • Primary finding: {stats['ga_regression']['interpretation']}"
        )
        print(
            f"   • Correlation (GA vs Time to FOF): {stats['correlations']['ga_vs_time_to_fof']:.3f}"
        )

        if stats["mechanical_ventilation_effect"]["significant"]:
            vent_impact = (
                stats["mechanical_ventilation_effect"]["mean_with_vent"]
                - stats["mechanical_ventilation_effect"]["mean_without_vent"]
            )
            print(
                f"   • Mechanical ventilation adds {vent_impact:.1f} days on average"
            )

    except Exception as e:
        print(f"   ⚠️  Could not load summary data: {e}")


def main():
    """Main function to orchestrate the complete analysis pipeline."""
    print_pipeline_header()

    # Step 1: Generate synthetic data
    print("📊 Step 1: Generating neonatal feeding study data...")
    print("-" * 50)
    try:
        data = data_generator.generate_synthetic_data(n_patients=100)
        print("✅ Synthetic data generated successfully!")
    except Exception as e:
        print(f"❌ Error generating data: {e}")
        sys.exit(1)

    # Step 2: Generate reports and visualizations
    print("\n📝 Step 2: Generating reports and visualizations...")
    print("-" * 50)
    try:
        # Generate statistical analysis
        report_generator.generate_statistical_report(data)

        # Create visualizations
        visualization.generate_charts(data)

        print("✅ Reports and visualizations generated successfully!")

        # Print summary of findings
        summary_file = (
            project_root / "docs" / "data" / "comprehensive_summary.json"
        )
        if summary_file.exists():
            print()
            print_data_summary(summary_file)

    except Exception as e:
        print(f"❌ Error generating reports: {e}")
        print("💡 Check that all dependencies are installed:")
        print("   pip install pandas altair numpy scipy")
        sys.exit(1)

    # Step 3: Provide next steps
    print("\n🎉 Pipeline completed successfully!")
    print("=" * 50)
    print("📁 Generated files:")
    print(f"   • Data: {project_root}/docs/data/")
    print(f"   • Charts: {project_root}/docs/charts/")
    print(f"   • Analysis: {project_root}/docs/analysis/")


if __name__ == "__main__":
    main()
