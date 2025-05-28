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
4. Build professional documentation site (MkDocs)
5. Deploy to GitHub Pages (via GitHub Actions)
"""

import subprocess
import sys
from pathlib import Path
import json

# Ensure the 'reports' module can be found
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

try:
    from reports import report_generator
except ImportError as e:
    print(f"❌ Error importing report_generator: {e}")
    print(f"   Ensure 'reports/report_generator.py' exists in: {project_root}")
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
    print("   • Build automated reports")
    print("   • Deploy to GitHub Pages")
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


def main(build_site=True):
    """
    Main function to orchestrate the complete analysis pipeline.

    Args:
        build_site (bool): Whether to build the MkDocs site after generating reports
    """
    print_pipeline_header()

    # Step 1: Generate report elements (data, charts, statistics)
    print("📊 Step 1: Generating neonatal feeding study data and analysis...")
    print("-" * 50)
    try:
        report_generator.generate_report_elements()
        print("✅ Report elements generated successfully!")

        # Print summary of findings
        summary_file = (
            project_root / "docs" / "data" / "comprehensive_summary.json"
        )
        if summary_file.exists():
            print()
            print_data_summary(summary_file)

    except Exception as e:
        print(f"❌ Error generating report elements: {e}")
        print("💡 Check that all dependencies are installed:")
        print("   pip install pandas altair numpy scipy")
        sys.exit(1)

    # Step 2: Build MkDocs site (optional)
    if build_site:
        print("\n🌐 Step 2: Building MkDocs documentation site...")
        print("-" * 50)
        try:
            # Run mkdocs build from the project root directory
            result = subprocess.run(
                ["mkdocs", "build"],
                capture_output=True,
                text=True,
                check=True,
                cwd=project_root,
            )
            print("✅ MkDocs site built successfully!")

            # Show build output if verbose
            if result.stdout:
                print("📝 Build details:")
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        print(f"   {line}")

            if result.stderr:
                print("⚠️  Build warnings:")
                for line in result.stderr.strip().split("\n"):
                    if line.strip():
                        print(f"   {line}")

        except subprocess.CalledProcessError as e:
            print(f"❌ Error building MkDocs site: {e}")
            print("📝 Build output:")
            if e.stdout:
                for line in e.stdout.strip().split("\n"):
                    print(f"   {line}")
            if e.stderr:
                for line in e.stderr.strip().split("\n"):
                    print(f"   {line}")
            print("\n💡 Troubleshooting tips:")
            print("   • Check that mkdocs.yml is properly configured")
            print("   • Ensure all chart JSON files were generated")
            print("   • Verify mkdocs-charts-plugin is installed")
            sys.exit(1)

        except FileNotFoundError:
            print("❌ Error: 'mkdocs' command not found")
            print(
                "💡 Install MkDocs with: pip install mkdocs mkdocs-material mkdocs-charts-plugin"
            )
            sys.exit(1)
    else:
        print("\n⏭️  Skipping MkDocs site build (--no-build flag used)")

    # Step 3: Provide next steps
    print("\n🎉 Pipeline completed successfully!")
    print("=" * 50)
    print("🚀 Next steps:")

    if build_site:
        print("   1. View your report locally:")
        print("      mkdocs serve")
        print("      Then open: http://127.0.0.1:8000")
        print()
        print("   2. Deploy to GitHub Pages:")
        print("      • Push to GitHub repository")
        print("      • Enable GitHub Pages in repository settings")
        print("      • GitHub Actions will automatically deploy updates")
    else:
        print("   1. Build the site:")
        print("      mkdocs build")
        print("   2. Serve locally:")
        print("      mkdocs serve")

    print()
    print("📁 Generated files:")
    print(f"   • Data: {project_root}/docs/data/")
    print(f"   • Charts: {project_root}/docs/charts/")
    print(f"   • Site: {project_root}/site/ (if built)")
    print()
    print("🔬 This pipeline demonstrates:")
    print("   ✓ Automated clinical data analysis")
    print("   ✓ Interactive data visualization")
    print("   ✓ Dynamic report generation")
    print("   ✓ Professional documentation")
    print("   ✓ Reproducible research workflows")


if __name__ == "__main__":
    # Parse command line arguments
    # Usage: python orchestrator.py [--no-build]
    should_build = True
    if len(sys.argv) > 1 and sys.argv[1] == "--no-build":
        should_build = False
        print("🔧 Running in --no-build mode (generate data/charts only)")
        print()

    main(build_site=should_build)
