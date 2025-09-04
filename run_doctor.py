#!/usr/bin/env python3
"""
Script to run the Python Production Doctor on a single file
"""
import sys
import os
from python_doctor import ProductionDoctor, ConfigManager, DiagnosticResult

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_doctor.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        sys.exit(1)
    
    # Create a config manager
    config = ConfigManager()
    
    # Create doctor instance
    doctor = ProductionDoctor(file_path, config)
    
    # Run diagnostics
    print(f"Running diagnostics on {file_path}")
    result = doctor.run_diagnostics()
    
    # Print results
    print(f"\nDiagnostics complete for {file_path}")
    print(f"Total issues found: {len(result.issues)}")
    
    # Categorize issues by severity
    critical = [i for i in result.issues if i.severity == 'critical']
    serious = [i for i in result.issues if i.severity == 'serious']
    minor = [i for i in result.issues if i.severity == 'minor']
    
    print(f"Critical issues: {len(critical)}")
    print(f"Serious issues: {len(serious)}")
    print(f"Minor issues: {len(minor)}")
    
    # Print details
    if result.issues:
        print("\nDetailed Issues:")
        for i, issue in enumerate(result.issues, 1):
            print(f"{i}. [{issue.severity.upper()}] {issue.category}: {issue.message} (line {issue.line_number})")
    
    # Print metrics
    print(f"\nCode Metrics:")
    for key, value in result.metrics.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()