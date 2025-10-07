# Automated Data Cleaning System

A smart, autonomous data cleaning tool that automatically detects and fixes common data quality issues in CSV, Excel, JSON, and other file formats.

# Install requirements
- pip install -r requirements.txt 

# Add Your Data
- Place your data files (CSV, Excel, JSON) in the data/raw/ folder
- Supported formats: CSV, XLSX, XLS, JSON, Parquet, TSV, Text

# Run the System

- python main.py

# Get Cleaned Data
- Find cleaned files in data/cleaned/
- View detailed reports in data/reports/

# What It Fixes Automatically
- Problem Detection
- Missing Values - Empty cells and blanks
- Duplicate Rows - Exact duplicate records
- Text Inconsistencies - Mixed cases, extra spaces
- Data Type Issues - Numbers stored as text, dates as strings
- Outliers - Extreme values in numeric columns

# Automatic Fixes
- Smart Imputation - Fills missing values intelligently
- Text Standardization - Consistent capitalization and formatting
- Duplicate Removal - Keeps only unique records
- Outlier Handling - Caps extreme values reasonably
- Email Validation - Formats and validates email addresses
- Phone Standardization - Consistent phone number formats

# Features
- Autonomous Cleaning
- No configuration needed! The system automatically
- Analyzes your data for problems
- Applies appropriate cleaning methods
- Generates detailed quality reports
- Smart Pattern Recognition
- Email detection - Validates and standardizes email formats
- Phone recognition - Formats phone numbers consistently
- Name formatting - Proper case for names, uppercase for IDs
- Department normalization - Standardizes category names

# Project Structure

- data_cleaning_system/
- ├── data/
- │   ├── raw/           # Put your input files here
- │   ├── cleaned/       # Cleaned files appear here
- │   └── reports/       # Detailed quality reports
- ├── main.py            # Main cleaning system
- ├── requirements.txt   # Python dependencies
- └── README.md         # This file

# Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn (for advanced imputation)
- openpyxl (for Excel support)


# What Makes This Special
- No configuration needed - Works out of the box
- Handles multiple file formats - CSV, Excel, JSON, more
- Machine learning powered - Smart imputation and detection
- Comprehensive reporting - Detailed quality analysis
- Autonomous operation - Finds and fixes problems automatically
