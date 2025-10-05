import pandas as pd
import numpy as np
import os
import warnings
import re
import glob
from pathlib import Path
from sklearn.impute import KNNImputer

# Loading files
class UniversalFileLoader:
    SUPPORTED_FORMATS = {
        '.csv': 'CSV',
        '.xlsx': 'Excel', 
        '.xls': 'Excel',
        '.json': 'JSON',
        '.parquet': 'Parquet',
        '.tsv': 'Tab-Separated',
        '.txt': 'Text File'
    }
    
    def load_file(self, file_path):
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
            
        file_ext = Path(file_path).suffix.lower()
        file_name = os.path.basename(file_path)
        
        print(f"Loading: {file_name} (format: {file_ext})")
        
        try:
            if file_ext == '.csv':
                return pd.read_csv(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            elif file_ext == '.json':
                return pd.read_json(file_path)
            elif file_ext == '.parquet':
                return pd.read_parquet(file_path)
            elif file_ext == '.tsv':
                return pd.read_csv(file_path, sep='\t')
            elif file_ext == '.txt':
                return self._load_text_file(file_path)
            else:
                print(f"Unsupported format: {file_ext}")
                return None
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            return None
    
    def _load_text_file(self, file_path):
        try:
            return pd.read_csv(file_path)
        except:
            try:
                return pd.read_csv(file_path, sep='\t')
            except:
                try:
                    return pd.read_csv(file_path, sep=' ')
                except Exception as e:
                    print(f"Could not parse text file: {e}")
                    return None
    
    def scan_directory(self, directory_path):
        if not os.path.exists(directory_path):
            return [], []
        
        supported_files = []
        unsupported_files = []
        
        for file_path in Path(directory_path).iterdir():
            if file_path.is_file():
                file_ext = file_path.suffix.lower()
                if file_ext in self.SUPPORTED_FORMATS:
                    supported_files.append(file_path.name)
                else:
                    unsupported_files.append(file_path.name)
        
        return supported_files, unsupported_files

class SmartDataCleaner:
    """Autonomous cleaner that detects and fixes data issues"""
    
    def __init__(self):
        self.cleaning_actions = []
        self.problems_detected = {}
    
    def analyze_data(self, df):
        """Comprehensive analysis to detect data problems"""
        analysis = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'total_missing': df.isnull().sum().sum(),
            'duplicates': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'text_columns': list(df.select_dtypes(include=['object']).columns),
            'outlier_columns': []
        }
        
        # Detecting outliers in numeric columns
        for col in analysis['numeric_columns']:
            if self._has_outliers(df[col]):
                analysis['outlier_columns'].append(col)
        
        # Detecting data quality issues
        analysis['problems'] = self._detect_problems(df, analysis)
        return analysis
    
    def _detect_problems(self, df, analysis):
        """Detect specific data quality problems"""
        problems = []
        
        # Missing values
        if analysis['total_missing'] > 0:
            problems.append(f"Missing values: {analysis['total_missing']} total")
        
        # Duplicates
        if analysis['duplicates'] > 0:
            problems.append(f"Duplicate rows: {analysis['duplicates']}")
        
        # Inconsistent text formatting
        text_issues = self._detect_text_issues(df, analysis['text_columns'])
        if text_issues:
            problems.extend(text_issues)
        
        # Outliers
        if analysis['outlier_columns']:
            problems.append(f"Outliers detected in: {', '.join(analysis['outlier_columns'])}")
        
        # Data type issues
        type_issues = self._detect_type_issues(df)
        if type_issues:
            problems.extend(type_issues)
        
        return problems
    
    def _clean_and_validate_emails(self, df):
        """Validate and clean email addresses"""
        df_clean = df.copy()
    
        # Finding email columns (any column with 'email' in the name)
        email_columns = [col for col in df_clean.columns if 'email' in col.lower()]
    
        for col in email_columns:
            print(f"Validating emails in: {col}")
        
            # Storing original count for reporting
            original_count = len(df_clean[col])
        
            # Converting to lowercase and strip whitespace
            df_clean[col] = df_clean[col].astype(str).str.lower().str.strip()
            
            # Email validation regex pattern
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            
            # Checking which emails are valid
            valid_emails = df_clean[col].str.match(email_pattern, na=False)
            invalid_count = (~valid_emails).sum()
            
            # Fixing common email issues automatically
            df_clean[col] = df_clean[col].apply(self._fix_common_email_issues)
            
            # Re-check validity after fixes
            valid_after_fix = df_clean[col].str.match(email_pattern, na=False)
            still_invalid = (~valid_after_fix).sum()
            
            # Reporting results
            if invalid_count > 0:
                fixed_count = invalid_count - still_invalid
                if fixed_count > 0:
                    self.cleaning_actions.append(f"Fixed {fixed_count} email formats in '{col}'")
                if still_invalid > 0:
                    self.cleaning_actions.append(f"Found {still_invalid} invalid emails in '{col}' (marked as 'invalid_email')")
                    
                    # Marking invalid emails
                    df_clean.loc[~valid_after_fix, col] = 'invalid_email' 
        
            return df_clean

    def _fix_common_email_issues(self, email):
        """Fix common email typos and issues"""
        if pd.isna(email) or email == 'nan':
            return 'unknown_email'
    
        email_str = str(email).strip()
       
        fixes = [
            (r'@gmailcom', '@gmail.com'),      
            (r'@gmail\.com$', '@gmail.com'),   
            (r'@yahooocom', '@yahoo.com'),    
            (r'@hotmailcom', '@hotmail.com'),  
            (r'@outlookcom', '@outlook.com'),  
            (r'@gmail\.com\.', '@gmail.com'),  
            (r'\s+', ''),                      
        ]
        
        for pattern, replacement in fixes:
            email_str = re.sub(pattern, replacement, email_str)
        
        return email_str
    
    def _standardize_phone_numbers(self, df):
        """Standardize phone number formats"""
        df_clean = df.copy()
        
        # Finding phone columns
        phone_columns = [col for col in df_clean.columns if any(keyword in col.lower() for keyword in ['phone', 'mobile', 'tel'])]
        
        for col in phone_columns:
            print(f"Standardizing phones in: {col}")
            
            original_count = len(df_clean[col])
            
            # Converting to string and clean
            df_clean[col] = df_clean[col].astype(str).str.strip()
            
            # Removing all non-numeric characters
            df_clean[col] = df_clean[col].str.replace(r'\D', '', regex=True)
            
            # Counting valid phone numbers
            valid_phones = df_clean[col].str.match(r'^\d{10}$', na=False)
            invalid_count = (~valid_phones).sum()
            
            # Format valid numbers as (XXX) XXX-XXXX
            def format_phone(phone):
                if pd.isna(phone) or phone == 'nan':
                    return 'unknown_phone'
                if len(phone) == 10:
                    return f"({phone[:3]}) {phone[3:6]}-{phone[6:]}"
                return 'invalid_phone'
            
            df_clean[col] = df_clean[col].apply(format_phone)
            
            # Report results
            if invalid_count > 0:
                self.cleaning_actions.append(f"Standardized {original_count - invalid_count} phone numbers in '{col}'")
                if invalid_count > 0:
                    self.cleaning_actions.append(f"Found {invalid_count} invalid phone numbers in '{col}'")
            
            print(f"Phones processed: {original_count}, Invalid: {invalid_count}")
    
        return df_clean

    def _detect_text_issues(self, df, text_columns):
        """Detect text formatting inconsistencies"""
        issues = []
        for col in text_columns:
            # Checking for case inconsistencies
            unique_values = df[col].dropna().unique()
            if len(unique_values) > 0:
                case_variations = len(set(str(val).strip().lower() for val in unique_values))
                if case_variations < len(unique_values):
                    issues.append(f"Case inconsistencies in '{col}'")
            
            # Checking for extra whitespace
            has_extra_space = any('  ' in str(val) for val in unique_values if pd.notna(val))
            if has_extra_space:
                issues.append(f"Extra whitespace in '{col}'")
        
        return issues
    
    def _detect_type_issues(self, df):
        """Detect potential data type conversion opportunities"""
        issues = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Checking if it's actually numeric data
                numeric_test = pd.to_numeric(df[col], errors='coerce')
                if numeric_test.notna().mean() > 0.7:  # 70% can be converted
                    issues.append(f"'{col}' appears to be numeric data stored as text")
                
                # Checking if it's date data
                date_test = pd.to_datetime(df[col], errors='coerce')
                if date_test.notna().mean() > 0.5:  # 50% can be converted
                    issues.append(f"'{col}' appears to be date data stored as text")
        
        return issues
    
    def _has_outliers(self, series):
        """Checking if a numeric series has outliers"""
        if len(series.dropna()) == 0:
            return False
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return ((series < lower_bound) | (series > upper_bound)).any()
    
    def auto_clean(self, df):
        """Automatically clean data based on detected problems"""
        df_clean = df.copy()
        self.cleaning_actions = []
        analysis = self.analyze_data(df)
        
        if not analysis['problems']:
            print("No data quality issues detected!")
            return df_clean
        
        print(f"Found {len(analysis['problems'])} issues:")
        for problem in analysis['problems']:
            print(f"  - {problem}")
        
        print("\nStarting automatic cleaning...")
        
        # Fix data types automatically
        df_clean = self._auto_fix_data_types(df_clean)
        
        # Handle missing values
        df_clean = self._auto_handle_missing(df_clean, analysis)
        
        # Clean text data
        df_clean = self._auto_clean_text(df_clean, analysis)
        
        # Validate emails
        df_clean = self._clean_and_validate_emails(df_clean)    
        
        # Standardize phone numbers
        df_clean = self._standardize_phone_numbers(df_clean)  
       
        # Handle outliers
        df_clean = self._auto_handle_outliers(df_clean, analysis)
        
        # Remove duplicates
        original_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed_duplicates = original_rows - len(df_clean)
        if removed_duplicates > 0:
            self.cleaning_actions.append(f"Removed {removed_duplicates} duplicate rows")
        
        print("Automatic cleaning completed!")
        return df_clean
    
    def _auto_fix_data_types(self, df):
        """Automatically fix data type issues"""
        df_clean = df.copy()
        
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # Trying numeric conversion
                numeric_version = pd.to_numeric(df_clean[col], errors='coerce')
                if numeric_version.notna().mean() > 0.8:
                    df_clean[col] = numeric_version
                    self.cleaning_actions.append(f"Converted '{col}' to numeric")
                    continue
                
                # Trying datetime conversion
                date_version = pd.to_datetime(df_clean[col], errors='coerce')
                if date_version.notna().mean() > 0.5:
                    df_clean[col] = date_version
                    self.cleaning_actions.append(f"Converted '{col}' to datetime")
        
        return df_clean
    
    def _auto_handle_missing(self, df, analysis):
        """Automatically handle missing values"""
        df_clean = df.copy()
        
        if analysis['total_missing'] == 0:
            return df_clean
        
        for col, missing_count in analysis['missing_values'].items():
            if missing_count > 0:
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    # Using ML imputation
                    if len(df_clean.columns) > 1:
                        # Using KNN for numeric columns with multiple features
                        numeric_cols = analysis['numeric_columns']
                        if len(numeric_cols) > 1:
                            imputer = KNNImputer(n_neighbors=5)
                            df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
                            self.cleaning_actions.append(f"Used ML imputation for numeric columns")
                            break  # All numeric columns handled at once
                    else:
                        # Basic median imputation
                        median_val = df_clean[col].median()
                        df_clean[col] = df_clean[col].fillna(median_val)
                        self.cleaning_actions.append(f"Filled missing values in '{col}' with median")
                else:
                    # For text columns, use mode
                    mode_val = df_clean[col].mode()
                    fill_value = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                    df_clean[col] = df_clean[col].fillna(fill_value)
                    self.cleaning_actions.append(f"Filled missing values in '{col}' with '{fill_value}'")
        
        return df_clean
    
    def _auto_clean_text(self, df, analysis):
        """Automatically clean text data"""
        df_clean = df.copy()
        
        for col in analysis['text_columns']:
            # Basic cleaning
            original_sample = df_clean[col].iloc[0] if len(df_clean[col]) > 0 else ""
            df_clean[col] = (
                df_clean[col]
                .astype(str)
                .str.strip()
                .str.replace(r'\s+', ' ', regex=True)
            )
            
            # Smart formatting based on column name patterns
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['name', 'first', 'last', 'full', 'city', 'country']):
                df_clean[col] = df_clean[col].str.title()
                self.cleaning_actions.append(f"Standardized '{col}' to title case")
            elif any(keyword in col_lower for keyword in ['email']):
                df_clean[col] = df_clean[col].str.lower()
                self.cleaning_actions.append(f"Standardized '{col}' to lowercase")
            elif any(keyword in col_lower for keyword in ['id', 'code', 'sku']):
                df_clean[col] = df_clean[col].str.upper().str.replace(' ', '')
                self.cleaning_actions.append(f"Standardized '{col}' to uppercase codes")
        
        return df_clean
    
    def _auto_handle_outliers(self, df, analysis):
        """Automatically handle outliers"""
        df_clean = df.copy()
        
        for col in analysis['outlier_columns']:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_before = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            
            # Caping outliers
            df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
            df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
            
            outliers_after = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            
            if outliers_before > 0:
                self.cleaning_actions.append(f"Capped {outliers_before} outliers in '{col}'")
        
        return df_clean
    
    def get_cleaning_summary(self, original_df, cleaned_df):
        """Generate cleaning summary"""
        return {
            'original_rows': len(original_df),
            'cleaned_rows': len(cleaned_df),
            'duplicates_removed': len(original_df) - len(cleaned_df),
            'original_missing': original_df.isnull().sum().sum(),
            'cleaned_missing': cleaned_df.isnull().sum().sum(),
            'actions_taken': self.cleaning_actions,
            'problems_fixed': len(self.cleaning_actions)
        }
    
    def save_cleaned_data(self, df, output_path):
        """Save cleaned data"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

# MAIN FUNCTION
def main_autonomous():
    cleaner = SmartDataCleaner()
    file_loader = UniversalFileLoader()
    
    # Ensuring directories exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/cleaned', exist_ok=True)
    os.makedirs('data/reports', exist_ok=True)
    
    # Finding data files
    raw_files = glob.glob('data/raw/*')
    
    if not raw_files:
        print("Please add your data files to the data/raw/ folder")
        return
    
    for file_path in raw_files:
        file_name = os.path.basename(file_path)
        file_ext = Path(file_path).suffix.lower()
        
        # Skipping unsupported files
        if file_ext not in ['.csv', '.xlsx', '.xls', '.json']:
            print(f"Skipping: {file_name} (unsupported format)")
            continue        
        try:
            # Loading data
            df = file_loader.load_file(file_path)
            if df is None:
                continue
            
            print(f"Dataset: {len(df)} rows, {len(df.columns)} columns")
            
            # Autonomous cleaning
            df_clean = cleaner.auto_clean(df)
            
            # Getting summary
            summary = cleaner.get_cleaning_summary(df, df_clean)
            
            # Showing results
            print(f"\nCLEANING COMPLETED!")
            print(f"   Rows processed: {summary['cleaned_rows']}")
            print(f"   Problems fixed: {summary['problems_fixed']}")
            print(f"   Duplicates removed: {summary['duplicates_removed']}")
            print(f"   Missing values fixed: {summary['original_missing'] - summary['cleaned_missing']}")
            
            if summary['actions_taken']:
                print(f"\nActions performed:")
                for action in summary['actions_taken']:
                    print(f" {action}")
            
            # Saving cleaned data
            output_file = f"data/cleaned/auto_cleaned_{Path(file_name).stem}.csv"
            cleaner.save_cleaned_data(df_clean, output_file)
            
            # Saving detailed report
            report_file = f"data/reports/auto_report_{Path(file_name).stem}.txt"
            with open(report_file, 'w') as f:
                f.write(f"AUTONOMOUS CLEANING REPORT - {file_name}\n")
                f.write("SUMMARY:\n")
                for key, value in summary.items():
                    if key != 'actions_taken':
                        f.write(f"  {key}: {value}\n")
                
                f.write("\nACTIONS TAKEN:\n")
                for action in summary['actions_taken']:
                    f.write(f"  {action}\n")
            
            print(f" Detailed report: {report_file}")
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")


if __name__ == "__main__":
    main_autonomous()         