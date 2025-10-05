# Configuration settings for the cleaning system

CLEANING_CONFIGS = {
    'basic': {
        'ml_imputation': False,
        'detect_outliers': False,
        'auto_feature_selection': False,
        'text_cleaning': True,
        'remove_duplicates': True
    },
    'standard': {
        'ml_imputation': True,
        'detect_outliers': True,
        'auto_feature_selection': False,
        'text_cleaning': True,
        'remove_duplicates': True
    },
    'advanced': {
        'ml_imputation': True,
        'detect_outliers': True,
        'auto_feature_selection': True,
        'text_cleaning': True,
        'remove_duplicates': True,
        'smart_type_detection': True
    }
}

FILE_SETTINGS = {
    'supported_formats': ['.csv', '.xlsx', '.xls', '.json', '.parquet', '.tsv', '.txt'],
    'default_output_format': 'csv',
    'auto_create_directories': True
}

