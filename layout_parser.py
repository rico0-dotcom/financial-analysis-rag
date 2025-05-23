import pandas as pd
import numpy as np
import re
import spacy


nlp = spacy.load("en_core_web_sm")

# Detecting layout if there is horizontal data
def detect_horizontal_layout(df, min_date_cols=1):
    """Detect horizontal layout if column names or header rows contain date patterns."""
    year_pattern = re.compile(r'(20\d{2})')
   # Check column names (excluding first)
    col_dates = sum(bool(year_pattern.search(str(c))) for c in df.columns[1:])
    if col_dates >= min_date_cols:
        return True
    # Fallback: check first few rows for dates
    for idx in df.index[:3]:
        row = df.loc[idx].astype(str).tolist()[1:]
        row_dates = sum(bool(year_pattern.search(c)) for c in row)
        if row_dates >= min_date_cols:
            return True
    return False

# Cleaning & Transposing Horizontal Layout
def clean_horizontal_layout(df):
    """Convert horizontal layout to vertical: dates -> rows, terms -> columns."""
    df2 = df.copy()
    # Determine header row index
    year_pattern = re.compile(r'(20\d{2})')
    if sum(bool(year_pattern.search(str(c))) for c in df2.columns[1:]) >= 1:
        header_idx = -1
    else:
        header_idx = 0
        for i in range(min(3, len(df2))):
            row = df2.iloc[i].astype(str).tolist()[1:]
            if sum(bool(year_pattern.search(c)) for c in row) >= 1:
                header_idx = i
                break
    # Set header
    if header_idx >= 0:
        df2.columns = df2.iloc[header_idx].astype(str).tolist()
        df2 = df2.iloc[header_idx+1:]
   
    # Extract terms and data
    terms = df2.iloc[:,0].astype(str).tolist()
    data = df2.iloc[:,1:].copy()
    # Convert to float safely
    data = data.map(safe_float)
    # Build vertical DataFrame
    cleaned_df = pd.DataFrame(data.values.T, index=data.columns, columns=terms)
    return cleaned_df

#Convert values like'$1,234', '(56)', '-' to float or NaN
def safe_float(val):
     
    if pd.isnull(val):
        return np.nan
    val = str(val).replace(',', '').replace('$', '').replace('(', '-').replace(')', '').strip()
    val = re.sub(r'[^0-9.\-]+$', '', val)
    try:
        return float(val)
    except ValueError:
        return np.nan

#Detects and transforms layout into a clean format.
def smart_layout_parser(df, vertical_threshold=0.6):
 
    
    
    if detect_horizontal_layout(df):
        layout_type = "horizontal"
        cleaned_df = clean_horizontal_layout(df)
    else:
        layout_type = "vertical"
        
        #
        cleaned_df = df.copy()
        
        #cleaned_df=clean_vertical_layout(cleaned_df)
        cleaned_df=cleaned_df.set_index(cleaned_df.columns[0])
        cleaned_df = cleaned_df.map(safe_float)
    cleaned_df.dropna(axis=1,how='all',inplace=True)
    return cleaned_df, layout_type
