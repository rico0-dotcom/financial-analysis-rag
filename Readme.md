Financial Statement Analysis System

**1. Overview**
This project implements a Python-based pipeline for extracting, structuring, and analyzing financial statement data from Excel files.

The system is designed to handle inconsistencies in financial statement formats and convert them into structured datasets suitable for empirical financial analysis.

**2. Functionality**
- Parses financial statements (Income Statement, Balance Sheet, Cash Flow)
- Extracts and normalizes line items from raw Excel inputs
- Maps extracted entries to standardized financial variables
- Generates structured datasets for analysis
- Computes financial ratios
- Performs multi-period trend analysis
  
**3. Methodology**

3.1 Data Extraction
- Reads Excel-based financial statements
- Identifies sections and line items using rule-based logic and pattern recognition
  
3.2 Line Item Mapping
- Rule-based keyword matching
- Fuzzy string matching (RapidFuzz)
- Embedding similarity (SentenceTransformers)
- Maps raw entries (e.g., 'Net profit after tax') to standardized variables
  
3.3 Feedback-Driven Mapping System
- Persistent mapping using JSON file
- Stores validated mappings across runs
- Human-in-the-loop (HITL) review for low-confidence matches
- Ensures auditability and reproducibility
  
3.4 Data Structuring
- Converts extracted data into tabular format
- Handles inconsistencies and missing values
- Produces datasets ready for analysis
  
3.5 Ratio Computation
- Computes profitability, liquidity, and leverage ratios
- Supports custom user-defined ratios
  
3.6 Trend Analysis
- Multi-period comparison of financial metrics
- Identifies directional trends
  
**4. Key Features**
- Handles heterogeneous financial statement formats
- Combines rule-based and NLP-based extraction
- Produces structured datasets for empirical analysis
- Persistent mapping system for consistency
- Human-in-the-loop validation
- Custom ratio support
  
**5. Example Workflow**
1. Upload Excel financial statements
2. Run parsing pipeline
3. Generate structured dataset
4. Compute ratios and trends

**6. Technologies Used**
- Python (Pandas, NumPy)
- SentenceTransformers
- RapidFuzz
- Scikit-learn
  
**7. Repository Structure**
- main.py
- financial_parsing_horizontal.py
- ratio.py
- mapping.json
- requirements.txt
  
**8. Relevance**
- Financial data extraction and cleaning
- Handling unstructured financial disclosures
- Preparing datasets for econometric analysis

**9. Status**
Work in progress. Core pipeline implemented with ongoing improvements.

**10. How to Run**
pip install -r requirements.txt
python main.py
