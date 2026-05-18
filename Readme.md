# Automated Financial Statement Analysis System

Hybrid NLP-based pipeline for extracting, structuring, and analyzing financial statement data from heterogeneous Excel disclosures.

---

# Overview

This project implements a modular financial statement intelligence pipeline designed to process raw financial disclosures and convert them into structured analytical datasets suitable for financial analysis, empirical research, and downstream modeling.

The system combines rule-based extraction, fuzzy matching, transformer embeddings, and human-in-the-loop validation to handle inconsistencies commonly found in financial statement formats.

The pipeline supports:

- Financial statement extraction
- Line-item normalization
- Ratio computation
- Multi-period trend analysis
- Custom ratio creation
- Analytical insight generation

---

# System Architecture

![Architecture](ratio%20engine%20flowchart.png)

---

# Research Motivation

Financial disclosures are often heterogeneous, semi-structured, and inconsistent across firms and reporting formats. Extracting structured financial variables from raw statements is therefore a non-trivial data engineering problem.

This project was developed to create a reproducible framework for transforming raw financial statement disclosures into standardized analytical datasets suitable for:

- Financial analysis
- Ratio modeling
- Empirical finance research
- Trend analysis
- Downstream machine learning applications

---

# Core Capabilities

- Parses Income Statement, Balance Sheet, and Cash Flow statements
- Extracts and normalizes financial line items from Excel files
- Maps raw entries to standardized financial variables
- Computes profitability, liquidity, leverage, and valuation ratios
- Supports custom user-defined financial ratios
- Performs multi-period trend analysis
- Generates analytical summaries and insights
- Maintains persistent validated mappings across runs
- Supports human-in-the-loop validation for ambiguous mappings

---

# System Design

The system combines symbolic and semantic extraction approaches:

- Rule-based parsing for structural consistency
- Fuzzy matching for noisy financial labels
- Transformer embeddings for semantic similarity
- Persistent mapping storage for reproducibility
- Human-in-the-loop validation for low-confidence matches

This hybrid approach improves robustness across heterogeneous financial statement formats.

---

# Methodology

## 1. Data Extraction

- Reads Excel-based financial statements
- Detects financial sections and line items
- Handles inconsistent formatting and layouts
- Extracts numerical financial data into structured representations

---

## 2. Line Item Mapping

The mapping pipeline combines multiple approaches:

### Rule-Based Matching
- Keyword logic
- Pattern recognition
- Financial terminology heuristics

### Fuzzy Matching
- RapidFuzz similarity matching
- Handles noisy and inconsistent labels

### Semantic Similarity
- SentenceTransformer embeddings
- Maps semantically similar financial terms

Example:

| Raw Entry | Standardized Variable |
|---|---|
| Net profit after tax | Net Income |
| Trade receivables | Accounts Receivable |
| PPE | Property Plant & Equipment |

---

## 3. Feedback-Driven Mapping System

The system includes a persistent mapping framework:

- Stores validated mappings in JSON format
- Reuses mappings across runs
- Supports auditability and reproducibility
- Human-in-the-loop review for ambiguous matches

---

## 4. Data Structuring

- Converts extracted data into tabular datasets
- Aligns financial variables across periods
- Handles missing values and inconsistencies
- Produces datasets ready for analysis

---

## 5. Ratio Engine

The analytical engine computes:

### Profitability Ratios
- Gross Margin
- Operating Margin
- Net Profit Margin
- Return on Assets
- Return on Equity

### Liquidity Ratios
- Current Ratio
- Quick Ratio
- Cash Ratio

### Leverage Ratios
- Debt-to-Equity
- Debt Ratio
- Interest Coverage

### Valuation Ratios
- Earnings-based valuation metrics
- Market-related financial indicators

### Custom Ratios
- User-defined ratio formulas
- Dynamic ratio computation framework

---

## 6. Trend Analysis

The system performs multi-period analysis to identify:

- Financial performance trends
- Growth patterns
- Margin expansion/contraction
- Liquidity deterioration/improvement
- Changes in leverage profile

---

# Example Workflow

1. Upload Excel financial statements  
2. Run parsing pipeline  
3. Extract and standardize line items  
4. Generate structured financial dataset  
5. Compute ratios and trends  
6. Generate analytical insights and reports  

---

# Demo

The repository includes a demonstration video showing:

- Uploading raw financial statements
- Automated financial parsing
- Financial ratio computation
- Multi-period trend analysis
- Insight generation pipeline

Demo file:

```text
working of ratioEngine20mb.mp4
```

---


---

# Sample Files

| File | Description |
|---|---|
| `financial_parsing_horizontal.py` | Hybrid financial statement extraction pipeline |
| `ratio.py` | Ratio computation and analytical engine |
| `mapping.json` | Persistent validated financial mappings |
| `Walmart_financial_report.xlsx` | Example financial disclosure input |
| `working of ratioEngine20mb.mp4` | End-to-end demonstration |
| `ratio engine flowchart.png` | System architecture diagram |

---

# Technologies Used

## Programming & Data Processing
- Python
- Pandas
- NumPy

## NLP & Semantic Matching
- SentenceTransformers
- RapidFuzz

## Machine Learning & Analytics
- Scikit-learn

---

# Key Features

- Handles heterogeneous financial statement formats
- Hybrid symbolic + semantic extraction framework
- Persistent mapping architecture
- Human-in-the-loop validation
- Multi-period financial analysis
- Custom ratio framework
- Trend analysis pipeline
- Structured analytical dataset generation
- Low-confidence mappings can optionally be escalated to LLM-assisted interpretation for additional semantic resolution.

---

# Applications

This system can support:

- Financial statement analysis
- Empirical finance research
- Financial data engineering
- Accounting data normalization
- Econometric dataset preparation
- Automated financial analytics pipelines

---

# Current Limitations

- Performance depends on statement formatting quality
- Some mappings may still require human validation
- Optimized primarily for Excel-based disclosures
- LLM-assisted analytical summaries remain experimental
- Cross-company accounting standard differences may require additional normalization

---

# Current Status

Core extraction and analytical pipeline implemented.

Ongoing development includes:

- Improved semantic matching
- Enhanced analytical reporting
- Expanded ratio libraries
- Interactive querying capabilities
- Improved visualization support

---

# Setup Notes

Some analytical insight-generation components rely on external API access and environment variables that are intentionally excluded from the repository for security reasons.

As a result:

- Core financial parsing and ratio computation modules are included
- Financial extraction pipeline is fully visible
- Ratio engine and trend analysis components are included
- LLM-assisted analytical insight generation requires separate API configuration

---

# Environment Configuration

Create a `.env` file for API-based functionality:

```env
API_KEY=your_api_key
```

---

# How to Run

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Launch the Main Notebook

Open:

```text
Sheet_classification.ipynb
```


Run notebook cells sequentially.

---

# Included Functionality

The repository includes:

- Financial statement extraction
- Line-item normalization
- Ratio computation
- Trend analysis
- Persistent mapping system
- Custom ratio framework

LLM-assisted insight generation requires external API configuration.
## Example Input

```text
Walmart_financial_report.xlsx
```

# Future Improvements

Planned enhancements include:

- PDF financial statement parsing
- Advanced visualization dashboards
- Automated report generation
- Expanded LLM-assisted analytics
- Sector-specific financial templates
- Cross-company benchmarking
- Time-series forecasting support

---

# License

This repository is intended for academic and research purposes.
