# financial_analytics_system.py
"""
Enhanced financial analytics system with improved formatting, trend analysis, and interactive features.
 Enhanced output formatting with detailed formula substitution
 Comprehensive trend interpretation with financial reasoning
 Interactive Q&A for ratio explanations
 Custom ratio handling with user-defined formulas
Persistent memory for custom ratios
"""
import os
import re
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer, util
import torch
from rapidfuzz import process, fuzz
from config import api_key, endpoint
from datetime import datetime
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger("FinancialAnalyticsSystem")

class FinancialParser:
    """Self-learning financial statement parser"""
    def __init__(self):
        self.config = self._load_config()
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.pattern_manager = self._load_patterns()
        self.hitl_history = self._load_hitl_history()
        
    def _load_config(self):
        """Load configuration with defaults"""
        return {
            'fuzzy_threshold': 80,
            'bert_threshold': 0.68,
            'enable_hitl': False,
            'llm_timeout': 15,
            'hitl_auto_approve_threshold': 0.95
        }
    
    def _load_patterns(self):
        """Load mapping patterns"""
        try:
            with open('mapping_patterns.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'income': {},
                'balance': {},
                'cashflow': {}
            }
    
    def _load_hitl_history(self):
        """Load HITL decisions"""
        try:
            with open('hitl_history.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def normalize_label(self, label: str) -> str:
        """Normalize financial label for matching"""
        clean = re.sub(r'[^\w\s]', '', str(label)).lower()
        return re.sub(r'\s+', ' ', clean).strip()
    
    def extract_years(self, series: pd.Series) -> List[str]:
        """Extract years from series index"""
        years = []
        for idx in series.index:
            year_match = re.search(r'\b(20[2345][0-9])\b', str(idx))
            if year_match:
                years.append(year_match.group(1))
        return sorted(set(years))

class RatioEngine:
    def __init__(self, income_map, balance_map, cashflow_map):
        self.income = income_map.get('data', {})
        self.balance = balance_map.get('data', {})
        self.cashflow = cashflow_map.get('data', {})
        self.market_prices = {}
        self.essential_ratios = self._load_ratios('essential_ratios.json')
        self.custom_ratios = self._load_ratios('custom_ratios.json')
        self.azure_client = self._init_azure_client()
        self.model_name = "gpt-4o-mini"
        self.current_ratio_results = {}
        self.industry = None  
        self.progress_bars = {}
        self.progress_symbols = ["......"]
        self.current_symbol = 0
        self.active_processes = {}
        self.industry_averages = self._load_industry_averages()
        self.display_name_mapping = {
            'eps': 'Earnings Per Share (EPS)',
            'current_ratio': 'Current Ratio',
            'quick_ratio': 'Quick Ratio',
            'gross_margin': 'Gross Margin',
            'net_margin': 'Net Margin',
            'return_on_equity': 'Return on Equity (ROE)',
            'debt_to_equity': 'Debt-to-Equity Ratio',
            'ebitda': 'EBITDA',
            'inventory_turnover': 'Inventory Turnover',
            'asset_turnover': 'Asset Turnover',
            'operating_margin': 'Operating Margin'
        }
        
        self.component_descriptions = {
            'net_income': 'Net Income',
            'shares_basic': 'Weighted Average Shares Outstanding',
            'current_assets': 'Current Assets',
            'current_liabilities': 'Current Liabilities',
            'cash': 'Cash and Cash Equivalents',
            'accounts_receivable': 'Accounts Receivable',
            'gross_profit': 'Gross Profit',
            'total_revenue': 'Total Revenue',
            'operating_income': 'Operating Income',
            'depreciation_amortization': 'Depreciation & Amortization',
            'total_debt': 'Total Debt',
            'shareholders_equity': 'Shareholders\' Equity',
            'interest_expense': 'Interest Expense',
            'market_price': 'Market Price',
            'cost_of_goods_sold': 'Cost of Goods Sold',
            'inventory': 'Inventory',
            'total_assets': 'Total Assets'
        }
        self.key_mapping = self._load_key_mapping()
        self.percent_formatted_ratios = {
            'gross_margin', 'net_margin', 'return_on_equity', 
            'operating_margin', 'ebitda_margin'
        }
         
        self.original_essential_ratios = self._deep_copy_ratios(self.essential_ratios)
        self.original_custom_ratios = self._deep_copy_ratios(self.custom_ratios)
    
    def show_progress(self, process_name: str, message: str):
        """Display animated progress for background processes"""
        if process_name not in self.progress_bars:
            self.progress_bars[process_name] = 0
            self.active_processes[process_name] = message
        
        self.progress_bars[process_name] += 1
        self.current_symbol = (self.current_symbol + 1) % len(self.progress_symbols)
    
        # Build progress display
        display = f"\n{self.progress_symbols[self.current_symbol]} BACKGROUND PROCESSES {self.progress_symbols[self.current_symbol]}"
        for proc, count in self.progress_bars.items():
            dots = "•" * (count % 4) + " " * (3 - (count % 4))
            display += f"\n  {self.active_processes[proc]} {dots}"
    
        print(display, end="\r")

    def start_process(self, process_name: str, message: str):
        """Begin a new background process"""
        self.progress_bars[process_name] = 0
        self.active_processes[process_name] = message
        self.show_progress(process_name, message)

    def complete_process(self, process_name: str):
        """Mark a process as completed"""
        if process_name in self.progress_bars:
            del self.progress_bars[process_name]
            del self.active_processes[process_name]
            print(f"\n✅ {process_name.replace('_', ' ').title()} COMPLETE")
    
    def _load_industry_averages(self):
        """Load industry average ratios from file"""
        try:
            with open('industry_averages.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "technology": {
                    "current_ratio": 1.8,
                    "quick_ratio": 1.5,
                    "debt_to_equity": 0.35,
                    "gross_margin": 0.55,
                    "net_margin": 0.15,
                    "return_on_equity": 0.20,
                    "eps_growth": 0.25
                },
                "manufacturing": {
                    "current_ratio": 1.5,
                    "quick_ratio": 1.0,
                    "debt_to_equity": 0.60,
                    "gross_margin": 0.35,
                    "net_margin": 0.08,
                    "return_on_equity": 0.15,
                    "eps_growth": 0.10
                },
                "retail": {
                    "current_ratio": 1.2,
                    "quick_ratio": 0.8,
                    "debt_to_equity": 0.45,
                    "gross_margin": 0.30,
                    "net_margin": 0.05,
                    "return_on_equity": 0.12,
                    "eps_growth": 0.08
                }
            }

    
    def delete_ratio(self):
        """Allow user to delete a custom ratio"""
        print("\n" + "="*60)
        print("DELETE CUSTOM RATIO")
        print("="*60)
        
        # Collect all custom ratios
        custom_ratios = []
        for category, ratios in self.custom_ratios.items():
            for ratio in ratios:
                # Only allow deletion of custom ratios, not essential ones
                if category != 'essential' or ratio['name'] not in self.essential_ratios:
                    custom_ratios.append({
                        'name': ratio['name'],
                        'category': category,
                        'formula': ratio['formula'],
                        'description': ratio.get('description', '')
                    })
        
        if not custom_ratios:
            print("No custom ratios available for deletion.")
            return False
            
        # Display custom ratios
        print("Custom Ratios Available for Deletion:")
        for i, ratio in enumerate(custom_ratios, 1):
            print(f"{i}. {ratio['name']} ({ratio['category']})")
            print(f"   Formula: {ratio['formula']}")
            print(f"   Description: {ratio['description']}")
        
        try:
            selection = int(input("\nSelect ratio to delete (number): "))
            if 1 <= selection <= len(custom_ratios):
                ratio_to_delete = custom_ratios[selection-1]
                
                # Confirm deletion
                confirm = input(f"\nAre you sure you want to delete '{ratio_to_delete['name']}'? (yes/no): ").lower()
                if confirm in ['y', 'yes']:
                    # Remove from custom ratios
                    for category, ratios in self.custom_ratios.items():
                        # Create a new list without the deleted ratio
                        self.custom_ratios[category] = [
                            r for r in ratios 
                            if r['name'] != ratio_to_delete['name']
                        ]
                    
                    # Remove from essential ratios if it was added there
                    if ratio_to_delete['category'] in self.essential_ratios:
                        self.essential_ratios[ratio_to_delete['category']] = [
                            r for r in self.essential_ratios[ratio_to_delete['category']]
                            if r['name'] != ratio_to_delete['name']
                        ]
                    
                    # Save changes
                    with open('custom_ratios.json', 'w') as f:
                        json.dump(self.custom_ratios, f, indent=4)
                    with open('essential_ratios.json', 'w') as f:
                        json.dump(self.essential_ratios, f, indent=4)
                    
                    print(f"\n✅ '{ratio_to_delete['name']}' has been deleted successfully!")
                    return True
                else:
                    print("Deletion cancelled.")
            else:
                print("Invalid selection.")
        except ValueError:
            print("Please enter a valid number.")
        return False
    
    def reset_ratios(self):
        """Reset all ratios to original definitions"""
        print("\n" + "="*60)
        print("RESET RATIOS TO DEFAULT")
        print("="*60)
        
        confirm = input("Are you sure you want to reset ALL ratios to their original definitions? (yes/no): ").lower()
        if confirm in ['y', 'yes']:
            self.essential_ratios = self._deep_copy_ratios(self.original_essential_ratios)
            self.custom_ratios = self._deep_copy_ratios(self.original_custom_ratios)
            
            # Save the reset ratios to files
            with open('essential_ratios.json', 'w') as f:
                json.dump(self.essential_ratios, f, indent=4)
            with open('custom_ratios.json', 'w') as f:
                json.dump(self.custom_ratios, f, indent=4)
            
            print("\n✅ All ratios have been reset to their original definitions!")
            
            # Recalculate all ratios
            self.calculate_all_ratios()
            return True
        else:
            print("Reset cancelled")
        return False
    
    def reset_single_ratio(self):
        """Reset a single ratio to its original definition"""
        print("\n" + "="*60)
        print("RESET SINGLE RATIO")
        print("="*60)
        
        # Show all available ratios
        print("Available Ratios:")
        all_ratios = []
        for category, ratios in self.essential_ratios.items():
            for ratio in ratios:
                all_ratios.append((ratio['name'], category, "essential"))
                
        for category, ratios in self.custom_ratios.items():
            for ratio in ratios:
                all_ratios.append((ratio['name'], category, "custom"))
        
        # Display ratios with categories
        for i, (name, category, source) in enumerate(all_ratios, 1):
            print(f"{i}. {name} ({category} - {source})")
        
        try:
            selection = int(input("\nSelect ratio to reset (number): "))
            if 1 <= selection <= len(all_ratios):
                ratio_name, category, source = all_ratios[selection-1]
                
                # Find the original ratio
                original_ratio = None
                if source == "essential":
                    original_list = self.original_essential_ratios[category]
                    original_ratio = next((r for r in original_list if r['name'] == ratio_name), None)
                else:
                    original_list = self.original_custom_ratios[category]
                    original_ratio = next((r for r in original_list if r['name'] == ratio_name), None)
                
                if original_ratio:
                    # Find the current ratio
                    if source == "essential":
                        current_list = self.essential_ratios[category]
                        current_ratio = next((r for r in current_list if r['name'] == ratio_name), None)
                    else:
                        current_list = self.custom_ratios[category]
                        current_ratio = next((r for r in current_list if r['name'] == ratio_name), None)
                    
                    if current_ratio:
                        # Reset to original values
                        current_ratio['formula'] = original_ratio['formula']
                        current_ratio['description'] = original_ratio['description']
                        
                        print(f"\n✅ {ratio_name} has been reset to its original definition:")
                        print(f"Formula: {original_ratio['formula']}")
                        print(f"Description: {original_ratio['description']}")
                        
                        # Recalculate all ratios to reflect changes
                        self.calculate_all_ratios()
                        return True
                else:
                    print("Original ratio not found")
            else:
                print("Invalid selection")
        except ValueError:
            print("Please enter a valid number")
        return False
    def _deep_copy_ratios(self, ratios_dict):
        """Create a deep copy of ratios for reset functionality"""
        return json.loads(json.dumps(ratios_dict))
        
    def _init_azure_client(self):
        try:
            return AzureOpenAI(
                api_key=api_key,
                api_version="2023-05-15",
                azure_endpoint=endpoint
            )
        except Exception as e:
            logger.warning(f"LLM unavailable: {e}")
            return None

    def _load_ratios(self, filename: str):
        try:
            with open(filename, 'r') as f:
                ratios = json.load(f)
            
        # Ensure all ratios have descriptions
            for category, ratio_list in ratios.items():
                for ratio in ratio_list:
                    if 'description' not in ratio:
                        ratio['description'] = f"Custom {ratio['name'].replace('_', ' ')} ratio"
                    
            return ratios
        except FileNotFoundError:
            essential = {
                "profitability": [
                    {"name": "gross_margin", "formula": "gross_profit / total_revenue", "description": "Measures profitability after direct costs"},
                    {"name": "net_margin", "formula": "net_income / total_revenue", "description": "Measures overall profitability"},
                    {"name": "return_on_equity", "formula": "net_income / shareholders_equity", "description": "Measures return generated on shareholder investment"},
                    {"name": "operating_margin", "formula": "operating_income / total_revenue", "description": "Measures profitability from core operations"}
                ],
                "liquidity": [
                    {"name": "current_ratio", "formula": "current_assets / current_liabilities", "description": "Measures ability to pay short-term obligations"},
                    {"name": "quick_ratio", "formula": "(cash + accounts_receivable) / current_liabilities", "description": "Measures immediate liquidity without inventory"}
                ],
                "leverage": [
                    {"name": "debt_to_equity", "formula": "total_debt / shareholders_equity", "description": "Measures financial leverage and risk"},
                    {"name": "interest_coverage", "formula": "operating_income / interest_expense", "description": "Measures ability to pay interest expenses"}
                ],
                "efficiency": [
                    {"name": "inventory_turnover", "formula": "cost_of_goods_sold / inventory", "description": "Measures how efficiently inventory is managed"},
                    {"name": "asset_turnover", "formula": "total_revenue / total_assets", "description": "Measures efficiency of asset utilization"}
                ],
                "valuation": [
                    {"name": "eps", "formula": "net_income / shares_basic", "description": "Earnings per common share outstanding"},
                    {"name": "pe_ratio", "formula": "market_price / eps", "description": "Price-to-earnings valuation ratio"}
                ],
                "performance": [
                    {"name": "ebitda", "formula": "operating_income + depreciation_amortization", "description": "Earnings Before Interest, Taxes, Depreciation, and Amortization"}
                ]
            }
            with open(filename, 'w') as f:
                json.dump(essential, f, indent=4)
            return essential

    def _load_key_mapping(self):
        try:
            with open('key_mapping.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _save_key_mapping(self):
        with open('key_mapping.json', 'w') as f:
            json.dump(self.key_mapping, f, indent=4)

    def resolve_key(self, standard_key, available_keys):
        # 1. Check cache
        if standard_key in self.key_mapping:
            return self.key_mapping[standard_key]
        # 2. Fuzzy match
        fuzzy_result = process.extractOne(standard_key, available_keys, scorer=fuzz.token_sort_ratio)
        if fuzzy_result:
            match, score, _ = fuzzy_result
            if score > 80:
                self.key_mapping[standard_key] = match
                self._save_key_mapping()
                return match
        # 3. LLM fallback
        if self.azure_client:
            prompt = f"Map the standard financial term '{standard_key}' to the closest match from this list: {available_keys}. Only return the best match."
            try:
                response = self.azure_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                llm_match = response.choices[0].message.content.strip()
                if llm_match in available_keys:
                    self.key_mapping[standard_key] = llm_match
                    self._save_key_mapping()
                    return llm_match
            except Exception as e:
                logger.warning(f"LLM mapping failed: {e}")
        # 4. Not found
        return None

    def get_available_years(self) -> List[str]:
        all_years = set()
        for statement in [self.income, self.balance, self.cashflow]:
            for series in statement.values():
                if hasattr(series, 'index'):
                    for idx in series.index:
                        year_match = re.search(r'\b(20[2345][0-9])\b', str(idx))
                        if year_match:
                            all_years.add(year_match.group(1))
        return sorted(list(all_years))

    def get_value(self, key: str, year: str) -> Optional[float]:
        if key == 'shares_basic':
            value = self._get_shares_outstanding(year)
            if value is not None:
                return value
        if key == 'eps':
            return None
        for statement in [self.income, self.balance, self.cashflow]:
            available_keys = list(statement.keys())
            mapped_key = self.resolve_key(key, available_keys)
            if mapped_key and mapped_key in statement:
                series = statement[mapped_key]
                for idx in series.index:
                    if str(year) in str(idx):
                        value = series[idx]
                        if pd.notna(value) and value != '':
                            try:
                                return float(value)
                            except (ValueError, TypeError):
                                pass
        return None
    def _get_shares_outstanding(self, year: str) -> Optional[float]:
        """Special handling for shares outstanding which is critical for EPS"""
        # First try standard lookup
        for statement in [self.income, self.balance, self.cashflow]:
            available_keys = list(statement.keys())
            mapped_key = self.resolve_key('shares_basic', available_keys)
            if mapped_key and mapped_key in statement:
                series = statement[mapped_key]
                for idx in series.index:
                    if str(year) in str(idx):
                        value = series[idx]
                        if pd.notna(value) and value != '':
                            try:
                                return float(value)
                            except (ValueError, TypeError):
                                pass
        
        # If not found, prompt the user
        print(f"\n{'='*60}")
        print(f"SHARES OUTSTANDING REQUIRED FOR {year}")
        print(f"{'='*60}")
        print("Earnings Per Share (EPS) and P/E ratio require shares outstanding.")
        print("Please provide the weighted average shares outstanding.")
        
        while True:
            shares = input(f"\nEnter shares outstanding for {year} (or press Enter to skip): ").strip()
            if not shares:
                return None
            try:
                shares_val = float(shares)
                # Store for future use
                if 'shares_basic' not in self.income:
                    self.income['shares_basic'] = {}
                self.income['shares_basic'][year] = shares_val
                return shares_val
            except ValueError:
                print("Invalid input. Please enter a numeric value.")


    
    def get_market_price(self, year: str) -> Optional[float]:
        """Get market price for a specific year, prompting user if missing"""
        # First check if we already have it
        if year in self.market_prices:
            return self.market_prices[year]
            
        # Try to find in data sources
        for statement in [self.income, self.balance, self.cashflow]:
            available_keys = list(statement.keys())
            mapped_key = self.resolve_key('market_price', available_keys)
            if mapped_key and mapped_key in statement:
                series = statement[mapped_key]
                for idx in series.index:
                    if str(year) in str(idx):
                        value = series[idx]
                        if pd.notna(value) and value != '':
                            try:
                                price = float(value)
                                self.market_prices[year] = price
                                return price
                            except (ValueError, TypeError):
                                pass
        
        # If not found, prompt the user
        print(f"\n{'='*60}")
        print(f"MARKET PRICE REQUIRED FOR VALUATION RATIOS ({year})")
        print(f"{'='*60}")
        print("Valuation ratios require the market price per share.")
        print("Please provide the closing stock price at the end of the fiscal year.")
        print("If you don't know the price, press Enter to skip valuation ratios.")
        
        while True:
            price_input = input(f"\nEnter market price per share for {year} (or press Enter to skip): ").strip()
            if not price_input:
                self.market_prices[year] = None
                return None
            try:
                price_val = float(price_input)
                self.market_prices[year] = price_val
                return price_val
            except ValueError:
                print("Invalid input. Please enter a numeric value.")
                
    def calculate_ratio(self, formula: str, year: str) -> Tuple[Optional[float], str, Dict[str, float]]:
        

        variables = re.findall(r'\b(\w+)\b', formula)
        context = {}
        values_used = {}
        formula_display = formula
        
        # Special handling for ratios that depend on other ratios
        ratio_dependencies = {
        'eps': 'eps',  # Map variable name to ratio name
        'book_value_per_share': 'book_value_per_share'
    }
        
        # Check if this ratio depends on another ratio
       # dependency = None
        for var in variables:
            if var in ratio_dependencies:
                dep_ratio_name = ratio_dependencies[var]
            # First check if we have it in current results
                if self.current_ratio_results.get(dep_ratio_name, {}).get('values', {}).get(year):
                    dep_value = self.current_ratio_results[dep_ratio_name]['values'][year]
                    context[var] = dep_value
                # Format for display
                    dep_display_name = self.get_ratio_display_name(dep_ratio_name)
                    formula_display = formula_display.replace(
                    var, 
                    f"{dep_display_name} ({self.format_ratio_value(dep_ratio_name, dep_value)})"
                )
                    values_used[var] = dep_value
                else:
                    # Calculate the dependency if not available
                    dep_ratio = next((r for cat in self.essential_ratios.values() for r in cat if r['name'] == dep_ratio_name), None)
                    if dep_ratio:
                        dep_value, dep_formula, dep_components = self.calculate_ratio(dep_ratio["formula"], year)
                        if dep_value is None:
                            return None, formula, {}
                        context[var] = dep_value
                        values_used.update(dep_components)
                    # Format dependency display
                        dep_display_name = self.get_ratio_display_name(dep_ratio_name)
                        formula_display = formula_display.replace(
                        var, 
                        f"{dep_display_name} ({self.format_ratio_value(dep_ratio_name, dep_value)})"
                    )
        
        # Process all variables
        for var in variables:
            if var in context:
                continue  # Skip if already handled
                
            # Special case for market price
            if var == 'market_price':
                value = self.get_market_price(year)
                if value is None:
                    return None, formula, {}
            else:
                value = self.get_value(var, year)
                
            if value is not None:
                context[var] = value
                display_name = self.component_descriptions.get(var, var.replace('_', ' ').title())
                
                # Format market price specially
                if var == 'market_price':
                    formula_display = formula_display.replace(
                        var, 
                        f"{display_name} (${value:,.2f})"
                    )
                else:
                    formula_display = formula_display.replace(
                        var, 
                        f"{display_name} ({value:,.0f})"
                    )
                    
                values_used[var] = value
            else:
                # If any required value is missing, skip calculation
                return None, formula, {}
                
        try:
            # Special handling for P/E ratio
            if 'pe_ratio' in formula:
                # Ensure we have valid market price and EPS
                if 'market_price' not in context or 'eps' not in context:
                    return None, formula, {}
    
                # Calculate P/E using the already-computed EPS from context
                result = context['market_price'] / context['eps']
    
                    # Format EPS value in the formula display
                eps_display = self.format_ratio_value('eps', context['eps'])
                formula_display = formula_display.replace(
        f"Eps ({context['eps']:,.0f})", 
        f"EPS ({eps_display})"
    )
    
                return round(result, 4), formula_display, values_used
            

            
            # General calculation for other ratios
            result = eval(formula, {"__builtins__": None}, context)
            return round(result, 4), formula_display, values_used
        except ZeroDivisionError:
            logger.warning(f"Division by zero in formula: {formula} for {year}")
            return None, formula, {}
        except Exception as e:
            logger.error(f"Calculation error: {e}")
            return None, formula, {}




    def calculate_all_ratios(self) -> Dict:
        
        self.current_ratio_results = {}
        years = self.get_available_years()
        results = {}
        for category, ratios in self.essential_ratios.items():
            for ratio in ratios:
                ratio_name = ratio["name"]
                results[ratio_name] = {
                    "values": {},
                    "formulas": {},
                    "components": {},
                    "trend": {},
                    "description": ratio.get("description", ""),
                    "category": category,
                    "formula": ratio["formula"]
                }
                for year in years:
                    if 'market_price' in ratio["formula"]:
                        market_price = self.get_market_price(year)
                        if market_price is None:
                            continue  # Skip this year for valuation ratios
                    value, formula_display, components = self.calculate_ratio(ratio["formula"], year)
                    if value is not None:
                        results[ratio_name]["values"][year] = value
                        results[ratio_name]["formulas"][year] = formula_display
                        results[ratio_name]["components"][year] = components
                # Only add trend if we have at least 2 years of data
                if len(results[ratio_name]["values"]) >= 2:
                    results[ratio_name]["trend"] = self._analyze_trend(
                        results[ratio_name]["values"]
                    )
        for ratio in self.custom_ratios.get("custom", []):
            ratio_name = ratio["name"]
            results[ratio_name] = {
                "values": {},
                "formulas": {},
                "components": {},
                "trend": {},
                "description": ratio.get("description", ""),
                "category": "custom",
                "formula": ratio["formula"]
            }
            for year in years:
                value, formula_display, components = self.calculate_ratio(ratio["formula"], year)
                if value is not None:
                    results[ratio_name]["values"][year] = value
                    results[ratio_name]["formulas"][year] = formula_display
                    results[ratio_name]["components"][year] = components
            if results[ratio_name]["values"]:
                results[ratio_name]["trend"] = self._analyze_trend(
                    results[ratio_name]["values"]
                )
        self.current_ratio_results = results 
        
        return results

    def _analyze_trend(self, values: Dict[str, float]) -> Dict:
        if not values or len(values) < 2:
            return {
                "direction": "insufficient_data",
                "analysis": "Insufficient data for trend analysis",
                "change_pct": 0
            }
        years = sorted(values.keys())
        values_list = [values[year] for year in years]
        start_value = values_list[0]
        end_value = values_list[-1]
        change_pct = ((end_value - start_value) / abs(start_value)) * 100 if start_value != 0 else 0
        
        # Calculate year-over-year changes
        yoy_changes = {}
        for i in range(1, len(years)):
            yoy_change = ((values_list[i] - values_list[i-1]) / abs(values_list[i-1])) * 100 if values_list[i-1] != 0 else 0
            yoy_changes[f"{years[i-1]}-{years[i]}"] = yoy_change
        
        if all(values_list[i] <= values_list[i+1] for i in range(len(values_list)-1)):
            trend = "increasing"
        elif all(values_list[i] >= values_list[i+1] for i in range(len(values_list)-1)):
            trend = "decreasing"
        else:
            trend = "fluctuating"
        return {
            "direction": trend,
            "change_pct": change_pct,
            "start_year": years[0],
            "end_year": years[-1],
            "start_value": start_value,
            "end_value": end_value,
            "yoy_changes": yoy_changes
        }

    def get_ratio_display_name(self, ratio_name: str) -> str:
        return self.display_name_mapping.get(ratio_name, ratio_name.replace('_', ' ').title())

    def format_ratio_value(self, ratio_name: str, value: float) -> str:
        """Format ratio values appropriately based on ratio type"""
        if ratio_name in self.percent_formatted_ratios:
            return f"{value*100:.2f}%"
        elif ratio_name == 'eps':
            return f"{value:.4f}"
        elif ratio_name == 'pe_ratio':
            return f"{value:.2f}" if value > 1 else f"{value:.4f}"
        return f"{value:.2f}"

    def display_ratio(self, ratio_name: str, ratio_data: Dict):
        display_name = self.get_ratio_display_name(ratio_name)
        years = sorted(ratio_data["values"].keys())
        print(f"\n{'=' * 60}")
        print(f"RESULTS FOR {display_name.upper()}")
        print(f"{'=' * 60}")
        print(f"Description: {ratio_data['description']}\n")

        if ratio_name == 'pe_ratio' and ratio_data["values"]:
            print("\nDEBUG INFO (P/E Ratio Calculation):")
            for year, value in ratio_data["values"].items():
                components = ratio_data["components"][year]
                print(f"{year}: Market Price = {components.get('market_price')}, "
                      f"EPS = {components.get('eps')}, "
                      f"Calculated P/E = {value}")
                
        # Display formula with actual values for each year
        table_data = []
        for year in years:
            value = ratio_data["values"][year]
            formatted_value = self.format_ratio_value(ratio_name, value)
            
            # Format the formula with actual values
            formula_parts = ratio_data["formulas"][year].split('=')
            if len(formula_parts) > 1:
                calc_str = formula_parts[1].strip()
            else:
                calc_str = ratio_data["formulas"][year]
                
            # Add calculation result to formula display
            calc_str += f" = {formatted_value}"
            
            table_data.append([year, calc_str])
        
        # Print table using tabulate
        print(tabulate(table_data, 
                       headers=['Year', 'Calculation'], 
                       tablefmt='grid', 
                       colalign=('left', 'left')))
        print()
        
        trend = ratio_data.get("trend", {})
        if trend.get("direction") and trend["direction"] != "insufficient_data":
            self.display_trend_interpretation(ratio_name, trend)
        else:
            print("Insufficient data for trend analysis\n")
        print(f"{'=' * 60}\n")

    def display_trend_interpretation(self, ratio_name: str, trend_data: Dict):
        direction = trend_data.get("direction", "insufficient_data")
        change_pct = trend_data.get("change_pct", 0)
        start_year = trend_data.get("start_year", "")
        end_year = trend_data.get("end_year", "")
        start_value = trend_data.get("start_value", 0)
        end_value = trend_data.get("end_value", 0)
        yoy_changes = trend_data.get("yoy_changes", {})
        
        # Format values appropriately
        start_value_fmt = self.format_ratio_value(ratio_name, start_value)
        end_value_fmt = self.format_ratio_value(ratio_name, end_value)
        
        # Create YoY changes table
        yoy_table = []
        for period, change in yoy_changes.items():
            yoy_table.append([period, f"{change:+.2f}%"])
        
        interpretations = {
            "eps": {
                "increasing": f"EPS increased from {start_value_fmt} in {start_year} to {end_value_fmt} in {end_year} (+{change_pct:.2f}%), indicating improved profitability per share. This suggests the company is generating more earnings relative to its outstanding shares, which could be due to revenue growth, cost management, or share buybacks.",
                "decreasing": f"EPS decreased from {start_value_fmt} in {start_year} to {end_value_fmt} in {end_year} ({change_pct:+.2f}%), signaling reduced profitability per share. This could result from declining sales, rising costs, or dilution from share issuance.",
                "fluctuating": f"EPS fluctuated between {min(start_value, end_value):.4f} and {max(start_value, end_value):.4f} from {start_year} to {end_year}, showing inconsistent earnings performance. This volatility may indicate sensitivity to market conditions or operational instability."
            },
            "current_ratio": {
                "increasing": f"Current ratio strengthened from {start_value_fmt} in {start_year} to {end_value_fmt} in {end_year} (+{change_pct:.2f}%), indicating improved short-term liquidity. This suggests the company is better positioned to meet its short-term obligations, potentially due to better working capital management or increased cash reserves.",
                "decreasing": f"Current ratio weakened from {start_value_fmt} in {start_year} to {end_value_fmt} in {end_year} ({change_pct:+.2f}%), signaling potential liquidity challenges. This could result from increased debt obligations, declining sales, or poor inventory management. A ratio below 1.0 indicates potential difficulty meeting short-term liabilities.",
                "fluctuating": f"Current ratio fluctuated between {min(start_value, end_value):.2f} and {max(start_value, end_value):.2f} from {start_year} to {end_year}, showing inconsistent liquidity management. This may indicate seasonal business patterns or varying approaches to working capital management."
            },
            "gross_margin": {
                "increasing": f"Gross margin expanded from {start_value_fmt} in {start_year} to {end_value_fmt} in {end_year} (+{change_pct:.2f}%), showing improved cost efficiency or pricing power. This enhancement in core profitability could result from product mix improvements, cost reductions, or successful pricing strategies.",
                "decreasing": f"Gross margin contracted from {start_value_fmt} in {start_year} to {end_value_fmt} in {end_year} ({change_pct:+.2f}%), indicating cost pressures or competitive challenges. This could be due to rising input costs, increased competition forcing price reductions, or inefficient production processes.",
                "fluctuating": f"Gross margin fluctuated between {min(start_value, end_value)*100:.2f}% and {max(start_value, end_value)*100:.2f}% from {start_year} to {end_year}, suggesting inconsistent control over production costs or pricing stability. This volatility may indicate sensitivity to commodity prices or competitive pressures."
            },
            "debt_to_equity": {
                "increasing": f"Debt-to-equity ratio increased from {start_value_fmt} in {start_year} to {end_value_fmt} in {end_year} (+{change_pct:.2f}%), showing higher financial leverage. While this can boost ROE through leverage, it also increases financial risk, especially in rising interest rate environments. A ratio above 2.0 often signals high financial risk.",
                "decreasing": f"Debt-to-equity ratio decreased from {start_value_fmt} in {start_year} to {end_value_fmt} in {end_year} ({change_pct:+.2f}%), indicating a stronger equity position and reduced financial risk. This conservative capital structure provides more resilience during economic downturns but may limit growth potential.",
                "fluctuating": f"Debt-to-equity ratio fluctuated between {min(start_value, end_value):.2f} and {max(start_value, end_value):.2f} from {start_year} to {end_year}, reflecting changing capital structure decisions. This may indicate strategic shifts between debt and equity financing."
            },
            "inventory_turnover": {
                "increasing": f"Inventory turnover increased from {start_value_fmt} in {start_year} to {end_value_fmt} in {end_year} (+{change_pct:.2f}%), indicating improved inventory management efficiency. Higher turnover typically means the company is converting inventory to sales more quickly, reducing holding costs and obsolescence risk.",
                "decreasing": f"Inventory turnover decreased from {start_value_fmt} in {start_year} to {end_value_fmt} in {end_year} ({change_pct:+.2f}%), suggesting slower inventory movement. This could indicate overstocking, declining sales, or obsolete inventory, which ties up working capital and increases storage costs.",
                "fluctuating": f"Inventory turnover fluctuated between {min(start_value, end_value):.2f} and {max(start_value, end_value):.2f} from {start_year} to {end_year}, showing inconsistent inventory management. This may reflect seasonal demand patterns or changing product mix."
            },
            "default": {
                "increasing": f"The ratio shows a positive trend, increasing from {start_value_fmt} in {start_year} to {end_value_fmt} in {end_year} (+{change_pct:.2f}%). This improvement suggests enhanced financial performance in this area.",
                "decreasing": f"The ratio shows a concerning trend, decreasing from {start_value_fmt} in {start_year} to {end_value_fmt} in {end_year} ({change_pct:+.2f}%). This decline may indicate emerging challenges that require management attention.",
                "fluctuating": f"The ratio shows volatility, fluctuating between {min(start_value, end_value):.2f} and {max(start_value, end_value):.2f} from {start_year} to {end_year}. This inconsistency may reflect changing business conditions or operational variability."
            }
        }
        if ratio_name == 'pe_ratio':
            latest_value = trend_data.get("end_value", 0)
            if latest_value == 0:
                print("TREND ANALYSIS:")
                print("⚠️ P/E ratio calculated as 0 - this typically indicates:")
                print("- Earnings per share is zero (company broke even)")
                print("- Negative earnings (losses)")
                print("- Missing shares outstanding data")
                print("Please verify financial data inputs")
                return
        ratio_interpretation = interpretations.get(ratio_name, interpretations["default"])
        print("TREND ANALYSIS:")
        print(ratio_interpretation.get(direction, ratio_interpretation["increasing"]))
        
        # Show YoY changes if available
        if yoy_table:
            print("\nYear-over-Year Changes:")
            print(tabulate(yoy_table, headers=['Period', 'Change'], tablefmt='grid'))
            print()

    def interpret_custom_request(self, prompt: str) -> Dict:
        available_keys = list(self.income.keys()) + list(self.balance.keys()) + list(self.cashflow.keys())
        system_prompt = f"""
        You are a financial ratio analysis assistant. The user has requested: "{prompt}"
        Available financial keys: {', '.join(available_keys)}
        Requirements:
        1. Identify the ratio name (in snake_case)
        2. Use only available keys to create the formula
        3. Return JSON with ratio name, formula, and description
        4. Example: "quick_ratio = (cash + accounts_receivable) / current_liabilities"
        """
        try:
            response = self.azure_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": system_prompt}],
                temperature=0.3
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return {
                "name": "custom_ratio",
                "formula": "",
                "description": "Custom ratio created from user input"
            }

    def answer_ratio_question(self, ratio_name: str, ratio_data: Dict, question: str) -> str:
        if not self.azure_client:
            return "LLM service is currently unavailable"
        try:
            years = sorted(ratio_data["values"].keys())
            values_str = ", ".join([f"{year}: {ratio_data['values'][year]:.2f}" for year in years])
            if ratio_data["formulas"]:
                formula_example = next(iter(ratio_data["formulas"].values()))
            else:
                formula_example = "N/A"
            system_prompt = f"""
            You are a financial analyst AI. Answer the user's question about the {ratio_name} ratio.
            Ratio Context:
            - Description: {ratio_data.get('description', 'N/A')}
            - Formula: {formula_example}
            - Values: {values_str}
            - Trend: {ratio_data.get('trend', {}).get('analysis', 'N/A')}
            User Question: "{question}"
            Guidelines:
            1. Be precise and reference specific numbers from the data
            2. Explain in clear business terms, avoiding jargon when possible
            3. Highlight key drivers if applicable
            4. Provide actionable recommendations where appropriate
            5. Keep response under 5 sentences
            """
            response = self.azure_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": system_prompt}],
                temperature=0.5
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM Q&A error: {e}")
            return "Could not generate answer. Please try a different question."

    def handle_custom_ratio_request(self, prompt: str):
        try:
            # First try to get formula from LLM
            llm_result = self.interpret_custom_request(prompt)
            ratio_name = llm_result["name"]
            base_formula = llm_result.get("formula", "")
            description = llm_result.get("description", "Custom financial ratio")
            
            if base_formula:
                print(f"\nLLM-suggested formula: {ratio_name} = {base_formula}")
                print(f"Description: {description}")
                choice = input("\nUse this formula? (yes/no): ").lower()
                if choice in ['y', 'yes']:
                    formula = base_formula
                    custom_name = ratio_name
                    custom_desc = description
                else:
                    formula, custom_name, custom_desc = self.build_custom_ratio_manually()
            else:
                formula, custom_name, custom_desc = self.build_custom_ratio_manually()
            
            # Calculate and display the ratio
            self.calculate_and_display_custom_ratio(custom_name, formula, custom_desc)
            
            # Save if requested
            if input("\nSave this ratio for future use? (yes/no): ").lower() in ['y', 'yes']:
                category = self.select_ratio_category()
                self.save_custom_ratio(custom_name, formula, custom_desc, category)
                print(f"✅ {custom_name} saved to {category} ratios")
                
        except Exception as e:
            print(f"Could not process request: {e}")
            import traceback
            traceback.print_exc()
    
    def build_custom_ratio_manually(self) -> Tuple[str, str, str]:
        """Guide user through manual custom ratio creation"""
        print("\n" + "="*60)
        print("CUSTOM RATIO BUILDER")
        print("="*60)
        print("We'll help you build a custom financial ratio step-by-step")
        
        # Get ratio name and description
        custom_name = input("\nEnter a name for your custom ratio (snake_case): ").strip()
        if not custom_name:
            custom_name = "custom_ratio"
        custom_desc = input("Enter a description for this ratio: ").strip()
        if not custom_desc:
            custom_desc = "Custom financial ratio"
        
        # Initialize formula building
        formula_parts = []
        selected_components = []
        
        # Available statements
        statements = {
            "income": ("Income Statement", self.income),
            "balance": ("Balance Sheet", self.balance),
            "cashflow": ("Cash Flow Statement", self.cashflow)
        }
        
        while True:
            print("\nCurrent formula: " + " ".join(formula_parts) if formula_parts else "Empty")
            print("\nOptions:")
            print("1. Add component from financial statements")
            print("2. Add operator (+, -, *, /)")
            print("3. Add constant number")
            print("4. Add parentheses")
            print("5. Review and finish")
            print("6. Cancel and exit")
            
            choice = input("\nSelect an option: ").strip()
            
            if choice == '1':  # Add financial component
                comp = self.select_financial_component(statements)
                if comp:
                    formula_parts.append(comp)
                    selected_components.append(comp)
                    print(f"Added: {comp}")
                    
            elif choice == '2':  # Add operator
                operator = self.select_operator()
                if operator:
                    formula_parts.append(operator)
                    print(f"Added: {operator}")
                    
            elif choice == '3':  # Add constant
                constant = input("Enter a numeric constant: ").strip()
                try:
                    float(constant)
                    formula_parts.append(constant)
                    print(f"Added: {constant}")
                except ValueError:
                    print("Invalid number. Please try again.")
                    
            elif choice == '4':  # Add parentheses
                side = input("Add (1) Open parenthesis '(' or (2) Close parenthesis ')'? ").strip()
                if side == '1':
                    formula_parts.append('(')
                    print("Added: (")
                elif side == '2':
                    formula_parts.append(')')
                    print("Added: )")
                else:
                    print("Invalid choice")
                    
            elif choice == '5':  # Finish
                if not formula_parts:
                    print("Formula is empty! Please add components")
                    continue
                    
                formula = " ".join(formula_parts)
                print(f"\nFinal formula: {formula}")
                confirm = input("Use this formula? (yes/no): ").lower()
                if confirm in ['y', 'yes']:
                    return formula, custom_name, custom_desc
                else:
                    continue
                    
            elif choice == '6':  # Cancel
                print("Custom ratio creation cancelled")
                return "", "", ""
                
            else:
                print("Invalid option")

    def select_financial_component(self, statements: Dict) -> str:
        """Guide user to select a financial component from statements"""
        while True:
            print("\nSelect financial statement:")
            print("1. Income Statement")
            print("2. Balance Sheet")
            print("3. Cash Flow Statement")
            print("4. Back to main menu")
            
            choice = input("\nSelect statement: ").strip()
            
            if choice == '1':
                return self.select_from_statement("income", statements["income"][1])
            elif choice == '2':
                return self.select_from_statement("balance", statements["balance"][1])
            elif choice == '3':
                return self.select_from_statement("cashflow", statements["cashflow"][1])
            elif choice == '4':
                return ""
            else:
                print("Invalid choice")

    def select_from_statement(self, st_type: str, statement: Dict) -> str:
        """Display keys from a financial statement and let user select"""
        print(f"\nAvailable items in {st_type.replace('_', ' ').title()}:")
        keys = list(statement.keys())
        
        # Group keys by similarity to standard terms
        standard_terms = [
            'revenue', 'income', 'profit', 'assets', 'liabilities', 
            'equity', 'cash', 'operating', 'investing', 'financing'
        ]
        
        grouped = {term: [] for term in standard_terms}
        grouped['other'] = []
        
        for key in keys:
            matched = False
            for term in standard_terms:
                if term in key.lower():
                    grouped[term].append(key)
                    matched = True
                    break
            if not matched:
                grouped['other'].append(key)
        
        # Display grouped keys
        idx = 1
        options = []
        for category, items in grouped.items():
            if items:
                print(f"\n{category.title()}:")
                for item in items:
                    print(f"{idx}. {item}")
                    options.append(item)
                    idx += 1
        
        if not options:
            print("No items available in this statement")
            return ""
        
        print("\n0. Back to statement selection")
        
        while True:
            try:
                selection = input("\nSelect an item number: ").strip()
                if selection == '0':
                    return ""
                num = int(selection)
                if 1 <= num <= len(options):
                    return options[num-1]
                print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number")

    def select_operator(self) -> str:
        """Let user select a mathematical operator"""
        operators = {
            '1': '+',
            '2': '-',
            '3': '*',
            '4': '/',
            '5': '(',
            '6': ')'
        }
        
        print("\nSelect operator:")
        print("1. Addition (+)")
        print("2. Subtraction (-)")
        print("3. Multiplication (*)")
        print("4. Division (/)")
        print("5. Open Parenthesis (")
        print("6. Close Parenthesis )")
        
        choice = input("\nSelect operator: ").strip()
        return operators.get(choice, "")

    def select_ratio_category(self) -> str:
        """Let user select a category for the custom ratio"""
        categories = {
            '1': 'profitability',
            '2': 'liquidity',
            '3': 'leverage',
            '4': 'efficiency',
            '5': 'valuation',
            '6': 'performance',
            '7': 'custom'
        }
        
        print("\nSelect category for this ratio:")
        print("1. Profitability")
        print("2. Liquidity")
        print("3. Leverage")
        print("4. Efficiency")
        print("5. Valuation")
        print("6. Performance")
        print("7. Custom")
        
        choice = input("\nSelect category: ").strip()
        return categories.get(choice, 'custom')

    def calculate_and_display_custom_ratio(self, name: str, formula: str, description: str):
        """Calculate and display a custom ratio"""
        ratio_data = {
            "values": {},
            "formulas": {},
            "components": {},
            "trend": {},
            "description": description,
            "category": "custom",
            "formula": formula
        }
        
        years = self.get_available_years()
        for year in years:
            value, formula_display, components = self.calculate_ratio(formula, year)
            if value is not None:
                ratio_data["values"][year] = value
                ratio_data["formulas"][year] = formula_display
                ratio_data["components"][year] = components
        
        if ratio_data["values"]:
            ratio_data["trend"] = self._analyze_trend(ratio_data["values"])
        
        self.display_ratio(name, ratio_data)
        self.interactive_qa(name, ratio_data)

    def suggest_custom_ratio_formula(self, prompt: str) -> Optional[Dict]:
        """Suggest a formula for a custom ratio request"""
        common_ratios = {
            "quick ratio": {
                "name": "quick_ratio",
                "formula": "(cash + accounts_receivable) / current_liabilities",
                "description": "Measures immediate liquidity without inventory"
            },
            "operating cash flow ratio": {
                "name": "operating_cash_flow_ratio",
                "formula": "operating_cash_flow / current_liabilities",
                "description": "Measures ability to pay short-term liabilities with operating cash"
            },
            "return on assets": {
                "name": "return_on_assets",
                "formula": "net_income / total_assets",
                "description": "Measures how efficiently assets generate profit"
            },
            "days sales outstanding": {
                "name": "days_sales_outstanding",
                "formula": "(accounts_receivable / total_revenue) * 365",
                "description": "Measures average collection period for receivables"
            }
        }
        
        # Fuzzy match against common ratios
        match, score = process.extractOne(prompt, common_ratios.keys(), scorer=fuzz.token_sort_ratio)
        if score > 75:
            return common_ratios[match]
        
        return None

    def handle_unknown_industry(self, industry: str) -> bool:
        """Handle user-provided industry not in our database"""
        print(f"\n⚠️ '{industry.title()}' is not currently in our industry database.")
        print("We can still provide analysis, but industry comparisons won't be available.")
        print("\nOptions:")
        print("1. Select the closest matching industry")
        print("2. Continue without industry context")
        print("3. Enter custom industry averages")
        
        choice = input("\nSelect an option: ").strip()
        
        if choice == '1':
            print("\nAvailable industries:")
            for i, known_industry in enumerate(self.industry_averages.keys(), 1):
                print(f"{i}. {known_industry.title()}")
            
            try:
                selection = int(input("\nSelect the closest matching industry (number): "))
                industries = list(self.industry_averages.keys())
                if 1 <= selection <= len(industries):
                    self.industry = industries[selection-1]
                    print(f"Using '{self.industry.title()}' as the closest match")
                    return True
            except ValueError:
                print("Invalid selection. Continuing without industry context.")
        
        elif choice == '3':
            print("\nPlease provide benchmark averages for key ratios:")
            print("(Enter values as decimals, e.g., 0.15 for 15%)")
            
            custom_averages = {}
            try:
                custom_averages['current_ratio'] = float(input("Current Ratio: ").strip() or "0")
                custom_averages['quick_ratio'] = float(input("Quick Ratio: ").strip() or "0")
                custom_averages['debt_to_equity'] = float(input("Debt-to-Equity: ").strip() or "0")
                custom_averages['gross_margin'] = float(input("Gross Margin: ").strip() or "0")
                custom_averages['net_margin'] = float(input("Net Margin: ").strip() or "0")
                custom_averages['return_on_equity'] = float(input("Return on Equity: ").strip() or "0")
                custom_averages['eps_growth'] = float(input("EPS Growth Rate: ").strip() or "0")
                
                # Add to database
                self.industry_averages[industry.lower()] = custom_averages
                self.industry = industry.lower()
                
                # Save to file for future use
                with open('industry_averages.json', 'w') as f:
                    json.dump(self.industry_averages, f, indent=4)
                
                print(f"\n✅ Custom benchmarks for '{industry.title()}' saved successfully!")
                return True
            except ValueError:
                print("Invalid input. Must be numeric values. Continuing without industry context.")
        
        return False

    def set_industry(self, industry: str):
        """Set the industry context with flexible handling"""
        if industry.lower() in self.industry_averages:
            self.industry = industry.lower()
            print(f"Industry set to: {industry.title()}")
            return True
        
        return self.handle_unknown_industry(industry)

    def generate_contextual_questions(self, ratio_name: str, ratio_data: Dict) -> List[str]:
        """Generate context-aware questions based on ratio values and trends"""
        questions = []
        latest_year = max(ratio_data["values"].keys()) if ratio_data["values"] else ""
        latest_value = ratio_data["values"].get(latest_year, 0)
        
        # Always include these fundamental questions
        questions.append("What are the components of this ratio?")
        questions.append("How can a company improve this ratio?")
        questions.append("What factors influence this ratio?")
        
        # Add context-specific questions
        if ratio_name == "eps":
            if latest_value < 10:  # Low EPS threshold
                questions.append("Why is the EPS relatively low?")
            elif latest_value > 50:  # High EPS threshold
                questions.append("What's driving the high EPS?")
            else:
                questions.append("Is this EPS growth sustainable?")
                
            questions.append("How does buyback activity affect EPS?")
            
        elif ratio_name == "current_ratio":
            if latest_value < 1.0:
                questions.append("Why is the current ratio below 1.0?")
            elif latest_value < 1.5:
                questions.append("How can the company improve its liquidity position?")
            else:
                questions.append("Is excess liquidity being managed efficiently?")
        
        # Add industry comparison if available
        if self.industry:
            questions.append(f"How does this compare to {self.industry.title()} industry averages?")
        else:
            questions.append("How does this compare to industry averages?")
            
        # Add trend-based questions
        if "trend" in ratio_data:
            trend = ratio_data["trend"]["direction"]
            if trend == "increasing":
                questions.append("What's driving the improvement in this ratio?")
            elif trend == "decreasing":
                questions.append("What's causing the decline in this ratio?")
            elif trend == "fluctuating":
                questions.append("What's causing the volatility in this ratio?")
        
        return questions

    def interactive_qa(self, ratio_name: str, ratio_data: Dict):
        display_name = self.get_ratio_display_name(ratio_name)
        print(f"\nASK QUESTIONS ABOUT {display_name.upper()} (type 'exit' to finish)")
        print(f"{'=' * 60}")
        
        # Generate context-aware questions
        questions = self.generate_contextual_questions(ratio_name, ratio_data)
        
        print("\nSuggested questions based on your data:")
        for i, q in enumerate(questions, 1):
            print(f"{i}. {q}")
        print()
        
        while True:
            question = input("\nYour question: ").strip()
            if question.lower() in ['exit', 'quit', '']:
                break
                
            # Handle industry comparison questions
            if "industry average" in question.lower() or "industry comparison" in question.lower():
                answer = self.handle_industry_question(ratio_name, ratio_data, question)
            else:
                answer = self.answer_ratio_question(ratio_name, ratio_data, question)
                
            print(f"\n{'ANALYST INSIGHT':-^60}")
            print(f"{answer}\n{'-'*60}\n")

    def handle_industry_question(self, ratio_name: str, ratio_data: Dict, question: str) -> str:
        """Special handling for industry comparison questions"""
        if not self.industry:
            # Prompt user to set industry first
            print("\nTo provide industry comparison, please specify your industry.")
            print("Available industries: " + ", ".join(self.industry_averages.keys()))
            
            while True:
                industry = input("\nEnter your industry (or press Enter to skip): ").strip().lower()
                if not industry:
                    return "Industry comparison skipped"
                    
                if industry in self.industry_averages:
                    self.industry = industry
                    break
                else:
                    print(f"Unknown industry. Available options: {', '.join(self.industry_averages.keys())}")
        
        # Get industry average
        industry_avg = self.industry_averages[self.industry].get(ratio_name)
        if not industry_avg:
            return f"No industry average available for {ratio_name} in {self.industry.title()} industry"
        
        # Get latest ratio value
        latest_year = max(ratio_data["values"].keys()) if ratio_data["values"] else ""
        latest_value = ratio_data["values"].get(latest_year, 0)
        
        # Format comparison
        comparison = ""
        if ratio_name in self.percent_formatted_ratios:
            industry_avg_pct = industry_avg * 100
            latest_value_pct = latest_value * 100
            if latest_value > industry_avg:
                comparison = (f"The {display_name} of {latest_value_pct:.1f}% is above the {self.industry.title()} "
                             f"industry average of {industry_avg_pct:.1f}%, indicating strong performance.")
            elif latest_value < industry_avg:
                comparison = (f"The {display_name} of {latest_value_pct:.1f}% is below the {self.industry.title()} "
                             f"industry average of {industry_avg_pct:.1f}%, suggesting potential challenges.")
            else:
                comparison = (f"The {display_name} of {latest_value_pct:.1f}% is in line with the {self.industry.title()} "
                             f"industry average of {industry_avg_pct:.1f}%.")
        else:
            if latest_value > industry_avg:
                comparison = (f"The {display_name} of {latest_value:.2f} is above the {self.industry.title()} "
                             f"industry average of {industry_avg:.2f}, indicating strong performance.")
            elif latest_value < industry_avg:
                comparison = (f"The {display_name} of {latest_value:.2f} is below the {self.industry.title()} "
                             f"industry average of {industry_avg:.2f}, suggesting potential challenges.")
            else:
                comparison = (f"The {display_name} of {latest_value:.2f} is in line with the {self.industry.title()} "
                             f"industry average of {industry_avg:.2f}.")
        
        return comparison

    def save_custom_ratio(self, name: str, formula: str, description: str, category: str):
        """Save custom ratio to JSON file with category"""
        # First ensure the category exists
        if category not in self.custom_ratios:
            self.custom_ratios[category] = []
        
        # Check if ratio already exists
        for ratio in self.custom_ratios[category]:
            if ratio['name'] == name:
                ratio['formula'] = formula
                ratio['description'] = description
                break
        else:
            self.custom_ratios[category].append({
                "name": name,
                "formula": formula,
                "description": description
            })
        
        # Also add to essential ratios if it's a standard category
        if category != 'custom' and category in self.essential_ratios:
            # Check if already exists
            existing = next((r for r in self.essential_ratios[category] if r['name'] == name), None)
            if existing:
                existing['formula'] = formula
                existing['description'] = description
            else:
                self.essential_ratios[category].append({
                    "name": name,
                    "formula": formula,
                    "description": description
                })
            
            # Save essential ratios
            with open('essential_ratios.json', 'w') as f:
                json.dump(self.essential_ratios, f, indent=4)
        
        # Save custom ratios
        with open('custom_ratios.json', 'w') as f:
            json.dump(self.custom_ratios, f, indent=4)
        
        logger.info(f"Saved custom ratio: {name} in category {category}")


    def display_summary_table(self, ratio_results: Dict):
        print("\n" + "=" * 80)
        print("FINANCIAL RATIO DASHBOARD".center(80))
        print("=" * 80)
        
        # Collect all available years
        all_years = set()
        for ratio_data in ratio_results.values():
            all_years.update(ratio_data["values"].keys())
        years = sorted(all_years)
        
        # Create a list of all ratios that have data
        valid_ratios = []
        for ratio_name, ratio_data in ratio_results.items():
            if ratio_data["values"]:
                valid_ratios.append((ratio_name, ratio_data))
        
        # If no ratios, show error
        if not valid_ratios:
            print("\n⚠️ No ratios calculated! Possible reasons:")
            print("- Missing financial data in input statements")
            print("- Unmapped financial terms in ratio formulas")
            print("- Missing market prices for valuation ratios")
            print("Please check your input data and try again.\n")
            print("=" * 80)
            return
        
        # Create summary table
        table_data = []
        
        # Add header row
        headers = ["Ratio", f"Values ({' | '.join(years)})", "Description", "Trend", "Analysis Code"]
        table_data.append(headers)
        
        # Add divider row
        table_data.append(["-"*20, "-"*30, "-"*40, "-"*15, "-"*15])
        
        # Add all valid ratios to the table
        for ratio_name, ratio_data in valid_ratios:
            display_name = self.get_ratio_display_name(ratio_name)
            values = []
            for year in years:
                if year in ratio_data["values"]:
                    values.append(self.format_ratio_value(ratio_name, ratio_data["values"][year]))
                else:
                    values.append("-")
            
            # Format values string
            values_str = " | ".join([f"{year}: {val}" for year, val in zip(years, values)])
            
            # Add trend indicator
            trend_symbol = ""
            trend_desc = ""
            if "trend" in ratio_data and ratio_data["trend"].get("direction", "insufficient_data") != "insufficient_data":
                if ratio_data["trend"]["direction"] == "increasing":
                    trend_symbol = "↑"
                    trend_desc = "Improving"
                elif ratio_data["trend"]["direction"] == "decreasing":
                    trend_symbol = "↓"
                    trend_desc = "Declining"
                elif ratio_data["trend"]["direction"] == "fluctuating":
                    trend_symbol = "↕"
                    trend_desc = "Fluctuating"
            
            table_data.append([
                f"{display_name} {trend_symbol}",
                values_str,
                ratio_data["description"],
                trend_desc,
                f"[{ratio_name}]"
            ])
        
        # Print table
        if table_data:
            print(tabulate(
                table_data, 
                headers=["Ratio", f"Values ({' | '.join(years)})", "Description", "Trend", "Analysis Code"], 
                tablefmt="grid",
                maxcolwidths=[25, 30, 40, 15, 15]
            ))
        print("=" * 80)
        print("To view detailed analysis, enter the ratio's Analysis Code (e.g., current_ratio)")
        print("=" * 80)

    def interactive_analytics(self):
        self.start_process('system_init', 'Starting financial analytics')
        ratio_results = self.calculate_all_ratios()
        self.current_ratio_results = ratio_results
        
        # Generate overall analysis
        self.generate_overall_analysis(ratio_results)
        print("\nBefore we begin, please specify your industry for more relevant analysis.")
        print("Available industries: " + ", ".join([i.title() for i in self.industry_averages.keys()]))
        
        industry_set = False
        while not industry_set:
            industry = input("\nEnter your industry (or press Enter to skip): ").strip()
            if not industry:
                print("Continuing without industry context")
                break
            
            industry_set = self.set_industry(industry)
        self.complete_process('system_init')
        while True:
            self.display_summary_table(ratio_results)
            
            print("\nOptions:")
            print("1. View detailed ratio analysis")
            print("2. Calculate custom ratio")
            print("3. Ask about a specific ratio")
            print("4. Generate comprehensive financial health report")
            print("5. Delete a custom ratio")
            print("6. Reset ratios")
            print("7. Exit")
            
            choice = input("\nSelect an option: ").strip()
            
            if choice == '1':
                ratio_name = input("Enter ratio Analysis Code (e.g., current_ratio): ").strip().lower()
                ratio_name = ratio_name.replace(' ', '_')  # Normalize input
                
                if ratio_name in ratio_results:
                    self.display_ratio(ratio_name, ratio_results[ratio_name])
                    self.interactive_qa(ratio_name, ratio_results[ratio_name])
                else:
                    print(f"\n⚠️ Ratio '{ratio_name}' not found. Available ratios:")
                    available = [r for r in ratio_results.keys()]
                    print(", ".join(available))
            
            elif choice == '2':
                prompt = input("\nEnter your ratio request (e.g., 'quick ratio', 'EBITDA'): ").strip()
                if prompt:
                    self.handle_custom_ratio_request(prompt)
            
            elif choice == '3':
                ratio_name = input("\nEnter ratio Analysis Code: ").strip().lower()
                ratio_name = ratio_name.replace(' ', '_')
                
                if ratio_name in ratio_results:
                    question = input("Enter your question (e.g., 'Why did it increase?'): ").strip()
                    if question:
                        answer = self.answer_ratio_question(ratio_name, ratio_results[ratio_name], question)
                        print(f"\n{'ANALYST INSIGHT':-^60}")
                        print(f"{answer}\n{'-'*60}")
                else:
                    print(f"Ratio '{ratio_name}' not found")
            
            elif choice == '4':
                self.generate_comprehensive_report(ratio_results)
            
            elif choice == '5':
                if self.delete_ratio():
                    # Refresh ratio results after deletion
                    ratio_results = self.calculate_all_ratios()
                    self.current_ratio_results = ratio_results
            
            elif choice == '6':
                print("\nReset Options:")
                print("1. Reset all ratios to default")
                print("2. Reset a single ratio to default")
                print("3. Cancel")
                
                reset_choice = input("\nSelect reset option: ").strip()
                
                if reset_choice == '1':
                    if self.reset_ratios():
                        # Refresh ratio results after resetting
                        ratio_results = self.calculate_all_ratios()
                        self.current_ratio_results = ratio_results
                elif reset_choice == '2':
                    if self.reset_single_ratio():
                        # Refresh ratio results after resetting
                        ratio_results = self.calculate_all_ratios()
                        self.current_ratio_results = ratio_results
                elif reset_choice == '3':
                    print("Reset cancelled")
                else:
                    print("Invalid option")
            
            elif choice == '7':
                print("\nExiting financial analytics system")
                break
            
            else:
                print("Invalid option. Please try again.")
    def generate_overall_analysis(self, ratio_results: Dict):
        """Generate brief overall financial health analysis"""
        print("\n" + "=" * 80)
        print("INITIAL FINANCIAL HEALTH ASSESSMENT".center(80))
        print("=" * 80)

        if not any(len(ratio_data["values"]) > 0 for ratio_data in ratio_results.values()):
            print("\n⚠️ No financial ratios calculated!")
            print("Valuation ratios require market prices. Please provide market prices")
            print("for valuation ratios or skip them to focus on operational ratios.\n")
            print("=" * 80)
            return
            
        # Collect key insights
        insights = []
        years = sorted(self.get_available_years())
        latest_year = years[-1] if years else ""
        
        # Profitability analysis
        profitability = []
        for ratio in ['gross_margin', 'net_margin', 'return_on_equity']:
            if ratio in ratio_results and ratio_results[ratio]['values']:
                latest_value = ratio_results[ratio]['values'].get(latest_year, 0)
                trend = ratio_results[ratio].get('trend', {})
                profitability.append({
                    'name': ratio,
                    'value': latest_value,
                    'trend': trend.get('direction', 'stable')
                })
        
        # Liquidity analysis
        liquidity = []
        for ratio in ['current_ratio', 'quick_ratio']:
            if ratio in ratio_results and ratio_results[ratio]['values']:
                latest_value = ratio_results[ratio]['values'].get(latest_year, 0)
                trend = ratio_results[ratio].get('trend', {})
                liquidity.append({
                    'name': ratio,
                    'value': latest_value,
                    'trend': trend.get('direction', 'stable')
                })
        
        # Generate summary insights
        if profitability:
            improving = sum(1 for p in profitability if p['trend'] == 'increasing')
            insights.append(f"📈 Profitability: {improving}/{len(profitability)} key ratios improving")
        
        if liquidity:
            strong_liquidity = sum(1 for l in liquidity if 
                                  (l['name'] == 'current_ratio' and l['value'] > 1.5) or
                                  (l['name'] == 'quick_ratio' and l['value'] > 1.0))
            insights.append(f"💧 Liquidity: {'Strong' if strong_liquidity == len(liquidity) else 'Moderate'} position")
        
        # Valuation check - only if we have P/E ratio data
        if 'pe_ratio' in ratio_results and ratio_results['pe_ratio']['values']:
            pe = ratio_results['pe_ratio']['values'].get(latest_year)
            if pe is not None and pe > 0:  # Only show if valid positive value
                valuation = "Undervalued" if pe < 15 else "Fairly valued" if pe < 25 else "Overvalued"
                insights.append(f"💰 Valuation: {valuation} (P/E: {pe:.2f})")
        
        # Debt check
        if 'debt_to_equity' in ratio_results and ratio_results['debt_to_equity']['values'] and latest_year in ratio_results['debt_to_equity']['values']:
            de = ratio_results['debt_to_equity']['values'].get(latest_year, 0)
            debt_level = "Low" if de < 0.5 else "Moderate" if de < 1.0 else "High"
            insights.append(f"🏦 Leverage: {debt_level} debt (D/E: {de:.2f})")
        
        # Print insights
        if insights:
            print("\nKey Observations:")
            for insight in insights:
                print(f" - {insight}")
        else:
            print("⚠️ Insufficient data for initial assessment")
        if 'pe_ratio' in ratio_results and not ratio_results['pe_ratio']['values']:
            print("\n⚠️ P/E ratio missing - possible reasons:")
            print("- Missing market prices for some years")
            print("- Missing shares outstanding data")
            print("- Negative earnings or zero EPS")
        print("\n" + "=" * 80)
        print("For detailed analysis of individual ratios, select option 1")
        print("For a comprehensive financial health report, select option 4")
        print("=" * 80 + "\n")



    def generate_comprehensive_report(self, ratio_results: Dict):
        """Generate comprehensive financial health report using LLM"""
        if not self.azure_client:
            print("LLM service unavailable for comprehensive reporting")
            return
        
        try:
            # Prepare data for LLM
            report_data = []
            years = sorted(self.get_available_years())
            
            for ratio_name, data in ratio_results.items():
                if not data['values']:
                    continue
                
                values = []
                for year in years:
                    if year in data['values']:
                        values.append(f"{year}: {self.format_ratio_value(ratio_name, data['values'][year])}")
                
                trend = data.get('trend', {})
                trend_desc = ""
                if trend.get('direction') != "insufficient_data":
                    trend_desc = (f"Trend: {trend['direction']} from "
                                f"{self.format_ratio_value(ratio_name, trend['start_value'])} in {trend['start_year']} to "
                                f"{self.format_ratio_value(ratio_name, trend['end_value'])} in {trend['end_year']} "
                                f"({trend['change_pct']:+.1f}%)")
                
                report_data.append({
                    "ratio": self.get_ratio_display_name(ratio_name),
                    "values": ", ".join(values),
                    "trend": trend_desc,
                    "description": data['description']
                })
            
            # Prepare prompt
            prompt = (
                "You are a senior financial analyst. Generate a comprehensive financial health report "
                "based on the following ratios. Include:\n"
                "1. Executive summary of overall financial health\n"
                "2. Analysis by category (profitability, liquidity, etc.)\n"
                "3. Key strengths and weaknesses\n"
                "4. Recommendations for improvement\n"
                "5. Future outlook\n\n"
                "Ratio Data:\n"
            )
            
            for item in report_data:
                prompt += (
                    f"- {item['ratio']}: {item['values']}\n"
                    f"  Description: {item['description']}\n"
                    f"  {item['trend']}\n\n"
                )
            
            # Add financial statement context
            prompt += (
                "\nFinancial Statement Context:\n"
                f"Income Statement: {list(self.income.keys())[:3]}...\n"
                f"Balance Sheet: {list(self.balance.keys())[:3]}...\n"
                f"Cash Flow: {list(self.cashflow.keys())[:3]}...\n"
            )
            
            # Generate report
            print("\nGenerating comprehensive financial health report...")
            response = self.azure_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a financial analyst with 20 years of experience."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            report = response.choices[0].message.content
            
            # Display report
            print("\n" + "=" * 80)
            print("COMPREHENSIVE FINANCIAL HEALTH REPORT".center(80))
            print("=" * 80)
            print(report)
            print("=" * 80)
            
            # Save report to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"financial_report_{timestamp}.txt"
            with open(filename, 'w') as f:
                f.write(report)
            print(f"\nReport saved to {filename}")
            
        except Exception as e:
            print(f"Failed to generate report: {e}")