# this is just a basic version of SELF Learning parser (its only matches the mapping terms with financial statement no memory of it to learn and improve)
import re
import os
import pandas as pd
import torch
import numpy as np
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer, util
from openai import AzureOpenAI
from typing import Dict, List, Tuple, Optional, Union

# My model configuration 
MATCHING_CONFIG = {
    'regex_threshold': 1,
    'fuzzy_threshold': 85,
    'bert_threshold': 0.72,
    'bert_model': 'sentence-transformers/all-mpnet-base-v2',
    'enable_llm': True,
    'llm_model': 'gpt-4o-mini',
    'ratio_essential_fields': {
        'income': ['net_income', 'total_revenue', 'operating_income', 'ebitda'],
        'balance': ['total_assets', 'total_liabilities', 'shareholders_equity'],
        'cashflow': ['net_cash_operating', 'capital_expenditure']
    }
}

#  Matching patterns
MAPPING_PATTERNS: Dict[str, Dict[str, List[Dict]]] = {
    'income': {
            'net_sales': [
        {'pattern': r'^net(sales)$', 'priority': 1},
        {'pattern': r'^sales(net)?$', 'priority': 2}
    ],
    'total_revenue': [
        {'pattern': r'totalrevenues?', 'priority': 1},
        {'pattern': r'revenues?(net)?', 'priority': 2}
    ],
        'cost_of_sales': [
            {'pattern': r'costofsales', 'priority': 1},
            {'pattern': r'cogs', 'priority': 2}
        ],
        'operating_expenses': [
            {'pattern': r'operatingsellinggeneralandadministrativeexpenses', 'priority': 1},
            {'pattern': r'opex', 'priority': 2},  # Add this line
            {'pattern': r'costsandexpenses', 'priority': 3}
        ],
        'operating_income': [
            {'pattern': r'operatingincome', 'priority': 1},
            {'pattern': r'ebit', 'priority': 2}
        ],
        'interest_expense': [
    {'pattern': r'^interest(expense)?(net)?$', 'priority': 1},
    {'pattern': r'^interest(expense)?$', 'priority': 2},
    {'pattern': r'^interest(expense)?(and)?(finance)?(cost)?$', 'priority': 3},
    {'pattern': r'^interest(on)?(debt|lease)?$', 'priority': 4}
],
        'noncontrolling_interest': [
    {'pattern': r'noncontrolling', 'priority': 1}
],
        'interest_income': [
            {'pattern': r'interestincome', 'priority': 1}
        ],
        'other_gains_losses': [
            {'pattern': r'othergainsandlosses', 'priority': 1}
        ],
        'pre_tax_income': [
            {'pattern': r'incomebeforeincometaxes', 'priority': 1}
        ],
        'tax_expense': [
            {'pattern': r'provisionforincometaxes', 'priority': 1}
        ],
        'net_income': [
            {'pattern': r'^(?!basic|diluted)netincome', 'priority': 1},
            {'pattern': r'consolidatednetincomeattributableto.*', 'priority': 2},
            {'pattern': r'netearnings', 'priority': 3}
        ],
        'noncontrolling_interest': [
            {'pattern': r'netincomelossattributablenoncontrollinginterest', 'priority': 1}
        ],
        'eps_basic': [
            {'pattern': r'basicnetincomepercommonshare', 'priority': 1}
        ],
        'eps_diluted': [
            {'pattern': r'dilutednetincomepercommonshare', 'priority': 1}
        ],
        'shares_basic': [
            {'pattern': r'weightedaveragecommonsharesoutstandingbasic', 'priority': 1}
        ],
        'shares_diluted': [
            {'pattern': r'weightedaveragecommonsharesoutstandingdiluted', 'priority': 1}
        ],
        'dividends_per_share': [
            {'pattern': r'dividendsdeclaredpercommonshare', 'priority': 1}
        ],
        'gross_profit': [
        {'pattern': r'grossprofit', 'priority': 1}
    ],
    'preferred_dividends': [
        {'pattern': r'preferreddividends', 'priority': 1},
        {'pattern': r'dividendspreferred', 'priority': 2}
    ],
    'ebitda': [
        {'pattern': r'ebitda', 'priority': 1},
        {'pattern': r'earningsbeforeinteresttaxesdepreciationamortization', 'priority': 2}
    ]
    },
    'balance': {
       'cash_equivalents': [
        {'pattern': r'cash(and)?cashequivalents', 'priority': 1}
    ],
    'receivables': [
        {'pattern': r'receivablesnet', 'priority': 1},
        {'pattern': r'receivables', 'priority': 2}
    ],
    'inventory': [
        {'pattern': r'inventories', 'priority': 1}
    ],
    'prepaid_expenses': [
        {'pattern': r'prepaidexpenses(andother)?', 'priority': 1}
    ],
    'current_assets': [
        {'pattern': r'totalcurrentassets', 'priority': 1}
    ],
    'property_plant_equipment': [
        {'pattern': r'property(and)?equipment(net)?', 'priority': 1}
    ],
    'operating_lease_assets': [
        {'pattern': r'operatingleaserightofuseassets', 'priority': 1}
    ],
    'finance_lease_assets': [
        {'pattern': r'financeleaserightofuseassets', 'priority': 1}
    ],
    'goodwill': [
        {'pattern': r'goodwill', 'priority': 1}
    ],
    'other_long_term_assets': [
        {'pattern': r'otherlongtermassets', 'priority': 1}
    ],
    'total_assets': [
        {'pattern': r'totalassets', 'priority': 1}
    ],
    'short_term_borrowings': [
        {'pattern': r'shorttermborrowings', 'priority': 1}
    ],
    'accounts_payable': [
        {'pattern': r'accountspayable', 'priority': 1}
    ],
    'accrued_liabilities': [
        {'pattern': r'accruedliabilities', 'priority': 1}
    ],
    'accrued_income_taxes': [
        {'pattern': r'accruedincometaxes', 'priority': 1}
    ],
    'current_liabilities': [
        {'pattern': r'totalcurrentliabilities', 'priority': 1}
    ],
    'long_term_debt': [
        {'pattern': r'longterm(debt|debtdueafteroneyear)', 'priority': 1}
    ],
    'noncurrent_operating_lease_liabilities': [
        {'pattern': r'longtermoperatingleaseobligations', 'priority': 1}
    ],
    'noncurrent_finance_lease_liabilities': [
        {'pattern': r'longtermfinanceleaseobligations', 'priority': 1}
    ],
    'operating_lease_liabilities': [
        {'pattern': r'operatingleaseobligations(due)?withinoneyear', 'priority': 1}
    ],
    'finance_lease_liabilities': [
        {'pattern': r'financeleaseobligations(due)?withinoneyear', 'priority': 1}
    ],
    'total_liabilities': [
        {"pattern": r"^total liabilities$",  "priority": 1}
        
    ],
      'grand_total_liabilities_equity': [
    {
        "pattern": r"^totalliabilitiesredeemablenoncontrollinginterest(and)?(shareholders|stockholders)equity$", 
        "priority": 0
    },
    {
        "pattern": r"^total liabilities.*(equity|stockholders')$",
        "priority": 2
    }
],
        
    'deferred_taxes': [
        {'pattern': r'deferred(incometaxes)?(andother)?', 'priority': 1}
    ],
  'redeemable_noncontrolling_interest': [
    {"pattern": r"^redeemable noncontrolling interest$", "priority": 1},
    {"pattern": r"redeemablenoncontrollinginterest", "priority": 2}
],

    'nonredeemable_noncontrolling_interest': [
        {"pattern": r"^nonredeemable noncontrolling interest$", "priority": 1},
    {"pattern": r"nonredeemablenoncontrollinginterest", "priority": 2}
    ],
    'common_stock': [
        {'pattern': r'commonstock', 'priority': 1}
    ],
    'capital_in_excess_of_par_value': [
        {'pattern': r'capitalinexcessofparvalue', 'priority': 1}
    ],
    'retained_earnings': [
        {'pattern': r'retainedearnings', 'priority': 1}
    ],
    'accumulated_other_comprehensive_loss': [
        {'pattern': r'accumulatedothercomprehensiveloss', 'priority': 1}
    ],
    'shareholders_equity': [
        {"pattern": r"^total (stockholders'|shareholders') equity$", "priority": 1},
    {"pattern": r"^stockholders'? equity$", "priority": 2},
    {"pattern": r"^shareholders'? equity$", "priority": 3}
    ],
    'total_shareholders_equity': [
       {
        "pattern": r"(total )?.*?(stockholders'|shareholders') equity", "priority": 1
    }
    ],
    'preferred_stock': [
        {'pattern': r'preferredstock', 'priority': 1}
    ],
    'temporary_equity': [
        {'pattern': r'temporaryequity', 'priority': 1},
        {'pattern': r'redeemable.*equity', 'priority': 2}
    ],
    'treasury_stock': [
        {'pattern': r'treasurystock', 'priority': 1},
        {'pattern': r'commonsharesrepurchased', 'priority': 2}
    ]
},
    'cashflow': {
        'net_cash_operating': [
        {'pattern': r'netcashprovidedbyoperatingactivities', 'priority': 1}
    ],
    'depreciation_amortization': [
        {'pattern': r'depreciationandamortization', 'priority': 1}
    ],
    'investing_gains_losses': [
        {'pattern': r'investmentgainsandlossesnet', 'priority': 1}
    ],
    'deferred_taxes': [
        {'pattern': r'deferredincometaxes', 'priority': 1}
    ],
    'other_operating_activities': [
        {'pattern': r'otheroperatingactivities', 'priority': 1}
    ],
    'changes_in_receivables': [
        {'pattern': r'receivablesnet', 'priority': 1}
    ],
    'changes_in_inventory': [
        {'pattern': r'inventories', 'priority': 1}
    ],
    'changes_in_payables': [
        {'pattern': r'accountspayable', 'priority': 1}
    ],
    'changes_in_accrued_liabilities': [
        {'pattern': r'accruedliabilities', 'priority': 1}
    ],
    'changes_in_income_taxes': [
        {'pattern': r'accruedincometaxes', 'priority': 1}
    ],
    'income_taxes_paid': [
        {'pattern': r'incometaxespaid', 'priority': 1}
    ],
    'income_taxes_refunded': [
        {'pattern': r'incometaxesrefunded', 'priority': 1}
    ],
    'interest_paid': [
        {'pattern': r'interestpaid', 'priority': 1}
    ],
    'interest_received': [
        {'pattern': r'interestreceived', 'priority': 1}
    ],
    'net_cash_investing': [
        {'pattern': r'netcashusedininvestingactivities', 'priority': 1}
    ],
    'capital_expenditure': [
        {'pattern': r'paymentsforpropertyandequipment', 'priority': 1}
    ],
    'acquisition_of_businesses': [
        {'pattern': r'purchaseofbusiness', 'priority': 1},
        {'pattern': r'acquisitionofsubsidiaries', 'priority': 2}
    ],
    'proceeds_from_disposal': [
        {'pattern': r'proceedsfromsaleofpropertyandequipment', 'priority': 1},
        {'pattern': r'proceedsfromsaleofassets', 'priority': 2},
        {'pattern': r'proceedsfrom.*', 'priority': 3}
    ],
    'proceeds_from_maturities': [
        {'pattern': r'proceedsfrommaturitiesofinvestments', 'priority': 1}
    ],
    'net_cash_financing': [
        {'pattern': r'netcashusedinfinancingactivities', 'priority': 1}
    ],
    'debt_issuance': [
        {'pattern': r'proceedsfromissuanceoflongtermdebt', 'priority': 1}
    ],
    'debt_repayments': [
        {'pattern': r'repaymentsoflongtermdebt', 'priority': 1}
    ],
    'short_term_borrowings': [
        {'pattern': r'proceedsfromshorttermborrowings', 'priority': 1}
    ],
    'repayment_of_short_term_borrowings': [
        {'pattern': r'repaymentsofshorttermborrowings', 'priority': 1}
    ],
    'dividends_paid_to_common': [
        {'pattern': r'dividendspaidtocommon', 'priority': 1},
        {'pattern': r'dividendspaidtocommonstockholders', 'priority': 2},
        {'pattern': r'cashdividendspaidcommon', 'priority': 3}
    ],
    
    'dividends_paid_to_minority': [
        {'pattern': r'dividendspaidtominority', 'priority': 1},
        {'pattern': r'dividendspaidtominorityinterests', 'priority': 2},
        {'pattern': r'cashdividendspaidminority', 'priority': 3}
    ],
        
    'dividends_paid_to_nci': [
        {'pattern': r'dividendspaidtononcontrollinginterest', 'priority': 1}
    ],
    'purchase_of_stock': [
        {'pattern': r'purchaseofcompanystock', 'priority': 1},
        {'pattern': r'sharebuyback', 'priority': 2},
        {'pattern': r'sharerepurchase', 'priority': 2}
    ],
    'stock_issuance': [
        {'pattern': r'proceedsfromissuanceofcommonstock', 'priority': 1}
    ],
    'stock_repurchase': [
        {'pattern': r'purchasesofcommonstock', 'priority': 1},
        {'pattern': r'treasurystockrepurchased', 'priority': 2}
    ],
    'net_increase_cash': [
        {'pattern': r'netincrease(?:\(decrease\))?incashequivalents', 'priority': 1}
    ],
  
    'free_cash_flow': [
        {'pattern': r'freecashflow', 'priority': 1},
        {'pattern': r'fcf', 'priority': 2}
    ],
        'stock_based_compensation': [
    {'pattern': r'stockbasedcompensationexpense', 'priority': 1},
    {'pattern': r'sharebasedcompensation', 'priority': 2},
    {'pattern': r'stockoptionexpense', 'priority': 3}
],

'lease_liabilities_paid': [
    {'pattern': r'paymentsofprincipalleaseobligations', 'priority': 1},
    {'pattern': r'leasepayments', 'priority': 2}
],

'proceeds_from_preferred_stock': [
    {'pattern': r'proceedsfromissuanceofpreferredstock', 'priority': 1}
],
'net_income': [
    {'pattern': r'consolidatednetincome', 'priority': 1},
    {'pattern': r'netincome', 'priority': 2}
],

'business_acquisitions': [
    {'pattern': r'paymentsforbusinessacquisitions', 'priority': 1},
    {'pattern': r'acquisitionsofsubsidiaries', 'priority': 2}
],

'other_investing_activities': [
    {'pattern': r'otherinvestingactivities', 'priority': 1}
],

'purchase_noncontrolling_interest': [
    {'pattern': r'purchaseofnoncontrollinginterest', 'priority': 1}
],

'proceeds_from_subsidiary_stock_sale': [
    {'pattern': r'saleofsubsidiarystock', 'priority': 1},
    {'pattern': r'issuanceofsubsidiarystock', 'priority': 2}
],

'other_financing_activities': [
    {'pattern': r'otherfinancingactivities', 'priority': 1}
],

'fx_effect_on_cash': [
    {'pattern': r'effectofexchangeratesoncash', 'priority': 1},
    {'pattern': r'exchangerateimpact', 'priority': 2}
],
    'non_cash_investing': [
        {'pattern': r'noncashinvestingactivities', 'priority': 1},
        {'pattern': r'noncashinvesting', 'priority': 2},
        {'pattern': r'capitalizedleases', 'priority': 3}
    ],

    'non_cash_financing': [
        {'pattern': r'noncashfinancingactivities', 'priority': 1},
        {'pattern': r'conversionofdebttoequity', 'priority': 2},
        {'pattern': r'conversionofdebttoequityandother', 'priority': 3}
    ],

        
'cash_beginning': [
    {'pattern': r'cashequivalentsatbeginning', 'priority': 1},
    {'pattern': r'cashatbeginningofyear', 'priority': 2}
],

'cash_end': [
    {'pattern': r'cashequivalentsatend', 'priority': 1},
    {'pattern': r'cashatendofyear', 'priority': 2}
]

}
}


# using bert model cuda applicable
bert_model = SentenceTransformer(MATCHING_CONFIG['bert_model'])
if torch.cuda.is_available():
    bert_model = bert_model.to('cuda')

STD_LABEL_EMBEDDINGS: Dict[str, Dict[str, torch.Tensor]] = {}
for section, patterns in MAPPING_PATTERNS.items():
    labels = list(patterns.keys())
    embs = bert_model.encode(labels, convert_to_tensor=True)
    STD_LABEL_EMBEDDINGS[section] = {lbl: emb for lbl, emb in zip(labels, embs)}
'''
# My OpenAI configuration here(privacy issues put your key and endpoint to use LLM)
client = AzureOpenAI(
    api_key="apikeys",
    api_version="version",
    azure_endpoint="https://azure_endpoints"
)

MODEL_DEPLOYMENT_NAME = "put gpt model like 4o,4omini etc"


#fallback option to LLm at last if all models get failed


def llm_fallback(label: str, section: str) -> Optional[str]:
    prompt = (
        f"you're an expert in financial analysis"
        f"Map this financial line to a standardized key for the {section} statement:\n"
        f"Label: '{label}'\n"
        f"Possible keys: {list(MAPPING_PATTERNS.get(section, {}).keys())}\n"
        f"Answer with the exact key or 'None'."
    )

    response = client.chat.completions.create(
        model=MODEL_DEPLOYMENT_NAME,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0
    )

    ans = response.choices[0].message.content.strip()
    return ans if ans in MAPPING_PATTERNS.get(section, {}) else None
'''
# making data clear and normalizing the value for easy understandibility

def enhanced_normalization(label: str) -> str:
    label = re.sub(r'\([^)]*\)', '', label)
    label = re.sub(r'[^a-zA-Z0-9]+', ' ', label).lower()
    label = re.sub(r'\b(million|thousand|usd|%|percent)\b', '', label)
    return re.sub(r'\s+', '', label)

# methods for matching the financial term of sheet with described above mapping patterns

def final_match(label: str, section: str) -> Optional[str]:
    norm = enhanced_normalization(label)
    print(f"\n→ Matching: '{label}' | normalized: '{norm}' | section: '{section}'")
    
    # 1. Regex 
    best = None
    for std, cfgs in MAPPING_PATTERNS.get(section, {}).items():
        for cfg in cfgs:
            if re.search(cfg['pattern'], norm):
                if best is None or cfg['priority'] < best[1]:
                    best = (std, cfg['priority'])
    if best and best[1] <= MATCHING_CONFIG['regex_threshold']:
        print(f"✔ Regex match: {best[0]}")
        return best[0]

    # 2. Fuzzy 
    choices = list(MAPPING_PATTERNS.get(section, {}).keys())
    if choices:
        match, score, _ = process.extractOne(norm, choices, scorer=fuzz.token_set_ratio)
        if score >= MATCHING_CONFIG['fuzzy_threshold']:
            print(f"✔ Fuzzy match: {match} (score: {score})")
            return match

    # 3. BERT 
    embs = STD_LABEL_EMBEDDINGS.get(section, {})
    if embs:
        lab_emb = bert_model.encode([norm], convert_to_tensor=True)
        sims = util.cos_sim(lab_emb, torch.stack(list(embs.values())))
        i = int(torch.argmax(sims))
        top_score = float(sims[0][i])
        if top_score >= MATCHING_CONFIG['bert_threshold']:
            bert_key = list(embs.keys())[i]
            print(f"✔ BERT match: {bert_key} (score: {top_score:.2f})")
            return bert_key


    # 4. LLM (uncomment only if you have put your  key and endpoint in above AI configuration)
    '''
    if MATCHING_CONFIG.get('enable_llm', False):
        llm_key = llm_fallback(label, section)
        if llm_key:
            print(f"✔ LLM match: {llm_key}")
            return llm_key
    '''

    print("✘ No match")
    return None



 
def multi_strategy_matcher(label: str, section: str) -> Optional[str]:
    return final_match(label, section)

# cleaning the value stored in series
def smart_value_conversion(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    s = s.str.replace('[\$,]', '', regex=True)
    s = s.str.replace(r'^\((.+)\)$', r'-\1', regex=True)
    return pd.to_numeric(s, errors='coerce')

# 
def validate_essential_fields(mappings: Dict[str, pd.Series], section: str) -> bool:
    essentials = MATCHING_CONFIG['ratio_essential_fields'].get(section, [])
    missing = [f for f in essentials if f not in mappings]
    if missing:
        print(f"\u26a0\ufe0f Missing essential fields for {section}: {missing}") # with unicode for warning sign
    return not missing

def promote_date_row(df: pd.DataFrame) -> pd.DataFrame:
    """
    Search the first 4 rows for date-like patterns and promote header row
    """
    date_patterns = [
        re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},\s+\d{4}', re.IGNORECASE),
        re.compile(r'\d{4}-\d{2}-\d{2}'),
        re.compile(r'\d{2}/\d{2}/\d{4}'),
        re.compile(r'\d{4}_\d{2}_\d{2}')
    ]

    for i in range(min(4, len(df))):
        # Check if any cell matches date patterns
        match_count = df.iloc[i].astype(str).apply(
            lambda x: any(p.search(x) for p in date_patterns)
        ).sum()
        
        if match_count >= 1:  # Require at least 2 date-like columns
            # Promote header row
            df.columns = df.iloc[i].values
            return df.iloc[i+1:].reset_index(drop=True)
    
    return df

def extract_financial_data(df: pd.DataFrame, section: str) -> Dict[str, Union[pd.Series, dict]]:
    df = promote_date_row(df)

    # makes first column index
     
    if df.iloc[:, 0].dtype == object and not df.index.dtype == object:
        df = df.set_index(df.columns[0])

    mappings = {}

    for row_label in df.index:
        clean_label = str(row_label).strip()
        row_values = df.loc[row_label]

        # skip blank or NAN
        if row_values.isna().all() or row_values.astype(str).str.strip().eq('').all():
            continue

        key = multi_strategy_matcher(clean_label, section)
        if key:
            val = smart_value_conversion(row_values)
            if not val.isna().all():
                mappings[key] = val

    valid = validate_essential_fields(mappings, section)
    return {
        'data': mappings,
        'validation': {'section': section, 'valid': valid}
    }

# Example usage
# income = extract_financial_data(income_df, 'income') same for balance and CashFlow.

