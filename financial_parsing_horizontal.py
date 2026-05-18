#without def hierarchical
import re
import os
import json
import time
import logging
import threading
import pandas as pd
import numpy as np
import torch
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer, util
from openai import AzureOpenAI, APIError, APITimeoutError
from typing import Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler
from config import api_key ,endpoint

print("\n" + "="*60)
print("FINANCIAL DATA PARSER CONFIGURATION".center(60))
print("="*60)
user_expertise = input("Are you experienced in financial domain analysis? (yes/no): ").strip().lower()
enable_hitl = user_expertise in ['y', 'yes']
print(f"\nHITL REVIEW MODE: {'ENABLED' if enable_hitl else 'DISABLED'}")
if enable_hitl:
    print("You will review critical mapping decisions")
else:
    print("System will operate in fully automatic mode")
print("="*60 + "\n")

# ------------------ Logging Configuration ------------------
logger = logging.getLogger("FinancialParser")
logger.setLevel(logging.INFO)

log_formatter = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | [%(threadName)s] %(message)s'
)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# File handler with daily rotation
file_handler = TimedRotatingFileHandler(
    'financial_parser.log', when='midnight', backupCount=7
)
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# ------------------ Configuration ------------------
class Config:
    def __init__(self):
        self.regex_threshold = 3
        self.fuzzy_threshold = 80
        self.bert_threshold = 0.68
        self.bert_model = 'sentence-transformers/all-mpnet-base-v2'
        self.enable_llm = True
        self.llm_model = 'gpt-4o-mini'
        self.llm_timeout = 15
        self.essential_fields = {
            'income': ['net_income', 'total_revenue', 'operating_income'],
            'balance': ['total_assets', 'total_liabilities', 'shareholders_equity'],
            'cashflow': ['net_cash_operating', 'capital_expenditure']
        }
        self.llm_retry_attempts = 3
        self.llm_retry_min_wait = 2
        self.llm_retry_max_wait = 30
        self.max_llm_history = 1000
        self.retrain_interval_hours = 24
        self.dynamic_equivalents = {
            'net_cash_operating': ['net_cash_operating', 'cash_from_operations'],
            'capital_expenditure': ['capital_expenditure', 'capex']
        }
        self.enable_hitl = enable_hitl
    # Auto-approve high confidence matches
        self.hitl_auto_approve_threshold = 0.95  
        self.hitl_history_file = 'hitl_history.json'

config = Config()

# ------------------ Initialize Models ------------------
logger.info("Initializing BERT model...")
bert_model = SentenceTransformer(config.bert_model)
if torch.cuda.is_available():
    bert_model = bert_model.to('cuda')
    logger.info("Using CUDA acceleration")

# ------------------ Azure OpenAI Client ------------------"
class AzureClient:
    def __init__(self):
        self.client = AzureOpenAI(
    api_key=api_key,
    api_version="2023-05-15",
    azure_endpoint=endpoint
        )
        self.model_name = "gpt-4o-mini"
        
    @retry(
        stop=stop_after_attempt(config.llm_retry_attempts),
        wait=wait_exponential(
            multiplier=1, 
            min=config.llm_retry_min_wait, 
            max=config.llm_retry_max_wait
        ),
        retry=retry_if_exception_type((APITimeoutError, APIError)),
        before_sleep=lambda _: logger.warning("LLM call failed, retrying...")
    )
    def get_completion(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                timeout=config.llm_timeout
            )
            return response.choices[0].message.content
        except (APITimeoutError, APIError) as e:
            logger.error(f"LLM API error: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected LLM error: {e}")
            return ""

azure_client = AzureClient()

# ------------------ HITL Manager ------------------
class HITLManager:
    def __init__(self, pattern_manager):
        self.pattern_manager = pattern_manager
        self.review_queue = []
        self.history = self.load_history()
        
    def load_history(self) -> dict:
        try:
            if os.path.exists(config.hitl_history_file):
                with open(config.hitl_history_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading HITL history: {e}")
            return {}
            
    def save_history(self):
        try:
            with open(config.hitl_history_file, 'w') as f:
                json.dump(self.history, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving HITL history: {e}")
            
    def add_to_queue(self, item: dict):
        self.review_queue.append(item)
        logger.info(f"Added to HITL queue: {item['label']} -> {item.get('proposed_key', '?')}")
        
    def process_queue(self, df: pd.DataFrame = None):
        if not self.review_queue:
            return
            
        print("\n" + "="*60)
        print("HUMAN IN THE LOOP REQUIRED".center(60))
        print("="*60)
        
        for i, item in enumerate(self.review_queue[:]):  # Iterate over copy
            print(f"\nREVIEW ITEM {i+1}/{len(self.review_queue)}")
            print(f"Original Label: {item['label']}")
            print(f"Section: {item['section']}")
            print(f"LLM Suggestion: {item.get('proposed_key', 'New key creation')}")
            print(f"Match Type: {item.get('match_type', 'unknown')}")
            print(f"Confidence: {item.get('confidence', 'N/A')}%")
            
            if df is not None:
                print("\nData Preview:")
                print(df.head(3).to_string())
                
            print("\nOptions:")
            print("[A] Accept LLM suggestion")
            print("[E] Edit suggested key")
            print("[M] Map to different existing key")
            print("[S] Skip this item")
            print("[C] Create new custom key")
            
            action = input("\nChoose action: ").strip().lower()
            
            if action == 'a':  # Accept
                self.approve_mapping(item)
                self.review_queue.remove(item)
                
            elif action == 'e':  # Edit suggestion
                new_key = input(f"Edit suggested key [{item['proposed_key']}]: ").strip()
                if new_key:
                    item['proposed_key'] = new_key
                    self.approve_mapping(item)
                self.review_queue.remove(item)
                
            elif action == 'm':  # Map to existing key
                existing_keys = list(pattern_manager.base_patterns[item['section']].keys())
                print("\nExisting keys:")
                for idx, key in enumerate(existing_keys, 1):
                    print(f"{idx}. {key}")
                    
                choice = input("\nEnter number of key to map to (or 0 to cancel): ").strip()
                if choice.isdigit():
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(existing_keys):
                        item['proposed_key'] = existing_keys[choice_idx]
                        self.approve_mapping(item)
                self.review_queue.remove(item)
                
            elif action == 'c':  # Create custom key
                new_key = input("Enter new standardized key (snake_case): ").strip()
                if new_key:
                    # Create new key with the original label
                    pattern = PatternManager.generate_pattern(item['label'])
                    pattern_manager.base_patterns[item['section']][new_key] = [{
                        "pattern": pattern,
                        "priority": 1
                    }]
                    pattern_manager.save_patterns(
                        pattern_manager.base_patterns, 
                        'mapping_patterns.json'
                    )
                    embedding_manager.update_section(
                        item['section'], 
                        pattern_manager.base_patterns[item['section']]
                    )
                    pattern_manager.add_confirmed_mapping(item['label'], new_key, item['section'])
                    logger.info(f"Created new key: {new_key} for '{item['label']}'")
                self.review_queue.remove(item)
                
            elif action == 's':  # Skip
                print("Skipping item")
                self.review_queue.remove(item)
                continue
            
                
            # Add to history
            item['decision'] = action
            item['timestamp'] = datetime.now().isoformat()
            self.history[item['label']] = item
            self.save_history()
            
        print("\n" + "="*60)
        print("REVIEW COMPLETE".center(60))
        print("="*60)
        
    def approve_mapping(self, item: dict):
        """Add approved mapping to patterns"""
        label = item['label']
        section = item['section']
        key = item['proposed_key']
        
        # Add pattern with high priority
        pattern = PatternManager.generate_pattern(label)
        self.pattern_manager.base_patterns[section].setdefault(key, []).append({
            "pattern": pattern,
            "priority": 1
        })
        
        # Save patterns
        self.pattern_manager.save_patterns(self.pattern_manager.base_patterns, 'mapping_patterns.json')
        
        # Update embeddings
        embedding_manager.update_section(section, self.pattern_manager.base_patterns[section])
        
        # Add to confirmed mappings
        self.pattern_manager.add_confirmed_mapping(label, key, section)
        logger.info(f"Approved mapping: {label} -> {key}")
        
    def modify_mapping(self, item: dict):
        """User provides custom mapping"""
        label = item['label']
        section = item['section']
        key = input("Enter correct standardized key: ").strip()
        
        if not key:
            print("Invalid key, skipping")
            return
            
        # Generate pattern from label
        pattern = PatternManager.generate_pattern(label)
        
        # Add to base patterns
        self.pattern_manager.base_patterns[section].setdefault(key, []).append({
            "pattern": pattern,
            "priority": 1
        })
        
        # Save patterns
        self.pattern_manager.save_patterns(self.pattern_manager.base_patterns, 'mapping_patterns.json')
        
        # Update embeddings
        embedding_manager.update_section(section, self.pattern_manager.base_patterns[section])
        
        # Add to confirmed mappings
        self.pattern_manager.add_confirmed_mapping(label, key, section)
        logger.info(f"User modified mapping: {label} -> {key}")
        
    def handle_missing_fields(self, missing: list, section: str, df: pd.DataFrame):
        """Manual mapping for missing essential fields"""
        print("\n" + "="*60)
        print("MISSING ESSENTIAL FIELDS".center(60))
        print("="*60)
        print(f"Section: {section}")
        print(f"Missing: {', '.join(missing)}")
        #print("\nAvailable labels:")
        #print(df.index.tolist())
        
        mappings = {}
        available_labels = df.index.tolist()
        
        for field in missing:
            print(f"\nMapping for: {field}")
            
            # Get LLM suggestion for this field
            suggestion = self.get_llm_suggestion(field, section, available_labels)
            if suggestion:
                print(f"LLM Suggestion: {suggestion['label']} (confidence: {suggestion['confidence']}%)")
            
            print("\nAvailable labels:")
            for idx, label in enumerate(available_labels, 1):
                print(f"{idx}. {label}")
            
            print("\nOptions:")
            print("[number] Select label by number")
            if suggestion:
                print("[A] Accept LLM suggestion")
            print("[M] Manually enter label")
            print("[S] Skip this field")
            
            action = input("\nChoose action: ").strip().lower()
            
            if action == 'a' and suggestion:
                mappings[field] = suggestion['label']
                print(f"Accepted LLM suggestion: {suggestion['label']}")
            elif action.isdigit():
                choice_idx = int(action) - 1
                if 0 <= choice_idx < len(available_labels):
                    mappings[field] = available_labels[choice_idx]
                    print(f"Selected: {available_labels[choice_idx]}")
                else:
                    print("Invalid selection, skipping")
            elif action == 'm':
                while True:
                    label = input("Enter exact label: ").strip()
                    if label in available_labels:
                        mappings[field] = label
                        break
                    print("Label not found, try again")
            elif action == 's':
                print("Skipping field")
            else:
                print("Invalid option, skipping")
                
        return mappings
    
    def get_llm_suggestion(self, field: str, section: str, labels: list) -> dict:
        """Get LLM suggestion for the most suitable label"""
        prompt = f"""
        You are a financial data mapping assistant. For the standardized field:
        "{field}" in the {section} statement,
        which of these labels is the best match?
        
        Labels:
        {json.dumps(labels, indent=2)}
        
        Respond in JSON format:
        {{
            "label": "best_matching_label",
            "confidence": 0-100
        }}
        """
        
        try:
            response = azure_client.get_completion(prompt)
            if response:
                # Try to parse JSON response
                try:
                    result = json.loads(response)
                    if 'label' in result and result['label'] in labels:
                        return result
                except json.JSONDecodeError:
                    # Fallback: extract using regex
                    match = re.search(r'"label":\s*"([^"]+)"', response)
                    if match and match.group(1) in labels:
                        return {'label': match.group(1), 'confidence': 80}
        except Exception as e:
            logger.error(f"LLM suggestion failed: {e}")
        return None

# ------------------ Pattern Management ------------------
class PatternManager:
    def __init__(self):
        self.base_patterns = self.load_patterns('mapping_patterns.json')
        self.dynamic_patterns = self.load_patterns('dynamic_patterns.json')
        self.confirmed_mappings = self.load_patterns('confirmed_mappings.json')
        self.compiled_dynamic = {}
        
    def load_patterns(self, filename: str) -> Dict:
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    return json.load(f)
        # Return empty lists for confirmed mappings
            if filename == 'confirmed_mappings.json':
                return {'income': [], 'balance': [], 'cashflow': []}
            return {'income': {}, 'balance': {}, 'cashflow': {}}
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            if filename == 'confirmed_mappings.json':
                return {'income': [], 'balance': [], 'cashflow': []}
            return {'income': {}, 'balance': {}, 'cashflow': {}}
            
    def save_patterns(self, patterns: Dict, filename: str):
        try:
            with open(filename, 'w') as f:
                json.dump(patterns, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving {filename}: {e}")
            
    def get_compiled_dynamic(self, section: str) -> Dict:
        if section not in self.compiled_dynamic:
            self.compiled_dynamic[section] = {}
            for key, patterns in self.dynamic_patterns.get(section, {}).items():
                compiled = []
                for p in patterns:
                    try:
                        compiled.append(re.compile(p['pattern']))
                    except re.error:
                        logger.warning(f"Invalid regex pattern: {p['pattern']}")
                self.compiled_dynamic[section][key] = compiled
        return self.compiled_dynamic[section]
    
    def add_confirmed_mapping(self, label: str, key: str, section: str):
        """Add to confirmed mappings for retraining"""
        if section not in self.confirmed_mappings:
            self.confirmed_mappings[section] = []
        self.confirmed_mappings[section].append({
            'original_label': label,
            'key': key,
            'timestamp': datetime.now().isoformat()
    })
        self.save_patterns(self.confirmed_mappings, 'confirmed_mappings.json')
        
    @staticmethod
    def generate_pattern(label: str) -> str:
        """Create optimized regex pattern from label"""
        clean = re.sub(r'\b(and|or|the|of|in|for)\b', '', label.lower())
        clean = re.sub(r'[^a-zA-Z0-9]+', '.*', clean)
        return f"^{clean}.*"

# ------------------ Embedding Management ------------------
class EmbeddingManager:
    def __init__(self, bert_model):
        self.embeddings = {}
        self.lock = threading.Lock()
        self.bert_model = bert_model
        
    def update_section(self, section: str, patterns: Dict):
        with self.lock:
            labels = list(patterns.keys())
            if not labels:
                return
                
            logger.info(f"Updating embeddings for {section} section ({len(labels)} labels)")
            embs = self.bert_model.encode(labels, convert_to_tensor=True)
            if torch.cuda.is_available():
                embs = embs.to('cuda')
            self.embeddings[section] = {lbl: emb for lbl, emb in zip(labels, embs)}
            
    def get_embeddings(self, section: str) -> Dict:
        with self.lock:
            return self.embeddings.get(section, {})
            
    def initialize_all(self, patterns: Dict):
        for section in ['income', 'balance', 'cashflow']:
            self.update_section(section, patterns.get(section, {}))

pattern_manager = PatternManager()
embedding_manager = EmbeddingManager(bert_model)
embedding_manager.initialize_all(pattern_manager.base_patterns)
hitl_manager = HITLManager(pattern_manager)

# ------------------ Matching Strategies ------------------
class BaseMatcher:
    def __init__(self, config, pattern_manager, embedding_manager):
        self.config = config
        self.pattern_manager = pattern_manager
        self.embedding_manager = embedding_manager
        
    def match(self, norm_label: str, section: str) -> Optional[str]:
        if 'pershare' in norm_label:
            if 'basic' in norm_label:
                return 'basic_eps'
            elif 'diluted' in norm_label:
                return 'diluted_eps'
            elif 'dividend' in norm_label:
                return 'dividend_per_share'
        # 1. Regex Match
        best = None
        for std, cfgs in self.pattern_manager.base_patterns.get(section, {}).items():
            for cfg in cfgs:
                if re.search(cfg['pattern'], norm_label):
                    if best is None or cfg['priority'] < best[1]:
                        best = (std, cfg['priority'])
        if best and best[1] <= self.config.regex_threshold:
            logger.info(f" Base regex match: {best[0]}")
            return best[0]

        # 2. Fuzzy Match
        choices = list(self.pattern_manager.base_patterns.get(section, {}).keys())
        if choices:
            match, score, _ = process.extractOne(
                norm_label, choices, scorer=fuzz.token_set_ratio
            )
            if score >= self.config.fuzzy_threshold:
                logger.info(f"[ok] Base fuzzy match: {match} (score: {score})")
                return match

        # 3. BERT Match
        embs = self.embedding_manager.get_embeddings(section)
        if embs:
            lab_emb = bert_model.encode([norm_label], convert_to_tensor=True)
            sims = util.cos_sim(lab_emb, torch.stack(list(embs.values())))
            i = int(torch.argmax(sims))
            top_score = float(sims[0][i])
            if top_score >= self.config.bert_threshold:
                bert_key = list(embs.keys())[i]
                logger.info(f"[OK] Base BERT match: {bert_key} (score: {top_score:.2f})")
                return bert_key
                
        return None

class DynamicMatcher:
    def __init__(self, pattern_manager):
        self.pattern_manager = pattern_manager
        
    def match(self, norm_label: str, section: str) -> Optional[str]:
        compiled = self.pattern_manager.get_compiled_dynamic(section)
        for key, patterns in compiled.items():
            for pattern in patterns:
                if pattern.search(norm_label):
                    logger.info(f"✔ Dynamic match: {key}")
                    return key
        return None

class LLMMatcher:
    def __init__(self, config, pattern_manager, azure_client, hitl_manager):
        self.config = config
        self.pattern_manager = pattern_manager
        self.azure_client = azure_client
        self.hitl_manager = hitl_manager
        self.llm_history = {}
        self.lock = threading.Lock()
        
    def parse_llm_response(self, response: str) -> dict:
        """Robust parsing of LLM response with multiple fallbacks"""
        # Attempt 1: Direct JSON parsing
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Attempt 2: Extract JSON from code block
        json_match = re.search(r'```json\n({.*?})\n```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Attempt 3: Extract key-value pairs
        result = {}
        key_matches = re.finditer(r'"(match_type|key|confidence)":\s*"([^"]+)"', response)
        for match in key_matches:
            result[match.group(1)] = match.group(2)
        
        if "key" in result:
            return result
            
        # Attempt 4: Last line as key
        lines = response.strip().split('\n')
        if lines:
            return {"match_type": "new", "key": lines[-1].strip()}
            
        # Final fallback
        return {"match_type": "new", "key": "unknown_item"}

    def match(self, label: str, section: str) -> Optional[str]:
        with self.lock:
            # Check cache
            if label in self.llm_history:
                return self.llm_history[label]
                
            # Clear cache if too large
            if len(self.llm_history) > self.config.max_llm_history:
                self.llm_history.clear()
        
        # Build prompt
        prompt = f"""
        ### Financial Statement Mapping Task
        **Label**: "{label}"
        **Section**: {section} statement
        
        **Available Standard Keys**:
        {json.dumps(list(pattern_manager.base_patterns[section].keys()), indent=2)}
        
        **Instructions**:
        1. If EXACT match exists, use "existing" with the exact key
        2. If SIMILAR concept exists, use "similar" with closest key
        3. If NO match, create new snake_case key using "new"
        
        **Output Format (JSON)**:
        {{
            "match_type": "existing|similar|new",
            "key": "standard_key",
            "confidence": 0-100
        }}
        """
        
        # Get LLM response
        response = self.azure_client.get_completion(prompt)
        if not response:
            return None
            
        # Parse response
        result = self.parse_llm_response(response)
        
        # Convert confidence to float if possible
        try:
            confidence = float(result.get('confidence', 0))
        except (ValueError, TypeError):
            confidence = 0
        
        # Handle based on match type
        match_type = result.get('match_type', 'new')
        proposed_key = result.get('key', 'unknown_item')
        
        # HITL Decision Point
        if config.enable_hitl and confidence < config.hitl_auto_approve_threshold:
            # Add to HITL review queue
            self.hitl_manager.add_to_queue({
                'label': label,
                'section': section,
                'match_type': match_type,
                'proposed_key': proposed_key,
                'confidence': confidence
            })
            return None  # Defer decision until review
            
        # Auto-approve for non-HITL or high confidence
        return self.apply_llm_result(label, section, result, match_type, proposed_key)
        
    def apply_llm_result(self, label, section, result, match_type, proposed_key):
        """Apply LLM result without HITL review"""
        with self.lock:
            if match_type == 'existing':
                self.llm_history[label] = proposed_key
                return proposed_key
                
            elif match_type == 'similar':
                # Add dynamic pattern
                new_pattern = {
                    "pattern": PatternManager.generate_pattern(label),
                    "priority": 3
                }
                
                # Add to dynamic patterns
                if section not in self.pattern_manager.dynamic_patterns:
                    self.pattern_manager.dynamic_patterns[section] = {}
                    
                if proposed_key not in self.pattern_manager.dynamic_patterns[section]:
                    self.pattern_manager.dynamic_patterns[section][proposed_key] = []
                    
                self.pattern_manager.dynamic_patterns[section][proposed_key].append(new_pattern)
                self.pattern_manager.save_patterns(
                    self.pattern_manager.dynamic_patterns, 
                    'dynamic_patterns.json'
                )
                
                self.llm_history[label] = proposed_key
                return proposed_key
                
            else:  # New key
                new_key = proposed_key
                # Add to base patterns
                if section not in self.pattern_manager.base_patterns:
                    self.pattern_manager.base_patterns[section] = {}
                    
                self.pattern_manager.base_patterns[section][new_key] = [{
                    "pattern": PatternManager.generate_pattern(label),
                    "priority": 2
                }]
                
                # Save updated patterns
                self.pattern_manager.save_patterns(
                    self.pattern_manager.base_patterns, 
                    'mapping_patterns.json'
                )
                
                # Update embeddings
                embedding_manager.update_section(section, pattern_manager.base_patterns[section])
                
                # Store for retraining
                pattern_manager.add_confirmed_mapping(label, new_key, section)
                
                self.llm_history[label] = new_key
                logger.info(f"Created new key: {new_key} for '{label}'")
                return new_key

# ------------------ Adaptive Mapper ------------------
class AdaptiveFinancialMapper:
    def __init__(self, base_matcher, dynamic_matcher, llm_matcher):
        self.base_matcher = base_matcher
        self.dynamic_matcher = dynamic_matcher
        self.llm_matcher = llm_matcher
        
    def match(self, label: str, section: str) -> Optional[str]:
        norm_label = enhanced_normalization(label)
        
        # Phase 1: Base matching
        match = self.base_matcher.match(norm_label, section)
        if match:
            return match
            
        # Phase 2: Dynamic pattern matching
        match = self.dynamic_matcher.match(norm_label, section)
        if match:
            return match
            
        # Phase 3: LLM-assisted mapping
        if config.enable_llm:
            return self.llm_matcher.match(label, section)
            
        return None
  ##################################################################      
    def normalize_label(self, label: str) -> str:
        """Enhanced normalization preserving key financial terms"""
    # Preserve gain/loss indicators
        label = re.sub(r'\((gain|loss)\)', r'_\1_', label)
    # Remove parentheses but keep content
        label = re.sub(r'[()]', '', label)
    # Normalize characters
        label = re.sub(r'[^a-zA-Z0-9]+', ' ', label).lower()
    # Remove common financial modifiers
        label = re.sub(
            r'\b(million|thousand|usd|%|percent|consolidated|unaudited)\b', 
        '', 
        label
    )
        return re.sub(r'\s+', '', label)

# Initialize matchers
base_matcher = BaseMatcher(config, pattern_manager, embedding_manager)
dynamic_matcher = DynamicMatcher(pattern_manager)
llm_matcher = LLMMatcher(config, pattern_manager, azure_client, hitl_manager)

# Create adaptive mapper
adaptive_mapper = AdaptiveFinancialMapper(
    base_matcher=base_matcher,
    dynamic_matcher=dynamic_matcher,
    llm_matcher=llm_matcher
)

# ------------------ Feedback Loop ------------------
class RetrainScheduler:
    def __init__(self, pattern_manager, embedding_manager):
        self.pattern_manager = pattern_manager
        self.embedding_manager = embedding_manager
        self.last_retrain = datetime.now()
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        
    def run(self):
        while self.running:
            now = datetime.now()
            if (now - self.last_retrain) > timedelta(hours=config.retrain_interval_hours):
                self.retrain()
                self.last_retrain = now
            time.sleep(3600)  # Check hourly
            
    def retrain(self):
        with self.lock:
            logger.info("Starting retraining with confirmed mappings...")
            try:
                # 1. Reinforce confirmed mappings in base patterns
                for section, mappings in self.pattern_manager.confirmed_mappings.items():
                    for item in mappings:
                        label = item['original_label']
                        key = item['key']
                        pattern = PatternManager.generate_pattern(label)
                        
                        if key not in self.pattern_manager.base_patterns[section]:
                            self.pattern_manager.base_patterns[section][key] = []
                            
                        # Add pattern with high priority
                        self.pattern_manager.base_patterns[section][key].append({
                            "pattern": pattern,
                            "priority": 1
                        })
                
                # 2. Save updated patterns
                self.pattern_manager.save_patterns(
                    self.pattern_manager.base_patterns, 
                    'mapping_patterns.json'
                )
                
                # 3. Update embeddings
                for section in ['income', 'balance', 'cashflow']:
                    self.embedding_manager.update_section(
                        section, 
                        self.pattern_manager.base_patterns.get(section, {})
                    )
                
                # 4. Clear dynamic patterns
                self.pattern_manager.dynamic_patterns = {'income': {}, 'balance': {}, 'cashflow': {}}
                self.pattern_manager.save_patterns(
                    self.pattern_manager.dynamic_patterns, 
                    'dynamic_patterns.json'
                )
                self.pattern_manager.compiled_dynamic = {}
                
                logger.info("Retraining completed successfully")
                
            except Exception as e:
                logger.error(f"Retraining failed: {e}")

# Start retraining scheduler
retrain_scheduler = RetrainScheduler(pattern_manager, embedding_manager)

# ------------------ Data Processing ------------------
def smart_value_conversion(series: pd.Series) -> pd.Series:
    """Robust conversion of financial values"""
    s = series.astype(str)
    s = s.str.replace('[\$,]', '', regex=True)
    s = s.str.replace(r'^\((.+)\)$', r'-\1', regex=True)
    # Handle per-share values
    if any('pershare' in str(x).lower() for x in series.index):
        return pd.to_numeric(s, errors='coerce')
    return pd.to_numeric(s, errors='coerce')

def promote_date_row(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced date detection with SEC-specific patterns"""
    # Check for multi-period structure
    if any("Months Ended" in str(cell) for cell in df.iloc[0].values):
        return process_multi_period_header(df)
    date_patterns = [
        re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},\s+\d{4}', re.IGNORECASE),
        re.compile(r'\d{4}-\d{2}-\d{2}'),
        re.compile(r'\d{2}/\d{2}/\d{4}'),
        re.compile(r'\d{4}_\d{2}_\d{2}'),
        re.compile(r'\d{2}-[A-Za-z]{3}-\d{4}')  # SEC-specific: 31-Dec-2024
    ]

    for i in range(min(5, len(df))):  # Check first 5 rows
        match_count = df.iloc[i].astype(str).apply(
            lambda x: any(p.search(x) for p in date_patterns)
        ).sum()
        
        if match_count >= 2:  # Require at least 2 date-like columns
            df.columns = df.iloc[i].values
            clean_df = df.iloc[i+1:]
            
            # Remove ALL non-data rows (any row without financial values)
            clean_df = filter_non_data_rows(clean_df)
            return clean_df.reset_index(drop=True)
    
    return filter_non_data_rows(df)

def process_multi_period_header(df: pd.DataFrame) -> pd.DataFrame:
    """Process statements with multiple reporting periods"""
    # Handle empty dataframes
    if len(df) < 2:
        return df
    
    # Extract period headers
    new_columns = [df.columns[0]]  # Keep label column
    last_period = None

    # Build new headers column by column
    for col_idx in range(1, len(df.columns)):
        # Handle potential index issues
        if col_idx >= len(df.iloc[0]):
            new_columns.append("Unknown Period")
            continue
            
        period_header = df.iloc[0, col_idx]
        date = df.iloc[1, col_idx] if len(df) > 1 else ""
        
        # Handle merged/empty headers
        if not period_header or str(period_header).strip() == '':
            period_header = last_period
        else:
            last_period = period_header
        
        # Skip if no valid period header
        if not period_header:
            new_columns.append(f"Column_{col_idx}")
            continue
            
        # Normalize date format
        date = re.sub(r'Mar\.', 'March', str(date))
        new_columns.append(f"{period_header} {date}")

    # Create new dataframe
    clean_df = df.iloc[2:].copy()
    clean_df.columns = new_columns
    
    # Remove ALL non-data rows
    clean_df = filter_non_data_rows(clean_df)
    
    # Set first column as index if possible
    if clean_df.shape[0] > 0 and clean_df.iloc[:, 0].dtype == object:
        clean_df = clean_df.set_index(clean_df.columns[0])
    
    return clean_df

def filter_non_data_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove any row that doesn't contain financial data"""
    # Create list to track which rows to keep
    keep_rows = []
    
    # We'll iterate by integer position, not index label
    for i in range(len(df)):
        row = df.iloc[i]
        
        # Check if all data columns are empty
        if row.iloc[1:].isnull().all() or row.iloc[1:].astype(str).str.strip().eq('').all():
            continue  # Skip this row
            
        # Check for non-data patterns in label column
        label = str(row.iloc[0])
        non_data_patterns = [
            r'abstract',
            r'\[\d+\]',  # Footnotes like [1]
            #r'\(.*\)',    # Parenthetical explanations
            #r'share',     # Share information (handled separately)
            r'amounts? in',
            r'millions?|thousands?',
            r'currency',
            r'unaudited',
            r'see accompanying',
            r'note \d',
            r'table of contents',
            r'^\s*$'      # Empty strings
        ]
        
        if any(re.search(pattern, label, re.IGNORECASE) for pattern in non_data_patterns):
            continue  # Skip this row
            
        # Check if values are numeric or contain dollar signs
        values = row.iloc[1:].astype(str)
        has_numeric = any(re.match(r'^[\$\-\d,\.\s\(\)]+$', v.strip()) for v in values)
        
        if has_numeric:
            keep_rows.append(i)
    
    return df.iloc[keep_rows]
##############################
def validate_essential_fields(mappings: Dict, section: str) -> bool:
    """Validation with dynamic equivalents"""
    essentials = config.essential_fields.get(section, [])
    missing = []
    
    for e in essentials:
        equivalents = [e] + config.dynamic_equivalents.get(e, [])
        if not any(equiv in mappings for equiv in equivalents):
            missing.append(e)
            
    if missing:
        logger.warning(f"Missing essential fields for {section}: {missing}")
    return not missing
def enhanced_normalization(label: str) -> str:
    label = re.sub(r'per\s+(common\s+)?share', '_per_share', label, flags=re.IGNORECASE)
    label = re.sub(r'\([^)]*\)', '', label)
    label = re.sub(r'[^a-zA-Z0-9]+', ' ', label).lower()
    label = re.sub(r'\b(million|thousand|usd|%|percent)\b', '', label)
    return re.sub(r'\s+', '', label)
# ------------------ Main Extraction Pipeline ------------------
def extract_financial_data(df: pd.DataFrame, section: str) -> Dict:
    """SEC financial statement extraction pipeline"""
    logger.info(f"Starting {section} statement extraction")
    start_time = time.time()
    
    # Preprocess dataframe
    df_processed = promote_date_row(df.copy())

    # Special handling for per-share section
    per_share_section = False
    if any('pershare' in str(x).lower() for x in df_processed.index):
        per_share_section = True
        logger.info("Detected per-share data section")
    # Set first column as index if suitable
    if df_processed.iloc[:, 0].dtype == object:
        df_processed = df_processed.set_index(df_processed.columns[0])
    
    # Convert index to string and clean
    df_processed.index = df_processed.index.astype(str).str.replace(
        r'[^a-zA-Z0-9\s]', '', regex=True
    ).str.strip().str.lower()

     
    
    logger.info(f"Processed index sample: {df_processed.index[:10].tolist()}")
    
    mappings = {}
    stats = {'matched': 0, 'skipped': 0}
    skipped_rows = {}
    
    # Iterate using index positions to avoid KeyErrors
    for i in range(len(df_processed)):
        raw_label = df_processed.index[i]
        row_values = df_processed.iloc[i]
        
        # Skip blank rows
        if row_values.isna().all() or row_values.astype(str).str.strip().eq('').all():
            stats['skipped'] += 1
            skipped_rows[raw_label] = "Blank row"
            continue

        # Use adaptive mapper
        if per_share_section:
            key = adaptive_mapper.match(raw_label + " per share", section)
        else:
            key = adaptive_mapper.match(raw_label, section)
        
        if key:
            val = smart_value_conversion(row_values)
            if not val.isna().all():
                mappings[key] = val
                stats['matched'] += 1
            else:
                stats['skipped'] += 1
                skipped_rows[raw_label] = "All values NaN after conversion"
        else:
            stats['skipped'] += 1
            skipped_rows[raw_label] = "No matching key found"

    # Process HITL queue if enabled
    if config.enable_hitl and hitl_manager.review_queue:
        hitl_manager.process_queue(df_processed)
        # Re-run mapping for reviewed items
        for row_label in df_processed.index:
            if row_label not in mappings:
                key = adaptive_mapper.match(str(row_label).strip(), section)
                if key:
                    val = smart_value_conversion(df_processed.loc[row_label])
                    if not val.isna().all():
                        mappings[key] = val
                        stats['matched'] += 1
                        stats['skipped'] -= 1

    # Essential field validation with HITL fallback
    essentials = config.essential_fields.get(section, [])
    expanded = []
    for e in essentials:
        expanded.append(e)
        expanded.extend(config.dynamic_equivalents.get(e, []))
    
    missing = [f for f in expanded if f not in mappings]
    
    # HITL for missing essential fields
    if missing and config.enable_hitl:
        manual_mappings = hitl_manager.handle_missing_fields(missing, section, df_processed)
        for field, label in manual_mappings.items():
            if label in df_processed.index:
                mappings[field] = smart_value_conversion(df_processed.loc[label])
                stats['matched'] += 1
                logger.info(f"Manually mapped {label} -> {field}")
                # Add to confirmed mappings
                pattern_manager.add_confirmed_mapping(label, field, section)
    
    # Final validation check
    valid = validate_essential_fields(mappings, section)
    
    # Log performance
    duration = time.time() - start_time
    logger.info(
        f"Extracted {stats['matched']} items from {section} "
        f"({stats['skipped']} skipped) in {duration:.2f}s. "
        f"Validation: {'PASS' if valid else 'FAIL'}"
    )
    if skipped_rows:
        print("\nDEBUG: Skipped Rows")
        for label, reason in skipped_rows.items():
            print(f"- {label}: {reason}")
    return {
        'data': mappings,
        'validation': {'section': section, 'valid': valid},
        'stats': stats
        #'skipped_rows': skipped_rows
    }

# ------------------ Unit Test Stubs ------------------
import unittest
from unittest.mock import MagicMock, patch

class FinancialParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Disable logging during tests
        logging.disable(logging.CRITICAL)
    def setUp(self):
        # Create isolated config for tests
        global config
        self.orig_config = config
        self.test_config = Config()
        self.test_config.enable_hitl = False
        
        config = self.test_config
    
    def tearDown(self):
        # Restore original config
        global config
        config = self.orig_config
        
    def test_base_matcher(self):
        matcher = BaseMatcher(config, pattern_manager, embedding_manager)
        self.assertEqual(matcher.match("totalrevenue", "income"), "total_revenue")
        
    def test_normalization(self):
        mapper = AdaptiveFinancialMapper(None, None, None)
        self.assertEqual(
            mapper.normalize_label("Inventory (net of obsolescence)"), 
            "inventorynetofobsolescence"
        )
        
    @patch.object(AzureClient, 'get_completion')
    def test_llm_matcher(self, mock_llm):
        mock_llm.return_value = json.dumps({
            "match_type": "existing",
            "key": "net_income",
            "confidence": 99  # High confidence to bypass HITL
        })
        matcher = LLMMatcher(config, pattern_manager, azure_client, hitl_manager)
        self.assertEqual(matcher.match("Profit After Tax", "income"), "net_income")
        
    def test_value_conversion(self):
        series = pd.Series(["$1,000", "(500)", "N/A"])
        converted = smart_value_conversion(series)
        # Check values and types
        self.assertEqual(converted.iloc[0], 1000.0)
        self.assertEqual(converted.iloc[1], -500.0)
        self.assertTrue(np.isnan(converted.iloc[2]))
        
    def test_essential_validation(self):
        # Create test-specific config
        test_config = Config()
        test_config.essential_fields = {'cashflow': ['net_cash_operating']}
        test_config.dynamic_equivalents = {
        'net_cash_operating': ['cash_from_operations']
    }
    
    # Replace global config with test config
        global config
        original_config = config
        config = test_config
    
        try:
        # Test with equivalent field
            mappings = {'cash_from_operations': pd.Series([100])}
            self.assertTrue(validate_essential_fields(mappings, "cashflow"))
        
        # Test with direct field
            mappings = {'net_cash_operating': pd.Series([100])}
            self.assertTrue(validate_essential_fields(mappings, "cashflow"))
        
        # Test missing field
            mappings = {'other_field': pd.Series([100])}
            self.assertFalse(validate_essential_fields(mappings, "cashflow"))
        finally:
        # Restore original config
            config = original_config
        
    def test_hitl_enabled(self):
        config.enable_hitl = True
        self.assertTrue(config.enable_hitl)
        
    def test_hitl_disabled(self):
        config.enable_hitl = False
        self.assertFalse(config.enable_hitl)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("FINANCIAL DATA PARSER CONFIGURATION".center(60))
    print("="*60)
    user_expertise = input("Are you experienced in financial domain analysis? (yes/no): ").strip().lower()
    enable_hitl = user_expertise in ['y', 'yes']
    unittest.main(argv=[''], exit=False)