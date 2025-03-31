import torch
from transformers import BertModel, BertTokenizer
import numpy as np
import pandas as pd
import json
import re
import os
import sys
from pathlib import Path
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add LatinBERT repository path to Python path
latinbert_path = r"C:\Users\User\latin-bert"
sys.path.append(latinbert_path)

class LatinBertTokenizer:
    """
    A custom tokenizer for Latin text that reads the vocabulary from the encoder file
    """
    def __init__(self, tokenizer_path):
        self.vocab = {}
        self.inv_vocab = {}
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[CLS]': 2,
            '[SEP]': 3,
            '[MASK]': 4
        }
        
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        
        # Load vocabulary from tokenizer file
        try:
            with open(tokenizer_path, 'r', encoding='utf-8') as f:
                content = f.readlines()
            
            # Process tokens and add to vocabulary
            for i, line in enumerate(content):
                token = line.strip()
                if token:
                    self.vocab[token] = i + len(self.special_tokens)
                    self.inv_vocab[i + len(self.special_tokens)] = token
            
            # Add special tokens
            for token, idx in self.special_tokens.items():
                self.vocab[token] = idx
                self.inv_vocab[idx] = token
                
            logger.info(f"Loaded {len(self.vocab)} tokens from vocabulary")
        except Exception as e:
            logger.error(f"Error loading vocabulary: {e}")
            raise
    
    def tokenize(self, text):
        """
        Simple tokenization for Latin text
        """
        # Add CLS token at the beginning
        tokens = ['[CLS]']
        
        # Split by whitespace and punctuation
        words = re.findall(r'\b\w+\b', text.lower())
        for word in words:
            # Check if word is in vocabulary
            if word in self.vocab:
                tokens.append(word)
            else:
                # Handle word not in vocabulary by splitting into characters
                for char in word:
                    token = char if char in self.vocab else '[UNK]'
                    tokens.append(token)
        
        # Add SEP token at the end
        tokens.append('[SEP]')
        
        return tokens
    
    def convert_tokens_to_ids(self, tokens):
        """
        Convert tokens to their IDs in the vocabulary
        """
        return [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]

def normalize_whitespace(text):
    """
    Normalize whitespace in text:
    - Replace multiple spaces with a single space
    - Replace newlines and tabs with spaces
    - Trim leading and trailing whitespace
    """
    text = re.sub(r'[\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_sentence(text, match_start, match_end):
    """
    Get only the sentence containing the term.
    """
    # Get a text window around the match
    window_start = max(0, match_start - 300)
    window_end = min(len(text), match_end + 300)
    text_window = text[window_start:window_end]
    
    # Split into sentences
    sentences = re.split(r'([.!?]+(?:\s+|$))', text_window)
    full_sentences = []
    for i in range(0, len(sentences)-1, 2):
        if i+1 < len(sentences):
            full_sentences.append(sentences[i] + sentences[i+1])
        else:
            full_sentences.append(sentences[i])
    
    # Find the sentence containing our match
    relative_match_start = match_start - window_start
    current_pos = 0
    
    for sentence in full_sentences:
        next_pos = current_pos + len(sentence)
        if current_pos <= relative_match_start < next_pos:
            context = normalize_whitespace(sentence.strip())
            
            # Verify the variant is in the context
            actual_match = text[match_start:match_end]
            if actual_match.lower() in context.lower():
                return context
            else:
                logger.warning(f"Match '{actual_match}' not found in extracted sentence")
                return None
        
        current_pos = next_pos
    
    return None

def get_sentences_around(text, match_start, match_end):
    """
    Get the sentence containing the term plus one sentence before and one after.
    """
    window_start = max(0, match_start - 300)
    window_end = min(len(text), match_end + 300)
    text_window = text[window_start:window_end]
    
    # Split into sentences
    sentences = re.split(r'([.!?]+(?:\s+|$))', text_window)
    full_sentences = []
    for i in range(0, len(sentences)-1, 2):
        if i+1 < len(sentences):
            full_sentences.append(sentences[i] + sentences[i+1])
        else:
            full_sentences.append(sentences[i])
    
    # Find the sentence containing our match
    relative_match_start = match_start - window_start
    current_pos = 0
    target_sentence_idx = None
    
    for i, sentence in enumerate(full_sentences):
        next_pos = current_pos + len(sentence)
        if current_pos <= relative_match_start < next_pos:
            target_sentence_idx = i
            break
        current_pos = next_pos
    
    if target_sentence_idx is None:
        return None
        
    # Get the sentences
    context_sentences = [full_sentences[target_sentence_idx].strip()]
    
    if target_sentence_idx > 0:
        context_sentences.insert(0, full_sentences[target_sentence_idx - 1].strip())
    
    if target_sentence_idx < len(full_sentences) - 1:
        context_sentences.append(full_sentences[target_sentence_idx + 1].strip())
    
    # Join and normalize
    context = normalize_whitespace(" ".join(context_sentences))
    
    # Verify the match is in the context
    actual_match = text[match_start:match_end]
    if actual_match.lower() not in context.lower():
        logger.warning(f"Match '{actual_match}' not found in extracted context")
        return None
        
    return context

def tokenize_latin_text(text, latin_tokenizer):
    """
    Tokenize Latin text using the Latin tokenizer
    """
    try:
        # Tokenize using Latin tokenizer
        tokens = latin_tokenizer.tokenize(text)
        token_ids = latin_tokenizer.convert_tokens_to_ids(tokens)
        
        # Create tensors
        input_ids = torch.tensor([token_ids])
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'tokens': tokens
        }
    except Exception as e:
        logger.error(f"Error tokenizing text: {e}")
        logger.error(f"Text: {text[:100]}...")
        raise

def process_single_file(text_file_path, df, tokenizer, model, use_three_sentences=True):
    """
    Process a single text file and return embeddings for terms.
    """
    # Set for tracking matched terms
    found_terms = set()
    
    file_name = os.path.basename(text_file_path)
    file_stem = Path(file_name).stem  # Get filename without extension
    
    logger.info(f"Processing file: {file_name}")
    
    # Read text file with error handling
    try:
        with open(text_file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        logger.info(f"Loaded text file: {len(text)} characters")
    except UnicodeDecodeError:
        # If UTF-8 fails, try Latin-1
        with open(text_file_path, 'r', encoding='latin-1') as file:
            text = file.read()
        logger.info(f"Loaded text file with Latin-1 encoding: {len(text)} characters")
    
    # Dictionary to store all embeddings and metadata for this file
    all_embeddings = {}
    
    for idx, row in df.iterrows():
        word = row['word'].strip()
        category = row['category'].strip()
        # Get all non-empty variants
        variants = [str(v).strip() for v in row[2:] if pd.notna(v) and str(v).strip()]
        
        if idx % 10 == 0:
            logger.info(f"Processing term {idx + 1}/{len(df)}: {word}")
        
        # Find contexts
        contexts = []
        
        # Search for each variant using word boundaries
        for variant in variants + [word]:
            # Create pattern with word boundaries
            pattern = r'\b' + re.escape(variant) + r'\b'
            # Find all matches with word boundaries
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start_idx = match.start()
                end_idx = match.end()
                
                # Get context based on mode
                if use_three_sentences:
                    context = get_sentences_around(text, start_idx, end_idx)
                else:
                    context = get_sentence(text, start_idx, end_idx)
                
                if context:
                    contexts.append({
                        'context': context,
                        'variant_found': variant
                    })
        
        if contexts:
            found_terms.add(word)
            if len(contexts) > 3:  # Only log for terms with reasonable number of matches
                logger.info(f"  Found {len(contexts)} contexts for '{word}'")
            
            # Generate embeddings for contexts
            context_embeddings = []
            for context_data in contexts:
                context = context_data['context']
                try:
                    # Use the Latin tokenizer
                    encoded = tokenize_latin_text(context, tokenizer)
                    
                    with torch.no_grad():
                        outputs = model(
                            input_ids=encoded['input_ids'],
                            attention_mask=encoded['attention_mask']
                        )
                        context_embedding = outputs.last_hidden_state.mean(dim=1).numpy()[0]
                        context_embeddings.append(context_embedding)
                except Exception as e:
                    logger.error(f"Error processing context for '{word}': {e}")
                    continue
            
            if context_embeddings:
                # Average the embeddings
                final_embedding = np.mean(context_embeddings, axis=0)
                
                # Store in dictionary
                all_embeddings[word] = {
                    'embedding': final_embedding,
                    'category': category,
                    'variants': variants,
                    'context_count': len(contexts),
                    'sample_contexts': [contexts[i] for i in range(min(3, len(contexts)))]
                }
    
    # Count how many terms were found
    terms_found = len(all_embeddings)
    logger.info(f"Found embeddings for {terms_found}/{len(df)} terms in {file_name}")
    logger.info(f"Terms found: {', '.join(sorted(list(found_terms)[:10]))}{'...' if len(found_terms) > 10 else ''}")
    
    return file_stem, all_embeddings, found_terms

def get_term_embeddings(texts_dir_path, csv_file_path, use_three_sentences=True):
    logger.info("1. Reading CSV file...")
    # Read CSV file
    df = pd.read_csv(csv_file_path)
    logger.info(f"Found {len(df)} terms in CSV file")
    
    # Initialize LatinBERT model and tokenizer
    logger.info("2. Initializing LatinBERT model and tokenizer...")
    
    # CORRECTED Path for LatinBERT model - with the extra level of nesting
    model_path = os.path.join(latinbert_path, "models", "latin_bert", "latin_bert")
    tokenizer_path = os.path.join(latinbert_path, "models", "subword_tokenizer_latin", "latin.subword.encoder")
    
    # Check if paths exist
    if not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not os.path.exists(tokenizer_path):
        logger.error(f"Tokenizer path does not exist: {tokenizer_path}")
        raise FileNotFoundError(f"Tokenizer path not found: {tokenizer_path}")
    
    logger.info(f"Loading model from: {model_path}")
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    
    # Initialize Latin tokenizer
    tokenizer = LatinBertTokenizer(tokenizer_path)
    
    # Load the BERT model
    model = BertModel.from_pretrained(model_path)
    model.eval()
    
    model_name = "latinbert-authentic"
    
    # Get list of text files
    text_files = [os.path.join(texts_dir_path, f) for f in os.listdir(texts_dir_path) 
                  if f.endswith('.txt') and os.path.isfile(os.path.join(texts_dir_path, f))]
    
    logger.info(f"3. Found {len(text_files)} text files to process")
    
    # Process each file
    all_embeddings_by_file = {}
    all_terms_by_file = {}
    for text_file in text_files:
        file_stem, embeddings, found_terms = process_single_file(
            text_file, df, tokenizer, model, use_three_sentences)
        all_embeddings_by_file[file_stem] = embeddings
        # Convert set to list for JSON serialization
        all_terms_by_file[file_stem] = list(found_terms)
    
    logger.info("4. Saving results...")
    # Create output directory based on context mode
    context_mode = "three_sent" if use_three_sentences else "single_sent"
    output_dir = os.path.join(os.path.dirname(texts_dir_path), f"{model_name}_embeddings_{context_mode}_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # For each file, save its embeddings and metadata
    for file_stem, embeddings in all_embeddings_by_file.items():
        if not embeddings:
            logger.info(f"No embeddings found for {file_stem}, skipping")
            continue
            
        # Save embeddings
        embeddings_dict = {word: data['embedding'] for word, data in embeddings.items()}
        np.savez(os.path.join(output_dir, f"{file_stem}_{context_mode}_{model_name}_embeddings.npz"), **embeddings_dict)
        
        # Save metadata
        metadata = {
            word: {
                'category': data['category'],
                'variants': data['variants'],
                'context_count': data['context_count'],
                'sample_contexts': data['sample_contexts']
            }
            for word, data in embeddings.items()
        }
        
        with open(os.path.join(output_dir, f"{file_stem}_{context_mode}_{model_name}_metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
    # Also save a summary file
    summary = {
        file_stem: {
            'total_terms': len(embeddings),
            'terms_with_embeddings': list(embeddings.keys())
        }
        for file_stem, embeddings in all_embeddings_by_file.items()
    }
    
    # Save a comparison file to debug differences between runs
    with open(os.path.join(output_dir, f"terms_by_file_{context_mode}.json"), 'w', encoding='utf-8') as f:
        json.dump(all_terms_by_file, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(output_dir, f"{context_mode}_{model_name}_embeddings_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"All files processed and saved to: {output_dir}")
    logger.info("Files created:")
    for file_stem in all_embeddings_by_file.keys():
        logger.info(f"- {file_stem}_{context_mode}_{model_name}_embeddings.npz - Contains word embeddings for {file_stem}")
        logger.info(f"- {file_stem}_{context_mode}_{model_name}_metadata.json - Contains categories, variants, and sample contexts for {file_stem}")
    logger.info(f"- {context_mode}_{model_name}_embeddings_summary.json - Summary of terms found in each file")
    logger.info(f"- terms_by_file_{context_mode}.json - Detailed list of terms found in each file (for debugging)")
    
    return all_embeddings_by_file

# File paths
texts_dir_path = r'C:\Users\User\Documents\random py scripts\thesis\latintagger\books and count\Greeklist\greekbooks\greekbooks'
csv_file_path = r'C:\Users\User\Documents\random py scripts\thesis\latintagger\books and count\Greeklist\Greek_declensions.csv'

# Choose context mode:
# False = single sentence context (original)
# True = three sentence context
use_three_sentences = False  # Change to False for single sentence mode

# Main execution
if __name__ == "__main__":
    try:
        # Generate all embeddings using LatinBERT with its tokenizer
        embeddings_by_file = get_term_embeddings(texts_dir_path, csv_file_path, use_three_sentences)
        
        context_mode = "three_sent" if use_three_sentences else "single_sent"
        print(f"""
        To load these embeddings in another script, use:
        
        import numpy as np
        import json
        import os

        # Load embeddings for a specific file
        file_stem = "YourFileName"  # Without .txt extension
        model_name = "latinbert-authentic"
        data = np.load(f"{{file_stem}}_{context_mode}_{{model_name}}_embeddings.npz")
        embeddings = {{word: data[word] for word in data.files}}

        # Load metadata
        with open(f"{{file_stem}}_{context_mode}_{{model_name}}_metadata.json", 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        """)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Fallback to using multilingual BERT
        logger.info("Falling back to multilingual BERT...")
        
        from transformers import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = BertModel.from_pretrained('bert-base-multilingual-cased')
        
        # Continue with fallback implementation...
