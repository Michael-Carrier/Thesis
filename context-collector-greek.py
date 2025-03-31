import json
import os
import re

def load_metadata(filename):
    """Load and parse a metadata JSON file."""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_report_file(filename):
    """Parse the visualization report file to get place pairs and similarities."""
    pairs = []
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        # Modified regex to handle Greek characters and more flexible pattern matching
        matches = re.findall(r'([^\s:]+): [\d\.]+\s*-\s*Closest to (?:historical|myth): ([^\s(]+)\s*\(similarity: ([\d\.]+)', content)
        for match in matches:
            unid_place, known_place, similarity = match
            # Store original case for Greek characters
            pairs.append((unid_place, known_place, float(similarity)))
    return pairs

def get_contexts(metadata, place_name):
    """Get all contexts for a given place name."""
    contexts = []
    # Check both original case and variants
    for key in metadata:
        # Check if the key matches the place name (case-insensitive)
        if key.lower() == place_name.lower():
            for context in metadata[key].get('sample_contexts', []):
                contexts.append(context['context'])
        # Also check variants
        elif 'variants' in metadata[key]:
            variants = [v.lower() for v in metadata[key].get('variants', [])]
            if place_name.lower() in variants:
                for context in metadata[key].get('sample_contexts', []):
                    contexts.append(context['context'])
    return contexts

def process_file_pair(metadata_path, report_path):
    """Process a matching pair of metadata and report files."""
    # Get base name for header
    book_name = os.path.basename(metadata_path).replace('_metadata.json', '')
    
    # Create header with clear separation
    header = f"\n{'='*80}\n"
    header += f"CONTEXTS FROM: {book_name}\n"
    header += f"{'='*80}\n"
    
    # Load data
    metadata = load_metadata(metadata_path)
    pairs = parse_report_file(report_path)
    
    # Process each pair
    results = [header]  # Start with the header
    for unid_place, known_place, similarity in pairs:
        results.append(f"\n## {unid_place} → {known_place} (Similarity: {similarity:.3f})")
        
        # Unidentified place contexts
        unid_contexts = get_contexts(metadata, unid_place)
        results.append("\nUNIDENTIFIED PLACE CONTEXTS:")
        if unid_contexts:
            for i, context in enumerate(unid_contexts, 1):
                results.append(f"{i}. {context}")
        else:
            results.append("No contexts found")
            
        # Known place contexts
        known_contexts = get_contexts(metadata, known_place)
        results.append("\nKNOWN PLACE CONTEXTS:")
        if known_contexts:
            for i, context in enumerate(known_contexts, 1):
                results.append(f"{i}. {context}")
        else:
            results.append("No contexts found")
        
        results.append("\n" + "-"*40)  # Add separator between pairs
    
    return '\n'.join(results)

def find_matching_files(metadata_dir, report_dir):
    """Find matching metadata and report files based on common base names."""
    metadata_files = {}
    report_files = {}
    
    # Create dictionaries of files keyed by their base names
    for filename in os.listdir(metadata_dir):
        if filename.endswith('_metadata.json'):
            base_name = filename.replace('_metadata.json', '')
            metadata_files[base_name] = os.path.join(metadata_dir, filename)
            
    for filename in os.listdir(report_dir):
        if filename.endswith('_place_names_report.txt'):
            base_name = filename.replace('_place_names_report.txt', '')
            report_files[base_name] = os.path.join(report_dir, filename)
    
    # Find matches
    matches = []
    for base_name in metadata_files:
        if base_name in report_files:
            matches.append((metadata_files[base_name], report_files[base_name]))
    
    return matches

def main():
    # Directory paths
    metadata_dir = r"C:\Users\User\Documents\random py scripts\thesis\latintagger\books and count\Greeklist\greekbooks\greekbooks\embeddings_three_sentence_output"
    report_dir = r"C:\Users\User\Documents\random py scripts\thesis\latintagger\books and count\Greeklist\greekbooks\greekbooks\embeddings_three_sentence_output\visualizations"
    
    # Find matching file pairs
    matching_files = find_matching_files(metadata_dir, report_dir)
    
    # Print found matches
    print("Found matching files:")
    for metadata_path, report_path in matching_files:
        print(f"- {os.path.basename(metadata_path)} ↔ {os.path.basename(report_path)}")
    print("\nProcessing files...")
    
    # Add debugging for the first pair to see what's happening
    if matching_files:
        metadata_path, report_path = matching_files[0]
        print(f"\nDEBUGGING for {os.path.basename(metadata_path)}:")
        
        # Debug report file content
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                report_content = f.read()
                print(f"\nFirst 500 chars of report file:\n{report_content[:500]}...")
                
            # Debug regex matches
            matches = re.findall(r'([^\s:]+): [\d\.]+\s*-\s*Closest to (?:historical|myth): ([^\s(]+)\s*\(similarity: ([\d\.]+)', report_content)
            print(f"\nFound {len(matches)} matches in report file")
            for i, match in enumerate(matches[:3], 1):  # Show first 3 matches
                print(f"Match {i}: {match}")
                
            # Debug metadata keys
            metadata = load_metadata(metadata_path)
            print(f"\nFirst 5 metadata keys: {list(metadata.keys())[:5]}")
            
        except Exception as e:
            print(f"Error during debugging: {str(e)}")
    
    # Process each pair
    all_results = []
    for metadata_path, report_path in matching_files:
        try:
            results = process_file_pair(metadata_path, report_path)
            all_results.append(results)
        except Exception as e:
            print(f"Error processing {metadata_path}: {str(e)}")
    
    # Write results to output file
    output_path = os.path.join(metadata_dir, "paired_contexts_output.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_results))
    
    print(f"\nResults written to: {output_path}")

if __name__ == "__main__":
    main()