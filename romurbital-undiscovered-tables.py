import numpy as np
from scipy.spatial.distance import cosine
import json
import pandas as pd
from pathlib import Path
import traceback
import glob

def calculate_network_centrality(embedding, all_embeddings):
    """Calculate how central a node is in the network based on average similarity to all other nodes"""
    similarities = [1 - cosine(embedding, other_emb) for other_emb in all_embeddings]
    return np.mean(similarities)

def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings"""
    return 1 - cosine(embedding1, embedding2)

def calculate_discovery_similarity(embedding, discovered_embeddings, discovered_names):
    """Calculate the similarity to the closest discovered site"""
    # Find closest discovered place
    discovered_sims = [(calculate_similarity(embedding, disc_emb), name) for disc_emb, name in zip(discovered_embeddings, discovered_names)]
    
    # Get the closest match
    if discovered_sims:
        closest_sim, closest_name = max(discovered_sims, key=lambda x: x[0])
        return closest_sim, closest_name
    else:
        return 0, "None"

def create_site_comparison_tables(embeddings_file, metadata_file):
    """Create tables comparing undiscovered sites and romurbital sites to their most similar sites"""
    file_stem = Path(embeddings_file).stem.replace('_embeddings', '').replace('_single_sent_embeddings', '').replace('_three_sent_embeddings', '')
    print(f"\nProcessing {file_stem}...")
    
    # Load embeddings
    try:
        data = np.load(embeddings_file)
        embeddings = {word: data[word] for word in data.files}
    except Exception as e:
        print(f"Error loading embeddings from {embeddings_file}: {str(e)}")
        return None, None
    
    # Load metadata
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Error loading metadata from {metadata_file}: {str(e)}")
        return None, None
    
    # Check if there are enough terms to create a meaningful comparison
    if len(embeddings) < 3:
        print(f"Warning: {file_stem} has only {len(embeddings)} terms, skipping comparison")
        return None, None
    
    # Initialize lists for different types of places
    discovered_places = []    # Romurbital sites (known locations)
    undiscovered_places = []  # Unidentified sites (locations not yet identified)
    
    # Process each word in the embeddings
    for word, embedding in embeddings.items():
        if word not in metadata:
            print(f"Warning: {word} not found in metadata, skipping")
            continue
            
        category = metadata[word]['category'].lower()
        word_display = word.replace('_', ' ')
        
        # Categorize places
        if 'romurbital' in category:
            discovered_places.append((word_display, embedding))
        elif 'unidentified' in category:
            undiscovered_places.append((word_display, embedding))
    
    # Check if we have enough data
    if len(discovered_places) < 2:  # Need at least 2 discovered places to compare
        print(f"Warning: {file_stem} doesn't have enough discovered places for comparison (discovered: {len(discovered_places)})")
        return None, None
    
    # Calculate network centrality
    all_embeddings_list = []
    for _, emb in discovered_places + undiscovered_places:
        if isinstance(emb, np.ndarray):
            all_embeddings_list.append(emb)
        else:
            print(f"Warning: Found embedding of type {type(emb)}, expected numpy array")
    
    centrality_scores = {}
    for word, emb in discovered_places + undiscovered_places:
        if isinstance(emb, np.ndarray):
            centrality_scores[word] = calculate_network_centrality(emb, all_embeddings_list)
        else:
            print(f"Warning: Skipping centrality calculation due to invalid embedding type")
            centrality_scores[word] = 0.5  # Default value
    
    # Extract discovered site embeddings and names
    discovered_embeddings = []
    discovered_names = []
    for name, emb in discovered_places:
        if isinstance(emb, np.ndarray):
            discovered_embeddings.append(emb)
            discovered_names.append(name)
    
    # ---------------------------------------------------------------
    # TABLE 1: UNDISCOVERED SITES
    # ---------------------------------------------------------------
    
    # Data for undiscovered sites table
    undiscovered_table_data = []
    
    for word, emb in undiscovered_places:
        if not isinstance(emb, np.ndarray):
            print(f"Warning: Skipping {word} due to invalid embedding type")
            continue
            
        # Calculate similarity to discovered sites
        try:
            similarity, closest_site = calculate_discovery_similarity(emb, discovered_embeddings, discovered_names)
            
            # Add to table data
            undiscovered_table_data.append({
                "Undiscovered Site": word,
                "Most Similar To": closest_site,
                "Similarity Score": similarity,
                "Centrality Score": centrality_scores.get(word, 0),
                "Most Similar Site Centrality": centrality_scores.get(closest_site, 0)
            })
            
        except Exception as e:
            print(f"Error processing undiscovered place {word}: {str(e)}")
            traceback.print_exc()
    
    # Create the undiscovered sites dataframe
    if undiscovered_table_data:
        undiscovered_df = pd.DataFrame(undiscovered_table_data)
        # Sort by similarity score (descending)
        undiscovered_df = undiscovered_df.sort_values(by="Similarity Score", ascending=False)
    else:
        undiscovered_df = None
    
    # ---------------------------------------------------------------
    # TABLE 2: ROMURBITAL (DISCOVERED) SITES
    # ---------------------------------------------------------------
    
    # Data for romurbital sites table
    romurbital_table_data = []
    
    for i, (word, emb) in enumerate(discovered_places):
        if not isinstance(emb, np.ndarray):
            print(f"Warning: Skipping romurbital site {word} due to invalid embedding type")
            continue
        
        # Calculate similarity to other discovered sites (excluding self)
        try:
            # Create a list of other discovered sites (excluding self)
            other_discovered_embs = [e for j, (_, e) in enumerate(discovered_places) if j != i and isinstance(e, np.ndarray)]
            other_discovered_names = [n for j, (n, _) in enumerate(discovered_places) if j != i]
            
            if other_discovered_embs:
                similarity, closest_site = calculate_discovery_similarity(emb, other_discovered_embs, other_discovered_names)
                
                # Add to table data
                romurbital_table_data.append({
                    "Romurbital Site": word,
                    "Most Similar To": closest_site,
                    "Similarity Score": similarity,
                    "Centrality Score": centrality_scores.get(word, 0),
                    "Most Similar Site Centrality": centrality_scores.get(closest_site, 0)
                })
        except Exception as e:
            print(f"Error processing romurbital place {word}: {str(e)}")
            traceback.print_exc()
    
    # Create the romurbital sites dataframe
    if romurbital_table_data:
        romurbital_df = pd.DataFrame(romurbital_table_data)
        # Sort by similarity score (descending)
        romurbital_df = romurbital_df.sort_values(by="Similarity Score", ascending=False)
    else:
        romurbital_df = None
    
    return undiscovered_df, romurbital_df

def save_tables_to_csv(undiscovered_df, romurbital_df, output_dir, file_stem):
    """Save the tables to CSV files"""
    if undiscovered_df is not None:
        output_file = f"{output_dir}/{file_stem}_undiscovered_comparison.csv"
        undiscovered_df.to_csv(output_file, index=False)
        print(f"Saved undiscovered sites table to {output_file}")
    
    if romurbital_df is not None:
        output_file = f"{output_dir}/{file_stem}_romurbital_comparison.csv"
        romurbital_df.to_csv(output_file, index=False)
        print(f"Saved romurbital sites table to {output_file}")

# Main execution block
if __name__ == "__main__":
    # Hardcoded directory paths
    base_dir = r"C:\Users\User\Downloads\embeddings_three_sentence_output-20250326T165516Z-001\embeddings_three_sentence_output"  # Replace with your actual directory path
    output_dir = r"C:\Users\User\Downloads\embeddings_three_sentence_output-20250326T165516Z-001\embeddings_three_sentence_output\tables3"  # Replace with where you want results saved
    
    # Automatically find all embedding files in the directory
    embedding_files = [Path(f).name for f in glob.glob(f"{base_dir}/*_embeddings.npz")]
    
    if not embedding_files:
        print(f"No embedding files found in {base_dir}")
        print("Looking for files matching pattern: *_embeddings.npz")
        exit(1)
    
    print(f"Found {len(embedding_files)} embedding files to process:")
    for filename in embedding_files:
        print(f"  - {filename}")
    print()
    
    # Process each embedding file
    for embedding_filename in embedding_files:
        embeddings_file = f"{base_dir}/{embedding_filename}"
        
        # Generate metadata filename based on embedding filename pattern
        metadata_filename = embedding_filename.replace("_embeddings.npz", "_metadata.json")
        metadata_file = f"{base_dir}/{metadata_filename}"
        
        print(f"Processing {embedding_filename}...")
        
        # Get file stem for naming output files
        file_stem = Path(embedding_filename).stem.replace('_embeddings', '').replace('_single_sent_embeddings', '').replace('_three_sent_embeddings', '')
        
        # Create the tables
        undiscovered_df, romurbital_df = create_site_comparison_tables(embeddings_file, metadata_file)
        
        # Save tables to CSV files
        save_tables_to_csv(undiscovered_df, romurbital_df, output_dir, file_stem)
        
        # Print tables to console
        print("\n--- Undiscovered Sites Comparison Table ---")
        if undiscovered_df is not None:
            # Format table for better display
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            pd.set_option('display.float_format', '{:.3f}'.format)
            print(undiscovered_df)
        else:
            print("No data for undiscovered sites.")
        
        print("\n--- Romurbital Sites Comparison Table ---")
        if romurbital_df is not None:
            print(romurbital_df)
        else:
            print("No data for romurbital sites.")