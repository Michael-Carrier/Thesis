import numpy as np
try:
    import plotly.graph_objects as go
except ImportError:
    # Try alternative import in case the module is installed differently
    import plotly
    go = plotly.graph_objects
from plotly.subplots import make_subplots
from scipy.spatial.distance import cosine
import json
from collections import Counter
import os
from pathlib import Path
import traceback
import math

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

def map_similarity_to_position(similarity):
    """Map similarity score to position on x-axis using a linear scale
    
    This function transforms the similarity score (0-1) to an x-axis position (0-100)
    so that the position directly reflects the similarity percentage.
    """
    # Direct linear mapping from similarity to position (0-1 → 0-100)
    position = similarity * 100
    
    return position

def create_discovery_visualization(embeddings_file, metadata_file, output_dir):
    """Create spectrum visualization showing undiscovered sites and their relation to discovered sites and myth sites"""
    file_stem = Path(embeddings_file).stem.replace('_embeddings', '').replace('_single_sent_embeddings', '').replace('_three_sent_embeddings', '')
    print(f"\nProcessing {file_stem}...")
    
    # Load embeddings
    try:
        data = np.load(embeddings_file)
        embeddings = {word: data[word] for word in data.files}
    except Exception as e:
        print(f"Error loading embeddings from {embeddings_file}: {str(e)}")
        return None
    
    # Load metadata
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Error loading metadata from {metadata_file}: {str(e)}")
        return None
    
    # Check if there are enough terms to create a meaningful visualization
    if len(embeddings) < 3:
        print(f"Warning: {file_stem} has only {len(embeddings)} terms, skipping visualization")
        return None
    
    # Initialize lists for different types of places
    discovered_places = []    # Romurbital sites (known locations)
    undiscovered_places = []  # Unidentified sites (locations not yet identified)
    myth_places = []          # Myth sites (mythological locations)
    
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
        elif 'myth' in category:
            myth_places.append((word_display, embedding))
    
    # Check if we have enough data
    if len(discovered_places) < 2:  # Need at least 2 discovered places to compare
        print(f"Warning: {file_stem} doesn't have enough discovered places for visualization (discovered: {len(discovered_places)})")
        return None
    
    # Calculate network centrality
    all_embeddings_list = []
    for _, emb in discovered_places + undiscovered_places + myth_places:
        if isinstance(emb, np.ndarray):
            all_embeddings_list.append(emb)
        else:
            print(f"Warning: Found embedding of type {type(emb)}, expected numpy array")
    
    centrality_scores = {}
    for word, emb in discovered_places + undiscovered_places + myth_places:
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
    
    # Extract myth site embeddings and names
    myth_embeddings = []
    myth_names = []
    for name, emb in myth_places:
        if isinstance(emb, np.ndarray):
            myth_embeddings.append(emb)
            myth_names.append(name)
    
    # ---------------------------------------------------------------
    # ORIGINAL VISUALIZATION: UNDISCOVERED SITES
    # ---------------------------------------------------------------
    
    # Initialize dictionaries to store similarity data for both discovered and myth sites
    historical_similarities = {}  # {undiscovered_name: (similarity, position, closest_name)}
    myth_similarities = {}        # {undiscovered_name: (similarity, position, closest_name)}
    
    # Calculate similarities for all undiscovered places to both discovered and myth sites
    for word, emb in undiscovered_places:
        if not isinstance(emb, np.ndarray):
            print(f"Warning: Skipping {word} due to invalid embedding type")
            continue
            
        # Calculate similarity to discovered sites
        try:
            hist_similarity, hist_closest = calculate_discovery_similarity(emb, discovered_embeddings, discovered_names)
            hist_position = map_similarity_to_position(hist_similarity)
            historical_similarities[word] = (hist_similarity, hist_position, hist_closest)
        except Exception as e:
            print(f"Error processing undiscovered place {word} (historical): {str(e)}")
            traceback.print_exc()
            historical_similarities[word] = (0, 0, "None")
        
        # Calculate similarity to myth sites
        if myth_embeddings:
            try:
                # Find closest myth place
                myth_sims = [(calculate_similarity(emb, myth_emb), name) for myth_emb, name in zip(myth_embeddings, myth_names)]
                myth_closest_sim, myth_closest_name = max(myth_sims, key=lambda x: x[0])
                
                # Use same tiered scale for visualization
                myth_position = map_similarity_to_position(myth_closest_sim)
                
                myth_similarities[word] = (myth_closest_sim, myth_position, myth_closest_name)
            except Exception as e:
                print(f"Error processing undiscovered place {word} (myth): {str(e)}")
                traceback.print_exc()
                myth_similarities[word] = (0, 0, "None")
        else:
            myth_similarities[word] = (0, 0, "None")
    
    # Now decide where each undiscovered site belongs based on highest similarity
    historical_assigned = []  # [(word, position, similarity, closest)]
    myth_assigned = []       # [(word, position, similarity, closest)]
    
    connected_nodes = Counter()         # For historical connections
    myth_connected_nodes = Counter()    # For myth connections
    
    for word in historical_similarities:
        hist_sim, hist_pos, hist_closest = historical_similarities[word]
        myth_sim, myth_pos, myth_closest = myth_similarities[word]
        
        # Assign to the plot with higher similarity
        if hist_sim >= myth_sim:
            # Assign to historical plot
            print(f"{word} is assigned to historical plot - closest to {hist_closest} (similarity: {hist_sim:.3f})")
            historical_assigned.append((word, hist_pos, hist_sim, hist_closest))
            connected_nodes[hist_closest] += 1
        else:
            # Assign to myth plot
            if myth_sim >= 0.7:  # Only assign to myth if reasonably similar
                print(f"{word} is assigned to myth plot - closest to {myth_closest} (similarity: {myth_sim:.3f})")
                myth_assigned.append((word, myth_pos, myth_sim, myth_closest))
                myth_connected_nodes[myth_closest] += 1
            else:
                # If similarity to myth is too low, still assign to historical
                print(f"{word} has low similarity to both, defaulting to historical plot - closest to {hist_closest} (similarity: {hist_sim:.3f})")
                historical_assigned.append((word, hist_pos, hist_sim, hist_closest))
                connected_nodes[hist_closest] += 1
    
    # Sort by similarity for better visualization
    historical_assigned.sort(key=lambda x: x[2], reverse=True)
    myth_assigned.sort(key=lambda x: x[2], reverse=True)
    
    # ---------------------------------------------------------------
    # NEW VISUALIZATION: ROMURBITAL (DISCOVERED) SITES
    # ---------------------------------------------------------------
    
    # Initialize dictionaries to store similarity data for romurbital sites
    romurbital_historical_similarities = {}  # {romurbital_name: (similarity, position, closest_name)}
    romurbital_myth_similarities = {}        # {romurbital_name: (similarity, position, closest_name)}
    
    # Calculate similarities for all romurbital places to other discovered sites and myth sites
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
                hist_similarity, hist_closest = calculate_discovery_similarity(emb, other_discovered_embs, other_discovered_names)
                hist_position = map_similarity_to_position(hist_similarity)
                romurbital_historical_similarities[word] = (hist_similarity, hist_position, hist_closest)
        except Exception as e:
            print(f"Error processing romurbital place {word} (historical): {str(e)}")
            traceback.print_exc()
            romurbital_historical_similarities[word] = (0, 0, "None")
        
        # Calculate similarity to myth sites
        if myth_embeddings:
            try:
                # Find closest myth place
                myth_sims = [(calculate_similarity(emb, myth_emb), name) for myth_emb, name in zip(myth_embeddings, myth_names)]
                if myth_sims:
                    myth_closest_sim, myth_closest_name = max(myth_sims, key=lambda x: x[0])
                    
                    # Use same tiered scale for visualization
                    myth_position = map_similarity_to_position(myth_closest_sim)
                    
                    romurbital_myth_similarities[word] = (myth_closest_sim, myth_position, myth_closest_name)
            except Exception as e:
                print(f"Error processing romurbital place {word} (myth): {str(e)}")
                traceback.print_exc()
                romurbital_myth_similarities[word] = (0, 0, "None")
        else:
            romurbital_myth_similarities[word] = (0, 0, "None")
    
    # Now decide where each romurbital site belongs based on highest similarity
    romurbital_historical_assigned = []  # [(word, position, similarity, closest)]
    romurbital_myth_assigned = []       # [(word, position, similarity, closest)]
    
    romurbital_connected_nodes = Counter()         # For historical connections
    romurbital_myth_connected_nodes = Counter()    # For myth connections
    
    for word in romurbital_historical_similarities:
        hist_sim, hist_pos, hist_closest = romurbital_historical_similarities[word]
        
        # Only process if we actually found similarities to other places
        if hist_closest != "None":
            # Check if we have myth similarity data for this site
            if word in romurbital_myth_similarities:
                myth_sim, myth_pos, myth_closest = romurbital_myth_similarities[word]
            else:
                myth_sim, myth_pos, myth_closest = 0, 0, "None"
            
            # Assign to the plot with higher similarity
            if hist_sim >= myth_sim:
                # Assign to historical plot
                print(f"ROMURBITAL: {word} is assigned to historical plot - closest to {hist_closest} (similarity: {hist_sim:.3f})")
                romurbital_historical_assigned.append((word, hist_pos, hist_sim, hist_closest))
                romurbital_connected_nodes[hist_closest] += 1
            else:
                # Assign to myth plot
                if myth_sim >= 0.6:  # Only assign to myth if reasonably similar
                    print(f"ROMURBITAL: {word} is assigned to myth plot - closest to {myth_closest} (similarity: {myth_sim:.3f})")
                    romurbital_myth_assigned.append((word, myth_pos, myth_sim, myth_closest))
                    romurbital_myth_connected_nodes[myth_closest] += 1
                else:
                    # If similarity to myth is too low, still assign to historical
                    print(f"ROMURBITAL: {word} has low similarity to myth, defaulting to historical plot - closest to {hist_closest} (similarity: {hist_sim:.3f})")
                    romurbital_historical_assigned.append((word, hist_pos, hist_sim, hist_closest))
                    romurbital_connected_nodes[hist_closest] += 1
    
    # Sort by similarity for better visualization
    romurbital_historical_assigned.sort(key=lambda x: x[2], reverse=True)
    romurbital_myth_assigned.sort(key=lambda x: x[2], reverse=True)
    
    # Create interactive visualization with Plotly
    try:
        # ---------------------------------------------------------------
        # FIRST SET OF SUBPLOTS - UNDISCOVERED SITES (ORIGINAL)
        # ---------------------------------------------------------------
        
        # Create figure with subplots for undiscovered sites
        fig1 = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                "Discovery Spectrum - Undiscovered Sites and Their Similarity to Discovered Sites",
                "Myth Connections - Undiscovered Sites and Their Similarity to Myth Sites"
            ),
            specs=[[{"type": "scatter"}, {"type": "scatter"}]],
            horizontal_spacing=0.05
        )
        
        # FIRST SUBPLOT - DISCOVERY VISUALIZATION
        
        # Prepare data for discovered places (only those that are connected)
        connected_discovered = [(word, emb) for word, emb in discovered_places if word in connected_nodes]
        
        if connected_discovered:
            # Sort by centrality
            connected_discovered.sort(key=lambda x: centrality_scores.get(x[0], 0))
            
            # For discovered nodes, centrality affects y-position AND node size
            disc_names = [word for word, _ in connected_discovered]
            disc_y = [centrality_scores.get(word, 0.5) for word in disc_names]  # Positioned by centrality
            disc_x = [100] * len(disc_names)  # Place at 100 (right side)
            
            # Size based on both connections and centrality
            disc_sizes = [80 * (1 + 0.3 * connected_nodes[word] + 0.2 * centrality_scores.get(word, 0.5)) 
                          for word in disc_names]
            
            disc_text = [f"{word}<br>Connections: {connected_nodes[word]}<br>Centrality: {centrality_scores.get(word, 0):.3f}" 
                        for word in disc_names]
            
            fig1.add_trace(go.Scatter(
                x=disc_x,
                y=disc_y,
                mode='markers+text',
                name='Discovered Sites',
                marker=dict(
                    color='red',
                    size=disc_sizes,
                    sizemode='area',
                    sizeref=2.*max(disc_sizes)/(40.**2) if disc_sizes else 1,
                    sizemin=6
                ),
                text=disc_names,
                textposition="middle right",
                textfont=dict(size=10),
                hoverinfo='text',
                hovertext=disc_text
            ), row=1, col=1)
        
        # Plot undiscovered places assigned to historical visualization
        if historical_assigned:
            hist_undisc_names = [word for word, _, _, _ in historical_assigned]
            hist_undisc_x = [pos for _, pos, _, _ in historical_assigned]
            hist_undisc_y = [centrality_scores.get(word, 0.5) for word, _, _, _ in historical_assigned]
            
            # Size based on centrality
            hist_undisc_sizes = [80 * (1 + 0.2 * centrality_scores.get(word, 0.5)) for word, _, _, _ in historical_assigned]
            hist_undisc_text = []
            for (word, pos, sim, closest) in historical_assigned:
                hover_text = (f"{word}<br>"
                            f"Similarity Score: {sim:.3f}<br>"
                            f"Closest to: {closest}<br>"
                            f"Centrality: {centrality_scores.get(word, 0):.3f}")
                hist_undisc_text.append(hover_text)
            
            # Use the same sizing factor as discovered sites for consistency
            hist_undisc_sizes = [100] * len(hist_undisc_names)
            
            fig1.add_trace(go.Scatter(
                x=hist_undisc_x,
                y=hist_undisc_y,
                mode='markers+text',
                name='Undiscovered Sites',
                marker=dict(
                    color='green',
                    size=hist_undisc_sizes,
                    sizemode='area',
                    sizeref=2.*max(hist_undisc_sizes + disc_sizes)/(40.**2) if (hist_undisc_sizes and disc_sizes) else 1,
                    sizemin=6
                ),
                text=hist_undisc_names,
                textposition="middle left",
                textfont=dict(size=10),
                hoverinfo='text',
                hovertext=hist_undisc_text
            ), row=1, col=1)
            
            # Create a dictionary mapping discovered site names to their y-positions for drawing lines
            disc_y_positions = {name: y for name, y in zip(disc_names, disc_y)}
            
            # Add connection lines from undiscovered to discovered sites
            for i, (word, pos, _, closest) in enumerate(historical_assigned):
                if closest in disc_y_positions:
                    fig1.add_trace(go.Scatter(
                        x=[pos, 100],
                        y=[hist_undisc_y[i], disc_y_positions[closest]],
                        mode='lines',
                        line=dict(color='gray', width=1, dash='dash'),
                        opacity=0.7,
                        showlegend=False,
                        hoverinfo='none'
                    ), row=1, col=1)
        
        # SECOND SUBPLOT - MYTH VISUALIZATION
        
        # Only create myth visualization if there's myth data
        if myth_assigned and myth_places:
            # Prepare data for myth places (only those that are connected)
            connected_myths = [(word, emb) for word, emb in myth_places if word in myth_connected_nodes]
            
            if connected_myths:
                # Sort by centrality
                connected_myths.sort(key=lambda x: centrality_scores.get(x[0], 0))
                
                # For myth nodes, centrality affects y-position AND node size
                myth_names = [word for word, _ in connected_myths]
                myth_y = [centrality_scores.get(word, 0.5) for word in myth_names]  # Positioned by centrality
                myth_x = [100] * len(myth_names)  # Place at 100 (right side)
                
                # Size based on both connections and centrality
                myth_sizes = [80 * (1 + 0.3 * myth_connected_nodes[word] + 0.2 * centrality_scores.get(word, 0.5)) 
                            for word in myth_names]
                
                myth_text = [f"{word}<br>Connections: {myth_connected_nodes[word]}<br>Centrality: {centrality_scores.get(word, 0):.3f}" 
                            for word in myth_names]
                
                fig1.add_trace(go.Scatter(
                    x=myth_x,
                    y=myth_y,
                    mode='markers+text',
                    name='Myth Sites',
                    marker=dict(
                        color='blue',  # Blue for myth sites
                        size=myth_sizes,
                        sizemode='area',
                        sizeref=2.*max(myth_sizes)/(40.**2) if myth_sizes else 1,
                        sizemin=6
                    ),
                    text=myth_names,
                    textposition="middle right",
                    textfont=dict(size=10),
                    hoverinfo='text',
                    hovertext=myth_text
                ), row=1, col=2)
                
                # Plot undiscovered places assigned to myth visualization
                myth_undisc_names = [word for word, _, _, _ in myth_assigned]
                myth_undisc_x = [pos for _, pos, _, _ in myth_assigned]
                myth_undisc_y = [centrality_scores.get(word, 0.5) for word, _, _, _ in myth_assigned]
                
                # Prepare hover text
                myth_undisc_text = []
                for (word, pos, sim, closest) in myth_assigned:
                    hover_text = (f"{word}<br>"
                                f"Similarity Score: {sim:.3f}<br>"
                                f"Closest to: {closest}<br>"
                                f"Centrality: {centrality_scores.get(word, 0):.3f}")
                    myth_undisc_text.append(hover_text)
                
                # Use the same sizing factor for consistency
                myth_undisc_sizes = [100] * len(myth_undisc_names)
                
                fig1.add_trace(go.Scatter(
                    x=myth_undisc_x,
                    y=myth_undisc_y,
                    mode='markers+text',
                    name='Undiscovered Sites (Myth)',
                    marker=dict(
                        color='green',
                        size=myth_undisc_sizes,
                        sizemode='area',
                        sizeref=2.*max(myth_undisc_sizes + myth_sizes)/(40.**2) if (myth_undisc_sizes and myth_sizes) else 1,
                        sizemin=6
                    ),
                    text=myth_undisc_names,
                    textposition="middle left",
                    textfont=dict(size=10),
                    hoverinfo='text',
                    hovertext=myth_undisc_text,
                    showlegend=False  # Don't show in legend since it duplicates the first subplot
                ), row=1, col=2)
                
                # Create a dictionary mapping myth site names to their y-positions for drawing lines
                myth_y_positions = {name: y for name, y in zip(myth_names, myth_y)}
                
                # Add connection lines from undiscovered to myth sites
                for i, (word, pos, _, closest) in enumerate(myth_assigned):
                    if closest in myth_y_positions:
                        fig1.add_trace(go.Scatter(
                            x=[pos, 100],
                            y=[myth_undisc_y[i], myth_y_positions[closest]],
                            mode='lines',
                            line=dict(color='gray', width=1, dash='dash'),
                            opacity=0.7,
                            showlegend=False,
                            hoverinfo='none'
                        ), row=1, col=2)
        
        # Determine the x-axis range based on data
        all_positions = [pos for _, pos, _, _ in historical_assigned + myth_assigned]
        if all_positions:
            min_pos = min(all_positions)
            min_x = max(0, min_pos - 10)  # Start 10 units before the lowest similarity score, but not below 0
        else:
            min_x = 0
        
        # Update layout for both visualizations
        fig1.update_layout(
            title=f"{file_stem} - Undiscovered Sites: Discovery and Myth Spectrums",
            legend=dict(
                orientation="h",
                y=1.02,
                x=0.5,
                xanchor="center",
                yanchor="bottom"
            ),
            width=2000,  # Double width for side-by-side plots
            height=max(600, 100 + 60 * max(
                len(connected_discovered) if 'connected_discovered' in locals() else 0,
                len(connected_myths) if 'connected_myths' in locals() else 0
            )),
            hovermode='closest',
        )
        
        # Update axes for both subplots
        fig1.update_xaxes(
            title="Similarity to Closest Discovered Site<br>(Higher values = More Similar)",
            range=[min_x, 105],
            row=1, col=1
        )
        
        fig1.update_xaxes(
            title="Similarity to Closest Myth Site<br>(Higher values = More Similar)",
            range=[min_x, 105],
            row=1, col=2
        )
        
        fig1.update_yaxes(title="Network Centrality", row=1, col=1)
        fig1.update_yaxes(title="Network Centrality", row=1, col=2)
        
        # Add data summary annotations for first subplot
        fig1.add_annotation(
            text=(f"<b>Discovery Data Summary:</b><br>"
                  f"Total places analyzed: {len(discovered_places) + len(undiscovered_places)}<br>"
                  f"Discovered sites: {len(discovered_places)}<br>"
                  f"Undiscovered sites: {len(undiscovered_places)}<br>"
                  f"Undiscovered sites assigned to historical: {len(historical_assigned)}<br>"
                  f"Connected discovered sites: {len(connected_discovered) if 'connected_discovered' in locals() else 0}"),
            align="left",
            showarrow=False,
            xref="x",
            yref="paper",
            x=0,
            y=1.0,
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            bgcolor="white",
            opacity=0.8,
            row=1, col=1
        )
        
        # Add data summary annotations for second subplot
        if 'connected_myths' in locals() and connected_myths:
            fig1.add_annotation(
                text=(f"<b>Myth Data Summary:</b><br>"
                      f"Total myth sites: {len(myth_places)}<br>"
                      f"Connected myth sites: {len(connected_myths)}<br>"
                      f"Undiscovered sites assigned to myth: {len(myth_assigned)}"),
                align="left",
                showarrow=False,
                xref="x2",
                yref="paper",
                x=0,
                y=1.0,
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                bgcolor="white",
                opacity=0.8,
                row=1, col=2
            )
        
        # Save as HTML (interactive)
        output_path = os.path.join(output_dir, f"{file_stem}_discovery_myth_spectrum.html")
        fig1.write_html(output_path, full_html=True, include_plotlyjs='cdn')
        print(f"Saved interactive visualization to: {output_path}")
        
        # Try to save as static image for reference
        try:
            image_path = os.path.join(output_dir, f"{file_stem}_discovery_myth_spectrum.png")
            fig1.write_image(image_path)
            print(f"Saved static image to: {image_path}")
        except Exception as img_error:
            print(f"Note: Could not save static image due to: {str(img_error)}")
            print("This is normal if you don't have kaleido or another Plotly image export backend installed.")
            print("The interactive HTML version should still work fine.")
        
        # ---------------------------------------------------------------
        # SECOND SET OF SUBPLOTS - ROMURBITAL SITES (NEW)
        # ---------------------------------------------------------------
        
        # Create figure with subplots for romurbital sites
        fig2 = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                "Romurbital Sites - Similarity to Other Discovered Sites",
                "Romurbital Sites - Similarity to Myth Sites"
            ),
            specs=[[{"type": "scatter"}, {"type": "scatter"}]],
            horizontal_spacing=0.05
        )
        
        # FIRST SUBPLOT - ROMURBITAL VS HISTORICAL
        
        # Prepare data for discovered places (only those that are connected)
        romurbital_connected_discovered = [(word, emb) for word, emb in discovered_places 
                                         if word in romurbital_connected_nodes and word not in [w for w, _, _, _ in romurbital_historical_assigned]]
        
        if romurbital_connected_discovered:
            # Sort by centrality
            romurbital_connected_discovered.sort(key=lambda x: centrality_scores.get(x[0], 0))
            
            # For discovered nodes, centrality affects y-position AND node size
            rom_disc_names = [word for word, _ in romurbital_connected_discovered]
            rom_disc_y = [centrality_scores.get(word, 0.5) for word in rom_disc_names]  # Positioned by centrality
            rom_disc_x = [100] * len(rom_disc_names)  # Place at 100 (right side)
            
            # Size based on both connections and centrality
            rom_disc_sizes = [80 * (1 + 0.3 * romurbital_connected_nodes[word] + 0.2 * centrality_scores.get(word, 0.5)) 
                          for word in rom_disc_names]
            
            rom_disc_text = [f"{word}<br>Connections: {romurbital_connected_nodes[word]}<br>Centrality: {centrality_scores.get(word, 0):.3f}" 
                        for word in rom_disc_names]
            
            fig2.add_trace(go.Scatter(
                x=rom_disc_x,
                y=rom_disc_y,
                mode='markers+text',
                name='Target Romurbital Sites',
                marker=dict(
                    color='red',
                    size=rom_disc_sizes,
                    sizemode='area',
                    sizeref=2.*max(rom_disc_sizes)/(40.**2) if rom_disc_sizes else 1,
                    sizemin=6
                ),
                text=rom_disc_names,
                textposition="middle right",
                textfont=dict(size=10),
                hoverinfo='text',
                hovertext=rom_disc_text
            ), row=1, col=1)
        
        # Plot romurbital places assigned to historical visualization
        if romurbital_historical_assigned:
            rom_hist_names = [word for word, _, _, _ in romurbital_historical_assigned]
            rom_hist_x = [pos for _, pos, _, _ in romurbital_historical_assigned]
            rom_hist_y = [centrality_scores.get(word, 0.5) for word, _, _, _ in romurbital_historical_assigned]
            
            # Size based on centrality
            rom_hist_sizes = [80 * (1 + 0.2 * centrality_scores.get(word, 0.5)) for word, _, _, _ in romurbital_historical_assigned]
            rom_hist_text = []
            for (word, pos, sim, closest) in romurbital_historical_assigned:
                hover_text = (f"{word}<br>"
                            f"Similarity Score: {sim:.3f}<br>"
                            f"Closest to: {closest}<br>"
                            f"Centrality: {centrality_scores.get(word, 0):.3f}")
                rom_hist_text.append(hover_text)
            
            # Use the same sizing factor as discovered sites for consistency
            rom_hist_sizes = [100] * len(rom_hist_names)
            
            fig2.add_trace(go.Scatter(
                x=rom_hist_x,
                y=rom_hist_y,
                mode='markers+text',
                name='Source Romurbital Sites',
                marker=dict(
                    color='purple',  # Purple to distinguish from undiscovered sites
                    size=rom_hist_sizes,
                    sizemode='area',
                    sizeref=2.*max(rom_hist_sizes + rom_disc_sizes)/(40.**2) if (rom_hist_sizes and rom_disc_sizes) else 1,
                    sizemin=6
                ),
                text=rom_hist_names,
                textposition="middle left",
                textfont=dict(size=10),
                hoverinfo='text',
                hovertext=rom_hist_text
            ), row=1, col=1)
            
            # Create a dictionary mapping romurbital target site names to their y-positions for drawing lines
            rom_disc_y_positions = {name: y for name, y in zip(rom_disc_names, rom_disc_y)}
            
            # Add connection lines from romurbital source to romurbital target sites
            for i, (word, pos, _, closest) in enumerate(romurbital_historical_assigned):
                if closest in rom_disc_y_positions:
                    fig2.add_trace(go.Scatter(
                        x=[pos, 100],
                        y=[rom_hist_y[i], rom_disc_y_positions[closest]],
                        mode='lines',
                        line=dict(color='gray', width=1, dash='dash'),
                        opacity=0.7,
                        showlegend=False,
                        hoverinfo='none'
                    ), row=1, col=1)
        
        # SECOND SUBPLOT - ROMURBITAL VS MYTH VISUALIZATION
        
        # Only create myth visualization if there's myth data
        if romurbital_myth_assigned and myth_places:
            # Prepare data for myth places (only those that are connected to romurbital sites)
            rom_connected_myths = [(word, emb) for word, emb in myth_places if word in romurbital_myth_connected_nodes]
            
            if rom_connected_myths:
                # Sort by centrality
                rom_connected_myths.sort(key=lambda x: centrality_scores.get(x[0], 0))
                
                # For myth nodes, centrality affects y-position AND node size
                rom_myth_names = [word for word, _ in rom_connected_myths]
                rom_myth_y = [centrality_scores.get(word, 0.5) for word in rom_myth_names]  # Positioned by centrality
                rom_myth_x = [100] * len(rom_myth_names)  # Place at 100 (right side)
                
                # Size based on both connections and centrality
                rom_myth_sizes = [80 * (1 + 0.3 * romurbital_myth_connected_nodes[word] + 0.2 * centrality_scores.get(word, 0.5)) 
                            for word in rom_myth_names]
                
                rom_myth_text = [f"{word}<br>Connections: {romurbital_myth_connected_nodes[word]}<br>Centrality: {centrality_scores.get(word, 0):.3f}" 
                            for word in rom_myth_names]
                
                fig2.add_trace(go.Scatter(
                    x=rom_myth_x,
                    y=rom_myth_y,
                    mode='markers+text',
                    name='Myth Sites',
                    marker=dict(
                        color='blue',  # Blue for myth sites
                        size=rom_myth_sizes,
                        sizemode='area',
                        sizeref=2.*max(rom_myth_sizes)/(40.**2) if rom_myth_sizes else 1,
                        sizemin=6
                    ),
                    text=rom_myth_names,
                    textposition="middle right",
                    textfont=dict(size=10),
                    hoverinfo='text',
                    hovertext=rom_myth_text
                ), row=1, col=2)
                
                # Plot romurbital places assigned to myth visualization
                rom_myth_assigned_names = [word for word, _, _, _ in romurbital_myth_assigned]
                rom_myth_assigned_x = [pos for _, pos, _, _ in romurbital_myth_assigned]
                rom_myth_assigned_y = [centrality_scores.get(word, 0.5) for word, _, _, _ in romurbital_myth_assigned]
                
                # Prepare hover text
                rom_myth_assigned_text = []
                for (word, pos, sim, closest) in romurbital_myth_assigned:
                    hover_text = (f"{word}<br>"
                                f"Similarity Score: {sim:.3f}<br>"
                                f"Closest to: {closest}<br>"
                                f"Centrality: {centrality_scores.get(word, 0):.3f}")
                    rom_myth_assigned_text.append(hover_text)
                
                # Use the same sizing factor for consistency
                rom_myth_assigned_sizes = [100] * len(rom_myth_assigned_names)
                
                fig2.add_trace(go.Scatter(
                    x=rom_myth_assigned_x,
                    y=rom_myth_assigned_y,
                    mode='markers+text',
                    name='Romurbital Sites (Myth)',
                    marker=dict(
                        color='purple',  # Purple to distinguish from undiscovered sites
                        size=rom_myth_assigned_sizes,
                        sizemode='area',
                        sizeref=2.*max(rom_myth_assigned_sizes + rom_myth_sizes)/(40.**2) if (rom_myth_assigned_sizes and rom_myth_sizes) else 1,
                        sizemin=6
                    ),
                    text=rom_myth_assigned_names,
                    textposition="middle left",
                    textfont=dict(size=10),
                    hoverinfo='text',
                    hovertext=rom_myth_assigned_text,
                    showlegend=False  # Don't show in legend since it duplicates the first subplot
                ), row=1, col=2)
                
                # Create a dictionary mapping myth site names to their y-positions for drawing lines
                rom_myth_y_positions = {name: y for name, y in zip(rom_myth_names, rom_myth_y)}
                
                # Add connection lines from romurbital to myth sites
                for i, (word, pos, _, closest) in enumerate(romurbital_myth_assigned):
                    if closest in rom_myth_y_positions:
                        fig2.add_trace(go.Scatter(
                            x=[pos, 100],
                            y=[rom_myth_assigned_y[i], rom_myth_y_positions[closest]],
                            mode='lines',
                            line=dict(color='gray', width=1, dash='dash'),
                            opacity=0.7,
                            showlegend=False,
                            hoverinfo='none'
                        ), row=1, col=2)
        
        # Determine the x-axis range based on data
        rom_all_positions = [pos for _, pos, _, _ in romurbital_historical_assigned + romurbital_myth_assigned]
        if rom_all_positions:
            rom_min_pos = min(rom_all_positions)
            rom_min_x = max(0, rom_min_pos - 10)  # Start 10 units before the lowest similarity score, but not below 0
        else:
            rom_min_x = 0
        
        # Update layout for both visualizations
        fig2.update_layout(
            title=f"{file_stem} - Romurbital Sites: Historical and Myth Comparisons",
            legend=dict(
                orientation="h",
                y=1.02,
                x=0.5,
                xanchor="center",
                yanchor="bottom"
            ),
            width=2000,  # Double width for side-by-side plots
            height=max(600, 100 + 60 * max(
                len(romurbital_connected_discovered) if 'romurbital_connected_discovered' in locals() else 0,
                len(rom_connected_myths) if 'rom_connected_myths' in locals() else 0
            )),
            hovermode='closest',
        )
        
        # Update axes for both subplots
        fig2.update_xaxes(
            title="Similarity to Closest Other Romurbital Site<br>(Higher values = More Similar)",
            range=[rom_min_x, 105],
            row=1, col=1
        )
        
        fig2.update_xaxes(
            title="Similarity to Closest Myth Site<br>(Higher values = More Similar)",
            range=[rom_min_x, 105],
            row=1, col=2
        )
        
        fig2.update_yaxes(title="Network Centrality", row=1, col=1)
        fig2.update_yaxes(title="Network Centrality", row=1, col=2)
        
        # Add data summary annotations for romurbital subplots
        fig2.add_annotation(
            text=(f"<b>Romurbital Data Summary:</b><br>"
                  f"Total Romurbital sites: {len(discovered_places)}<br>"
                  f"Romurbital sites analyzed: {len(romurbital_historical_assigned) + len(romurbital_connected_discovered)}<br>"
                  f"Romurbital sites as sources: {len(romurbital_historical_assigned)}<br>"
                  f"Romurbital sites as targets: {len(romurbital_connected_discovered) if 'romurbital_connected_discovered' in locals() else 0}"),
            align="left",
            showarrow=False,
            xref="x",
            yref="paper",
            x=0,
            y=1.0,
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            bgcolor="white",
            opacity=0.8,
            row=1, col=1
        )
        
        # Add data summary annotations for romurbital-myth subplot
        if 'rom_connected_myths' in locals() and rom_connected_myths:
            fig2.add_annotation(
                text=(f"<b>Romurbital-Myth Data Summary:</b><br>"
                      f"Total myth sites: {len(myth_places)}<br>"
                      f"Connected myth sites: {len(rom_connected_myths)}<br>"
                      f"Romurbital sites assigned to myth: {len(romurbital_myth_assigned)}"),
                align="left",
                showarrow=False,
                xref="x2",
                yref="paper",
                x=0,
                y=1.0,
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                bgcolor="white",
                opacity=0.8,
                row=1, col=2
            )
        
        # Save as HTML (interactive)
        rom_output_path = os.path.join(output_dir, f"{file_stem}_romurbital_comparison_spectrum.html")
        fig2.write_html(rom_output_path, full_html=True, include_plotlyjs='cdn')
        print(f"Saved Romurbital interactive visualization to: {rom_output_path}")
        
        # Try to save as static image for reference
        try:
            rom_image_path = os.path.join(output_dir, f"{file_stem}_romurbital_comparison_spectrum.png")
            fig2.write_image(rom_image_path)
            print(f"Saved Romurbital static image to: {rom_image_path}")
        except Exception as rom_img_error:
            print(f"Note: Could not save Romurbital static image due to: {str(rom_img_error)}")
            print("This is normal if you don't have kaleido or another Plotly image export backend installed.")
            print("The interactive HTML version should still work fine.")
        
        # Create a text report that includes both original and romurbital analyses
        report_path = os.path.join(output_dir, f"{file_stem}_complete_analysis_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"COMPLETE ANALYSIS REPORT FOR {file_stem.upper()}\n")
            f.write("=" * 60 + "\n\n")
            
            # PART 1: ORIGINAL ANALYSIS (UNDISCOVERED SITES)
            f.write("PART 1: UNDISCOVERED SITES ANALYSIS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("DISCOVERY SPECTRUM:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total places analyzed: {len(discovered_places) + len(undiscovered_places)}\n")
            f.write(f"Discovered sites: {len(discovered_places)}\n")
            f.write(f"Undiscovered sites: {len(undiscovered_places)}\n")
            f.write(f"Undiscovered sites assigned to historical: {len(historical_assigned)}\n")
            f.write(f"Connected discovered sites: {len(connected_discovered) if 'connected_discovered' in locals() else 0}\n\n")
            
            f.write("UNDISCOVERED SITE SIMILARITIES TO DISCOVERED SITES:\n")
            f.write("-" * 50 + "\n")
            
            # Sort by similarity for the report
            sorted_data = sorted(historical_assigned, key=lambda x: x[2], reverse=True)
            for word, pos, similarity, closest in sorted_data:
                f.write(f"{word}: Closest to {closest} (similarity: {similarity:.3f}, "
                      f"centrality: {centrality_scores.get(word, 0):.3f})\n")
            
            if myth_assigned and myth_places:
                f.write("\n\nMYTH SPECTRUM:\n")
                f.write("-" * 50 + "\n")
                f.write(f"Total myth sites: {len(myth_places)}\n")
                f.write(f"Connected myth sites: {len(connected_myths) if 'connected_myths' in locals() else 0}\n")
                f.write(f"Undiscovered sites assigned to myth: {len(myth_assigned)}\n\n")
                
                f.write("UNDISCOVERED SITE SIMILARITIES TO MYTH SITES:\n")
                f.write("-" * 50 + "\n")
                
                # Sort by similarity for the report
                sorted_myth_data = sorted(myth_assigned, key=lambda x: x[2], reverse=True)
                for word, pos, similarity, closest in sorted_myth_data:
                    f.write(f"{word}: Closest to {closest} (similarity: {similarity:.3f}, "
                          f"centrality: {centrality_scores.get(word, 0):.3f})\n")
            
            # PART 2: ROMURBITAL ANALYSIS
            f.write("\n\n\nPART 2: ROMURBITAL SITES ANALYSIS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("ROMURBITAL-TO-ROMURBITAL SPECTRUM:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total Romurbital sites: {len(discovered_places)}\n")
            f.write(f"Romurbital sites analyzed: {len(romurbital_historical_assigned) + len(romurbital_connected_discovered) if 'romurbital_connected_discovered' in locals() else len(romurbital_historical_assigned)}\n")
            f.write(f"Romurbital source sites: {len(romurbital_historical_assigned)}\n")
            f.write(f"Romurbital target sites: {len(romurbital_connected_discovered) if 'romurbital_connected_discovered' in locals() else 0}\n\n")
            
            f.write("ROMURBITAL SITE SIMILARITIES TO OTHER ROMURBITAL SITES:\n")
            f.write("-" * 50 + "\n")
            
            # Sort by similarity for the report
            sorted_rom_data = sorted(romurbital_historical_assigned, key=lambda x: x[2], reverse=True)
            for word, pos, similarity, closest in sorted_rom_data:
                f.write(f"{word}: Closest to {closest} (similarity: {similarity:.3f}, "
                      f"centrality: {centrality_scores.get(word, 0):.3f})\n")
            
            if romurbital_myth_assigned and myth_places:
                f.write("\n\nROMURBITAL-TO-MYTH SPECTRUM:\n")
                f.write("-" * 50 + "\n")
                f.write(f"Total myth sites: {len(myth_places)}\n")
                f.write(f"Connected myth sites: {len(rom_connected_myths) if 'rom_connected_myths' in locals() else 0}\n")
                f.write(f"Romurbital sites assigned to myth: {len(romurbital_myth_assigned)}\n\n")
                
                f.write("ROMURBITAL SITE SIMILARITIES TO MYTH SITES:\n")
                f.write("-" * 50 + "\n")
                
                # Sort by similarity for the report
                sorted_rom_myth_data = sorted(romurbital_myth_assigned, key=lambda x: x[2], reverse=True)
                for word, pos, similarity, closest in sorted_rom_myth_data:
                    f.write(f"{word}: Closest to {closest} (similarity: {similarity:.3f}, "
                          f"centrality: {centrality_scores.get(word, 0):.3f})\n")
            
            # Add comparative summary between the two analyses
            f.write("\n\nCOMPARATIVE SUMMARY:\n")
            f.write("-" * 50 + "\n")
            
            # Create a set of all the places mentioned in undiscovered analysis
            undiscovered_mentioned = set([closest for _, _, _, closest in historical_assigned])
            if myth_assigned:
                undiscovered_mentioned.update([closest for _, _, _, closest in myth_assigned])
            
            # Create a set of all the places mentioned in romurbital analysis
            romurbital_mentioned = set([closest for _, _, _, closest in romurbital_historical_assigned])
            if romurbital_myth_assigned:
                romurbital_mentioned.update([closest for _, _, _, closest in romurbital_myth_assigned])
            
            # Find places that appear in both analyses
            common_places = undiscovered_mentioned.intersection(romurbital_mentioned)
            
            f.write(f"Places common to both undiscovered and romurbital analyses: {len(common_places)}\n")
            for place in sorted(common_places):
                f.write(f"  - {place}\n")
            
            f.write(f"\nPlaces unique to undiscovered analysis: {len(undiscovered_mentioned - common_places)}\n")
            for place in sorted(undiscovered_mentioned - common_places):
                f.write(f"  - {place}\n")
            
            f.write(f"\nPlaces unique to romurbital analysis: {len(romurbital_mentioned - common_places)}\n")
            for place in sorted(romurbital_mentioned - common_places):
                f.write(f"  - {place}\n")
        
        print(f"Saved complete analysis report to: {report_path}")
        
        return len(discovered_places), len(undiscovered_places), len(myth_places)
        
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        traceback.print_exc()
        return None

def process_embeddings_folder(embeddings_dir):
    """Process all embedding files in a folder"""
    # Create output directory
    output_dir = os.path.join(embeddings_dir, "discovery_myth_visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all embedding files
    embedding_files = []
    for file in os.listdir(embeddings_dir):
        if file.endswith("_embeddings.npz") or file.endswith("_single_sent_embeddings.npz") or file.endswith("_three_sent_embeddings.npz"):
            embedding_file = os.path.join(embeddings_dir, file)
            file_stem = file.replace("_embeddings.npz", "").replace("_single_sent_embeddings.npz", "").replace("_three_sent_embeddings.npz", "")
            
            # Look for corresponding metadata file
            metadata_variants = [
                f"{file_stem}_metadata.json",
                f"{file_stem}_single_sent_metadata.json",
                f"{file_stem}_three_sent_metadata.json"
            ]
            
            metadata_file = None
            for variant in metadata_variants:
                potential_file = os.path.join(embeddings_dir, variant)
                if os.path.exists(potential_file):
                    metadata_file = potential_file
                    break
            
            if metadata_file:
                embedding_files.append((embedding_file, metadata_file))
            else:
                print(f"Warning: No metadata file found for {file}, skipping")
    
    print(f"Found {len(embedding_files)} embedding files to process")
    
    # Process each file
    results = []
    for embedding_file, metadata_file in embedding_files:
        try:
            file_stats = create_discovery_visualization(embedding_file, metadata_file, output_dir)
            if file_stats:
                file_stem = Path(embedding_file).stem.replace('_embeddings', '').replace('_single_sent_embeddings', '').replace('_three_sent_embeddings', '')
                results.append((file_stem, *file_stats))
        except Exception as e:
            print(f"Error processing {embedding_file}: {str(e)}")
            traceback.print_exc()
    
    # Create a summary report
    if results:
        summary_path = os.path.join(output_dir, "discovery_myth_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("DISCOVERY AND MYTH VISUALIZATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total files processed: {len(results)}\n\n")
            
            f.write(f"{'File':<30} {'Discovered Sites':<15} {'Undiscovered Sites':<15} {'Myth Sites':<15}\n")
            f.write("-" * 80 + "\n")
            
            for result in sorted(results):
                file_stem = result[0]
                discovered = result[1] if len(result) > 1 else 0
                undiscovered = result[2] if len(result) > 2 else 0
                myths = result[3] if len(result) > 3 else 0
                
                f.write(f"{file_stem:<30} {discovered:<15} {undiscovered:<15} {myths:<15}\n")
        
        print(f"\nSummary saved to: {summary_path}")
        print(f"All discovery and myth visualizations saved to: {output_dir}")

# Entry point
if __name__ == "__main__":
    # Set the path to your embeddings directory
    embeddings_dir = r'C:\Users\User\Downloads\embeddings_single_sentence_output\embeddings_single_sentence_output'
    
    # Process main directory
    process_embeddings_folder(embeddings_dir)