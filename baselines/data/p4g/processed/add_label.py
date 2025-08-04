import json

def create_new_labels(conversations_file, analysis_file, output_file):
    """
    Merge conversation data with persuasion analysis results to create new labels.
    
    Args:
        conversations_file: Path to JSON file with conversation data
        analysis_file: Path to JSON file with persuasion analysis results
        output_file: Path to save the merged data
    """
    
    # Load the conversation data
    with open(conversations_file, 'r') as f:
        conversations = json.load(f)
    
    # Load the analysis results
    with open(analysis_file, 'r') as f:
        analysis_data = json.load(f)
    
    # Create a mapping from dialogue_id to analysis results
    analysis_map = {}
    for result in analysis_data['detailed_results']:
        dialogue_id = result['dialogue_id']
        analysis_map[dialogue_id] = result
    
    # Process each conversation
    for conversation in conversations:
        dialogue_id = conversation['dialogue_id']
        
        # Initialize new_label as unclear (2) by default
        new_label = 2
        
        # Check if we have analysis data for this conversation
        if dialogue_id in analysis_map:
            analysis = analysis_map[dialogue_id]
            stated_intention = analysis.get('stated_intention', '').lower()
            
            # Determine label based on stated intention
            if stated_intention == 'donate':
                new_label = 1  # Intended to donate
            elif stated_intention == 'not_donate':
                new_label = 0  # Did not intend to donate
            elif stated_intention == 'unclear' or stated_intention == '':
                new_label = 2  # Unclear decision
            else:
                # Handle other possible values - you may need to adjust this
                # based on the actual values in your analysis data
                new_label = 2
        
        # Add the new label to the conversation
        conversation['new_label'] = new_label
    
    # Save the updated conversations
    with open(output_file, 'w') as f:
        json.dump(conversations, f, indent=2)
    
    return conversations

def print_label_distribution(conversations):
    """Print the distribution of new labels"""
    label_counts = {0: 0, 1: 0, 2: 0}
    
    for conv in conversations:
        label = conv.get('new_label', 2)
        label_counts[label] += 1
    
    total = len(conversations)
    print(f"Label Distribution:")
    print(f"Did not intend to donate (0): {label_counts[0]} ({label_counts[0]/total*100:.1f}%)")
    print(f"Intended to donate (1): {label_counts[1]} ({label_counts[1]/total*100:.1f}%)")
    print(f"Unclear decision (2): {label_counts[2]} ({label_counts[2]/total*100:.1f}%)")
    print(f"Total conversations: {total}")

# Example usage:
if __name__ == "__main__":
    # Replace these with your actual file paths
    conversations_file = "baselines/data/p4g/processed/RAT_1_1_test.json"  # Your first JSON file
    analysis_file = "analysis/persuasion_analysis_results.json"    # Your second JSON file
    output_file = conversations_file
    
    # Create the new labels
    updated_conversations = create_new_labels(conversations_file, analysis_file, output_file)
    
    # Print distribution
    print_label_distribution(updated_conversations)
    
    print(f"\nUpdated conversations saved to: {output_file}")