import json
import csv
from collections import defaultdict

# Load JSON
with open('RA-wiki.json', 'r') as f:
    data = json.load(f)

# Group by dialog_id
dialog_groups = defaultdict(list)
for item in data:
    dialog_groups[item['dialog_id']].append(item)

# Flatten with utterance_idx and write CSV
with open('output.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['dialog_id', 'turn_id', 'utterance_idx', 'speaker', 'utterance', 'personal_attack', 'gpt-4o_speaker_intention'])

    for dialog_id in dialog_groups:
        dialog = sorted(dialog_groups[dialog_id], key=lambda x: x['turn_id'])
        for idx, item in enumerate(dialog):
            writer.writerow([
                item['dialog_id'],
                item['turn_id'],
                idx,
                item['speaker'],
                item['utterance'],
                item['personal_attack'],
                item['gpt-4o_speaker_intention']
            ])
