import re
import csv
import collections
from collections import defaultdict

class BrainLogParser:
    def __init__(self, weight=0.5, branch_threshold=3):
        """
        :param weight: Threshold multiplier to select longest common pattern.
        :param branch_threshold: Number of different words in a column to consider it a variable.
        """
        self.weight = weight
        self.branch_threshold = branch_threshold
        # Regex for common variables (IPs, numbers, etc.)
        self.regex_filters = [
            (r'blk_-?\d+', '<*>'),                             # HDFS block IDs
            (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<*>'), # IP Address
            (r'\b[0-9]+\b', '<*>'),                             # Numbers
            (r'0x[a-fA-F0-9]+', '<*>')                          # Hex
        ]

    def preprocess(self, log_lines):
        """Step 1: Preprocessing - Split words and filter common variables."""
        processed_logs = []
        for line_id, line in enumerate(log_lines):
            clean_line = line.strip()
            for regex, replacement in self.regex_filters:
                clean_line = re.sub(regex, replacement, clean_line)

            # Collapse huge delete lists into one placeholder to avoid overfitted templates.
            if 'BLOCK* ask' in clean_line and 'to delete' in clean_line:
                clean_line = re.sub(r'(to delete)\s+.+$', r'\1 <*>', clean_line)
            
            # Split by spaces (can be extended to other delimiters like '=' or ':')
            tokens = clean_line.split()
            processed_logs.append({'id': line_id, 'raw': line.strip(), 'tokens': tokens})
        return processed_logs

    def _calculate_frequencies(self, logs):
        """Calculates global frequency of each word."""
        freq_dict = defaultdict(int)
        for log in logs:
            for token in log['tokens']:
                if token != '<*>':
                    freq_dict[token] += 1
        return freq_dict

    def build_initial_groups(self, logs):
        """Step 2: Initial Group Creation based on Longest Word Combination."""
        freq_dict = self._calculate_frequencies(logs)
        
        # Group logs by length first
        length_groups = defaultdict(list)
        for log in logs:
            length_groups[len(log['tokens'])].append(log)
            
        initial_groups = defaultdict(list)
        
        for length, group_logs in length_groups.items():
            for log in group_logs:
                # Find max frequency in this log
                max_freq = 0
                for token in log['tokens']:
                    if token != '<*>':
                        max_freq = max(max_freq, freq_dict[token])
                
                threshold = max_freq * self.weight
                
                # Form the longest word combination (tokens with freq >= threshold)
                combination = []
                for idx, token in enumerate(log['tokens']):
                    if token != '<*>' and freq_dict.get(token, 0) >= threshold:
                        combination.append((idx, token))

                # Keep stable anchors so templates don't collapse into overly generic patterns.
                for anchor_idx in (3, 4, 5):
                    if anchor_idx < len(log['tokens']):
                        anchor = log['tokens'][anchor_idx]
                        if anchor != '<*>' and (anchor_idx, anchor) not in combination:
                            combination.append((anchor_idx, anchor))

                combination.sort(key=lambda x: x[0])
                
                combination_key = (length, tuple(combination))
                initial_groups[combination_key].append(log)
                
        return initial_groups

    def generate_templates(self, initial_groups):
        """Step 3-5: Node Updates and Template Generation."""
        templates = []
        seen_templates = set()
        
        for group_key, logs in initial_groups.items():
            if not logs:
                continue
            length, base_pattern = group_key
            template_tokens = ['<*>'] * length
            
            # Reconstruct the constant pattern base
            for idx, token in base_pattern:
                if idx >= length:
                    continue
                template_tokens[idx] = token
                
            # Check other columns for variance (Child/Parent tree logic approximation)
            for col_idx in range(length):
                if template_tokens[col_idx] != '<*>':
                    continue # Already a known constant from the initial pattern
                    
                # Collect all unique words in this column for this group
                unique_words = set()
                for log in logs:
                    if col_idx < len(log['tokens']):
                        unique_words.add(log['tokens'][col_idx])
                    
                # If the number of different words is below threshold, it's a constant
                if len(unique_words) <= self.branch_threshold and len(unique_words) > 0:
                    # In a true parallel tree, this splits the group into sub-templates.
                    # For simplicity, if it's identical across all logs in this specific subgroup,
                    # we promote it to a constant.
                    if len(unique_words) == 1:
                        template_tokens[col_idx] = list(unique_words)[0]
                else:
                    # Exceeds threshold -> remains a variable <*>
                    pass
            
            final_template = " ".join(template_tokens)

            if final_template not in seen_templates:
                seen_templates.add(final_template)
                templates.append(final_template)

        return templates

    def parse_file(self, input_filepath, output_filepath):
        """Main pipeline to read logs, parse them, and output to CSV."""
        with open(input_filepath, 'r', encoding='utf-8') as f:
            log_lines = f.readlines()
            
        print(f"Read {len(log_lines)} lines. Preprocessing...")
        processed_logs = self.preprocess(log_lines)
        
        print("Creating initial groups...")
        initial_groups = self.build_initial_groups(processed_logs)
        
        print("Generating templates...")
        parsed_templates = self.generate_templates(initial_groups)
        
        print(f"Writing results to {output_filepath}...")
        keys = ['EventId', 'EventTemplate']
        with open(output_filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(
                {'EventId': f'E{idx}', 'EventTemplate': template}
                for idx, template in enumerate(parsed_templates, start=1)
            )
            
        print("Done!")

# === Execution Example ===
if __name__ == "__main__":
    # Create dummy log file for testing
    sample_logs = [
        "proxy.cse.cuhk.edu.hk:5070 open through proxy proxy.cse.cuhk.edu.hk:5070 HTTPS\n",
        "proxy.cse.cuhk.edu.hk:5070 close, 0 bytes sent, 0 bytes received, lifetime 00:01\n",
        "proxy.cse.cuhk.edu.hk:5070 open through proxy p3p.sogou.com:80 HTTPS\n",
        "182.254.114.110:80 open through proxy 182.254.114.110:80 HTTPS\n"
    ]
    with open("input.log", "w") as f:
        f.writelines(sample_logs)

    # Initialize parser and run
    parser = BrainLogParser(weight=0.5, branch_threshold=3)
    parser.parse_file("HDFS_2k.log", "brain_temp.csv")