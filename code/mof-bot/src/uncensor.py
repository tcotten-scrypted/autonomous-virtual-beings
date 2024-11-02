import os
import json
import re

DATA_FILE = os.path.join(os.path.dirname(__file__), '../data/uncensor_mappings.json')

class Uncensor:
    OBSCURING_SYMBOLS = ['\*', '#', '@', '!', '\$', '%', '\^', '&']
    REGEX_HAS_OBSCURING_SYMBOL = re.compile(f"[{''.join(re.escape(symbol) for symbol in OBSCURING_SYMBOLS)}]")
    
    def __init__(self, config_file=DATA_FILE):
        with open(config_file, 'r') as file:
            self.vulgarity_patterns = json.load(file)
        
        # Update patterns with list of approved obscuring symbols
        self.vulgarity_patterns = self._process_vulgarity_patterns(self.vulgarity_patterns)
        print(self.vulgarity_patterns)
        
        # Compile patterns from JSON for regex matching
        self.compiled_patterns = {
            word: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for word, patterns in self.vulgarity_patterns.items()
        }
        
    def _process_vulgarity_patterns(self, vulgarity_patterns):
        # Escape each character in OBSCURING_SYMBOLS to ensure special characters are treated as literals
        symbol_pattern = f"[{''.join(Uncensor.OBSCURING_SYMBOLS)}]"

        processed_patterns = {}
        for key, patterns in vulgarity_patterns.items():
            processed_patterns[key] = [
                pattern.replace('�', symbol_pattern) + ".*" for pattern in patterns
            ]
        
        return processed_patterns

    def _detect_pattern(self, obscured_word):
        # Check each compiled pattern to see if it matches the obscured word
        for word, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.fullmatch(obscured_word):
                    return word, pattern.pattern  # Return both the word and the matching pattern
        return None, None

    def _repair_word(self, obscured_word):
        # Detect which base word and pattern matched the obscured word
        base_word, matched_pattern = self._detect_pattern(obscured_word)
        if not base_word:
            return obscured_word  # No match found, return the original word

        # Locate the section of the obscured word that aligns with the base word
        match = re.search(r'[\*\-\#\@]+', obscured_word)
        if match:
            start_index = match.start()
            end_index = match.end()

            # Calculate the replacement section from the base word
            replaced_section = base_word[:end_index - start_index]

            # Construct the repaired word
            repaired_word = (
                obscured_word[:start_index] +  # Part before the censor symbols
                replaced_section +             # Replaced section from base_word
                obscured_word[end_index:]      # Trailing characters (punctuation, etc.)
            )

            # Match capitalization of the original obscured word
            if obscured_word.isupper():
                return repaired_word.upper()
            elif obscured_word[0].isupper():
                return repaired_word.capitalize()
            else:
                return repaired_word

        return obscured_word  # Return original if no alignment found
    
    def identify(self, words):
        words_needing_repair = [] 
        
        for word_pair in words:
            word = word_pair[0]
            # Test to see if one of the symbols is found in the word
            if Uncensor.REGEX_HAS_OBSCURING_SYMBOL.search(word):
                repaired_word = self._repair_word(word)
                if repaired_word != word:
                    words_needing_repair.append([repaired_word, word_pair[1]])
            
        return words_needing_repair

    def repair_text(self, text):
        words = self.breakdown(text)
        repairable = self.identify(words)
        
        for repair in repairable:
            print(repair)
            text = text[:repair[1]] + repair[0] + text[repair[1] + len(repair[0]):]
            
        return text
    
    def breakdown(self, text):
        words_with_positions = []
        # Find all words and their starting positions in the text
        for match in re.finditer(r'\S+', text):
            word = match.group()
            start = match.start()
            words_with_positions.append([word, start])
            
        return words_with_positions

# Usage
repairer = Uncensor()
repaired_text = repairer.repair_text("you A**!!! complete a!!holio. what an a** thing to say ***hole. obvious ---hole move. This is a d--- fine example of a f**king good day!")
print(repaired_text)
