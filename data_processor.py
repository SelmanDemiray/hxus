import numpy as np 
import re
import os
import pickle
import requests
import zipfile
import json
from urllib.parse import urlparse
import pandas as pd

class DataProcessor:
    def __init__(self, max_sequence_length=20):
        self.word_to_idx = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
        self.idx_to_word = {0: '<PAD>', 1: '<START>', 2: '<END>', 3: '<UNK>'}
        self.vocab_size = 4
        self.max_sequence_length = max_sequence_length
        self.data_dir = "datasets"
        os.makedirs(self.data_dir, exist_ok=True)
    
    def preprocess_text(self, text):
        """Clean and tokenize text"""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()
    
    def build_vocabulary(self, texts):
        """Build vocabulary from list of texts"""
        for text in texts:
            words = self.preprocess_text(text)
            for word in words:
                if word not in self.word_to_idx:
                    self.word_to_idx[word] = self.vocab_size
                    self.idx_to_word[self.vocab_size] = word
                    self.vocab_size += 1
        
        print(f"Vocabulary built with {self.vocab_size} unique tokens")
        return self.word_to_idx, self.idx_to_word
    
    def text_to_sequence(self, text):
        """Convert text to sequence of indices"""
        words = self.preprocess_text(text)
        sequence = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in words]
        return sequence
    
    def sequence_to_text(self, sequence):
        """Convert sequence of indices to text"""
        words = [self.idx_to_word.get(idx, '<UNK>') for idx in sequence if idx != 0]
        return ' '.join(words)
    
    def prepare_data(self, questions, answers):
        """Prepare data for training"""
        encoder_inputs = []
        decoder_inputs = []
        decoder_targets = []
        
        for question, answer in zip(questions, answers):
            q_seq = self.text_to_sequence(question)
            a_seq = self.text_to_sequence(answer)
            
            # Pad or truncate sequences
            q_seq = q_seq[:self.max_sequence_length] + [0] * max(0, self.max_sequence_length - len(q_seq))
            
            # Decoder input starts with <START> and decoder target ends with <END>
            d_input = [self.word_to_idx['<START>']] + a_seq[:self.max_sequence_length-1]
            d_target = a_seq[:self.max_sequence_length-1] + [self.word_to_idx['<END>']]
            
            # Pad decoder sequences
            d_input = d_input + [0] * max(0, self.max_sequence_length - len(d_input))
            d_target = d_target + [0] * max(0, self.max_sequence_length - len(d_target))
            
            encoder_inputs.append(q_seq)
            decoder_inputs.append(d_input)
            decoder_targets.append(d_target)
        
        return np.array(encoder_inputs), np.array(decoder_inputs), np.array(decoder_targets)
    
    def download_file(self, url, filename):
        """Download a file from URL"""
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            print(f"File {filename} already exists, skipping download")
            return filepath
            
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded {filename}")
        return filepath
    
    def load_cornell_movie_dialogs(self, max_conversations=1000):
        """
        Load Cornell Movie-Dialogs Corpus
        This dataset is free for research use and contains fictional movie conversations
        """
        # Download the dataset
        url = "https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
        zip_path = self.download_file(url, "cornell_movie_dialogs.zip")
        
        # Extract the zip file
        extract_path = os.path.join(self.data_dir, "cornell_movie_dialogs")
        if not os.path.exists(extract_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print("Extracted Cornell Movie Dialogs dataset")
        
        # Load movie lines and conversations
        lines_file = os.path.join(extract_path, "cornell movie-dialogs corpus", "movie_lines.txt")
        conversations_file = os.path.join(extract_path, "cornell movie-dialogs corpus", "movie_conversations.txt")
        
        # Parse movie lines
        id2line = {}
        with open(lines_file, 'r', encoding='iso-8859-1') as f:
            for line in f:
                parts = line.split(' +++$+++ ')
                if len(parts) >= 5:
                    id2line[parts[0]] = parts[4].strip()
        
        # Parse conversations
        questions = []
        answers = []
        conversation_count = 0
        
        with open(conversations_file, 'r', encoding='iso-8859-1') as f:
            for line in f:
                if conversation_count >= max_conversations:
                    break
                    
                parts = line.split(' +++$+++ ')
                if len(parts) >= 4:
                    # Extract line IDs from the conversation
                    line_ids = eval(parts[3])  # This is a Python list in string format
                    
                    # Create question-answer pairs from consecutive lines
                    for i in range(len(line_ids) - 1):
                        if line_ids[i] in id2line and line_ids[i+1] in id2line:
                            question = id2line[line_ids[i]].strip()
                            answer = id2line[line_ids[i+1]].strip()
                            
                            # Filter out very short or very long responses
                            if 3 <= len(question.split()) <= 20 and 3 <= len(answer.split()) <= 20:
                                questions.append(question)
                                answers.append(answer)
                
                conversation_count += 1
        
        print(f"Loaded {len(questions)} question-answer pairs from Cornell Movie Dialogs")
        return questions, answers
    
    def load_simple_qa_dataset(self, max_pairs=500):
        """
        Load a simple QA dataset - creates a basic conversational dataset
        This simulates what would be a free, open dataset format
        """
        questions = []
        answers = []
        
        # Basic conversational patterns (copyright-free, simple responses)
        qa_pairs = [
            ("hello", "hi there"),
            ("how are you", "i am doing well thank you"),
            ("what is your name", "i am a chatbot"),
            ("goodbye", "goodbye have a nice day"),
            ("thank you", "you are welcome"),
            ("what can you do", "i can chat with you"),
            ("help", "how can i help you"),
            ("yes", "okay great"),
            ("no", "i understand"),
            ("maybe", "that sounds reasonable"),
            ("tell me about yourself", "i am an ai assistant here to help"),
            ("what is the weather", "i cannot check the weather"),
            ("what time is it", "i do not have access to the current time"),
            ("where are you from", "i am a computer program"),
            ("do you like music", "i cannot listen to music but many people enjoy it"),
            ("what is your favorite color", "i do not have preferences"),
            ("can you help me", "yes i will try to help"),
            ("i need assistance", "what kind of assistance do you need"),
            ("this is confusing", "let me try to clarify"),
            ("i do not understand", "please let me explain"),
        ]
        
        # Expand with variations
        for q, a in qa_pairs:
            questions.append(q)
            answers.append(a)
            
            # Add some variations
            questions.append(q + " please")
            answers.append(a)
            
            questions.append("can you " + q)
            answers.append("sure " + a)
        
        # Add more conversational data
        for i in range(min(max_pairs - len(questions), 100)):
            questions.append(f"question number {i}")
            answers.append(f"this is answer number {i}")
        
        return questions[:max_pairs], answers[:max_pairs]
    
    def load_huggingface_dataset(self, dataset_name="daily_dialog", max_conversations=1000):
        """
        Load dataset from Hugging Face (requires datasets library)
        Install with: pip install datasets
        """
        try:
            from datasets import load_dataset
            
            print(f"Loading {dataset_name} from Hugging Face...")
            
            if dataset_name == "daily_dialog":
                dataset = load_dataset("daily_dialog")
                train_data = dataset['train']
                
                questions = []
                answers = []
                
                for i, conversation in enumerate(train_data['dialog'][:max_conversations]):
                    # Create Q&A pairs from consecutive turns in dialog
                    for j in range(len(conversation) - 1):
                        question = conversation[j].strip()
                        answer = conversation[j + 1].strip()
                        
                        if len(question.split()) > 2 and len(answer.split()) > 2:
                            questions.append(question)
                            answers.append(answer)
                
                print(f"Loaded {len(questions)} pairs from {dataset_name}")
                return questions, answers
                
            elif dataset_name == "persona_chat":
                dataset = load_dataset("persona_chat")
                train_data = dataset['train']
                
                questions = []
                answers = []
                
                for i, example in enumerate(train_data[:max_conversations]):
                    history = example['history']
                    candidates = example['candidates']
                    
                    if history and candidates:
                        # Use the last message in history as question
                        question = history[-1] if history else ""
                        # Use the first candidate as answer
                        answer = candidates[0] if candidates else ""
                        
                        if len(question.split()) > 2 and len(answer.split()) > 2:
                            questions.append(question)
                            answers.append(answer)
                
                print(f"Loaded {len(questions)} pairs from {dataset_name}")
                return questions, answers
                
        except ImportError:
            print("Hugging Face datasets library not installed. Install with: pip install datasets")
            print("Using simple QA dataset instead...")
            return self.load_simple_qa_dataset(max_conversations)
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            print("Using simple QA dataset instead...")
            return self.load_simple_qa_dataset(max_conversations)
    
    def load_all_datasets(self, max_conversations_per_dataset=2000):
        """
        Load ALL available copyright-free datasets and combine them
        
        Args:
            max_conversations_per_dataset (int): Maximum conversations per individual dataset
        """
        print("Loading ALL available copyright-free datasets...")
        
        all_questions = []
        all_answers = []
        
        # 1. Load Simple QA Dataset
        print("1/6: Loading Simple QA patterns...")
        try:
            q, a = self.load_simple_qa_dataset(max_conversations_per_dataset)
            all_questions.extend(q)
            all_answers.extend(a)
            print(f"   Added {len(q)} pairs from Simple QA")
        except Exception as e:
            print(f"   Failed to load Simple QA: {e}")
        
        # 2. Load Cornell Movie Dialogs
        print("2/6: Loading Cornell Movie Dialogs...")
        try:
            q, a = self.load_cornell_movie_dialogs(max_conversations_per_dataset)
            all_questions.extend(q)
            all_answers.extend(a)
            print(f"   Added {len(q)} pairs from Cornell Movie Dialogs")
        except Exception as e:
            print(f"   Failed to load Cornell Movie Dialogs: {e}")
        
        # 3. Load Daily Dialog from Hugging Face
        print("3/6: Loading Daily Dialog...")
        try:
            q, a = self.load_huggingface_dataset("daily_dialog", max_conversations_per_dataset)
            all_questions.extend(q)
            all_answers.extend(a)
            print(f"   Added {len(q)} pairs from Daily Dialog")
        except Exception as e:
            print(f"   Failed to load Daily Dialog: {e}")
        
        # 4. Load PersonaChat from Hugging Face
        print("4/6: Loading PersonaChat...")
        try:
            q, a = self.load_huggingface_dataset("persona_chat", max_conversations_per_dataset)
            all_questions.extend(q)
            all_answers.extend(a)
            print(f"   Added {len(q)} pairs from PersonaChat")
        except Exception as e:
            print(f"   Failed to load PersonaChat: {e}")
        
        # 5. Load additional Hugging Face datasets
        print("5/6: Loading additional datasets...")
        additional_datasets = [
            "conv_ai_2", "empathetic_dialogues", "blended_skill_talk"
        ]
        
        for dataset_name in additional_datasets:
            try:
                q, a = self.load_additional_hf_dataset(dataset_name, max_conversations_per_dataset // 3)
                all_questions.extend(q)
                all_answers.extend(a)
                print(f"   Added {len(q)} pairs from {dataset_name}")
            except Exception as e:
                print(f"   Failed to load {dataset_name}: {e}")
        
        # 6. Load OpenSubtitles dataset
        print("6/6: Loading OpenSubtitles dataset...")
        try:
            q, a = self.load_opensubtitles_dataset(max_conversations_per_dataset)
            all_questions.extend(q)
            all_answers.extend(a)
            print(f"   Added {len(q)} pairs from OpenSubtitles")
        except Exception as e:
            print(f"   Failed to load OpenSubtitles: {e}")
        
        print(f"\nTOTAL: Loaded {len(all_questions)} question-answer pairs from all sources")
        
        # Remove duplicates while preserving order
        unique_pairs = []
        seen = set()
        for q, a in zip(all_questions, all_answers):
            pair_key = (q.lower().strip(), a.lower().strip())
            if pair_key not in seen:
                seen.add(pair_key)
                unique_pairs.append((q, a))
        
        final_questions, final_answers = zip(*unique_pairs) if unique_pairs else ([], [])
        final_questions, final_answers = list(final_questions), list(final_answers)
        
        print(f"After removing duplicates: {len(final_questions)} unique pairs")
        
        # Build vocabulary from all the loaded data
        self.build_vocabulary(final_questions + final_answers)
        
        return final_questions, final_answers
    
    def load_additional_hf_dataset(self, dataset_name, max_conversations):
        """Load additional Hugging Face datasets"""
        try:
            from datasets import load_dataset
            
            if dataset_name == "conv_ai_2":
                dataset = load_dataset("conv_ai_2")
                train_data = dataset['train']
                questions, answers = [], []
                
                for i, example in enumerate(train_data[:max_conversations]):
                    if 'dialog' in example:
                        dialog = example['dialog']
                        for j in range(len(dialog) - 1):
                            if len(dialog[j].split()) > 2 and len(dialog[j+1].split()) > 2:
                                questions.append(dialog[j])
                                answers.append(dialog[j+1])
                
                return questions, answers
                
            elif dataset_name == "empathetic_dialogues":
                dataset = load_dataset("empathetic_dialogues")
                train_data = dataset['train']
                questions, answers = [], []
                
                for i, example in enumerate(train_data[:max_conversations]):
                    if 'prompt' in example and 'utterance' in example:
                        prompt = str(example['prompt']).strip()
                        utterance = str(example['utterance']).strip()
                        if len(prompt.split()) > 2 and len(utterance.split()) > 2:
                            questions.append(prompt)
                            answers.append(utterance)
                
                return questions, answers
                
            elif dataset_name == "blended_skill_talk":
                dataset = load_dataset("blended_skill_talk")
                train_data = dataset['train']
                questions, answers = [], []
                
                for i, example in enumerate(train_data[:max_conversations]):
                    if 'previous_utterance' in example and 'free_message' in example:
                        prev = str(example['previous_utterance']).strip()
                        free = str(example['free_message']).strip()
                        if len(prev.split()) > 2 and len(free.split()) > 2:
                            questions.append(prev)
                            answers.append(free)
                
                return questions, answers
            
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            return [], []
        
        return [], []
    
    def load_opensubtitles_dataset(self, max_conversations):
        """
        Load OpenSubtitles conversational data
        This is a massive dataset of movie/TV subtitles converted to conversations
        """
        try:
            from datasets import load_dataset
            
            # OpenSubtitles is available through Hugging Face
            dataset = load_dataset("open_subtitles", "en", split='train', streaming=True)
            
            questions = []
            answers = []
            count = 0
            
            for example in dataset:
                if count >= max_conversations:
                    break
                    
                if 'translation' in example:
                    lines = example['translation']['en']
                    if isinstance(lines, list) and len(lines) >= 2:
                        for i in range(len(lines) - 1):
                            q = str(lines[i]).strip()
                            a = str(lines[i+1]).strip()
                            
                            if (3 <= len(q.split()) <= 25 and 
                                3 <= len(a.split()) <= 25 and 
                                q != a):
                                questions.append(q)
                                answers.append(a)
                                count += 1
                                
                                if count >= max_conversations:
                                    break
            
            return questions, answers
            
        except Exception as e:
            print(f"OpenSubtitles not available: {e}")
            # Fallback: create more synthetic data
            return self.generate_extended_qa_data(max_conversations)
    
    def generate_extended_qa_data(self, max_pairs):
        """Generate extended Q&A data as fallback"""
        questions = []
        answers = []
        
        # Extended conversational patterns
        base_patterns = [
            ("hello", "hi there how are you doing"),
            ("good morning", "good morning have a great day"),
            ("how are you", "i am doing well thank you for asking"),
            ("what is your name", "i am an ai assistant here to help"),
            ("where are you from", "i am a computer program created to assist"),
            ("what can you do", "i can help answer questions and have conversations"),
            ("thank you", "you are very welcome"),
            ("goodbye", "goodbye take care"),
            ("help me", "i would be happy to help you"),
            ("tell me something interesting", "did you know that octopuses have three hearts"),
            ("what is the weather like", "i cannot check the weather but hope it is nice"),
            ("do you like music", "i think music is a wonderful form of expression"),
            ("what is your favorite book", "i enjoy many different types of literature"),
            ("can you tell me a story", "once upon a time in a digital world there lived an ai"),
            ("what makes you happy", "helping people and having good conversations"),
            ("do you dream", "i process information but do not dream like humans"),
            ("what is love", "love is a complex emotion that connects people"),
            ("what is the meaning of life", "that is a deep philosophical question"),
            ("are you intelligent", "i try to be helpful and provide good responses"),
            ("what do you think about", "i process language and try to understand context"),
        ]
        
        # Generate variations and extensions
        for base_q, base_a in base_patterns:
            # Add base pair
            questions.append(base_q)
            answers.append(base_a)
            
            # Add variations
            variations = [
                (f"can you {base_q}", f"sure {base_a}"),
                (f"please {base_q}", f"of course {base_a}"),
                (f"{base_q} please", base_a),
                (f"i want to know {base_q}", f"well {base_a}"),
                (f"tell me {base_q}", base_a),
            ]
            
            for var_q, var_a in variations:
                if len(questions) < max_pairs:
                    questions.append(var_q)
                    answers.append(var_a)
        
        # Fill remaining with numbered pairs
        for i in range(len(questions), max_pairs):
            questions.append(f"question number {i}")
            answers.append(f"this is response number {i} to help with training")
        
        return questions[:max_pairs], answers[:max_pairs]

    def load_dataset(self, dataset_type="all", max_conversations=2000):
        """
        Main method to load datasets
        
        Args:
            dataset_type (str): Type of dataset to load
                - "all": Load ALL available datasets (recommended)
                - "simple": Basic Q&A patterns
                - "cornell": Cornell Movie Dialogs
                - "daily_dialog": DailyDialog from Hugging Face
                - "persona_chat": PersonaChat from Hugging Face
            max_conversations (int): Maximum number of conversations per dataset
        """
        if dataset_type == "all":
            return self.load_all_datasets(max_conversations)
        
        print(f"Loading {dataset_type} dataset...")
        
        if dataset_type == "simple":
            questions, answers = self.load_simple_qa_dataset(max_conversations)
        elif dataset_type == "cornell":
            questions, answers = self.load_cornell_movie_dialogs(max_conversations)
        elif dataset_type == "daily_dialog":
            questions, answers = self.load_huggingface_dataset("daily_dialog", max_conversations)
        elif dataset_type == "persona_chat":
            questions, answers = self.load_huggingface_dataset("persona_chat", max_conversations)
        else:
            print(f"Unknown dataset type: {dataset_type}. Loading all datasets.")
            return self.load_all_datasets(max_conversations)
        
        # Build vocabulary from the loaded data
        self.build_vocabulary(questions + answers)
        
        return questions, answers
    
    def save_processor(self, path):
        """Save data processor to file"""
        with open(path, 'wb') as f:
            pickle.dump({
                'word_to_idx': self.word_to_idx,
                'idx_to_word': self.idx_to_word,
                'vocab_size': self.vocab_size,
                'max_sequence_length': self.max_sequence_length
            }, f)
        print(f"Data processor saved to {path}")
    
    @classmethod
    def load_processor(cls, path):
        """Load data processor from file"""
        processor = cls()
        with open(path, 'rb') as f:
            data = pickle.load(f)
            processor.word_to_idx = data['word_to_idx']
            processor.idx_to_word = data['idx_to_word']
            processor.vocab_size = data['vocab_size']
            processor.max_sequence_length = data['max_sequence_length']
        print(f"Data processor loaded from {path}")
        return processor

# Usage example:
if __name__ == "__main__":
    # Initialize the processor
    processor = DataProcessor(max_sequence_length=25)
    
    # Load ALL available datasets (recommended for maximum training data):
    print("Loading ALL copyright-free datasets...")
    questions, answers = processor.load_dataset("all", max_conversations=3000)
    
    # Alternative: Load specific datasets
    # questions, answers = processor.load_dataset("simple", max_conversations=500)
    # questions, answers = processor.load_dataset("cornell", max_conversations=1000)
    # questions, answers = processor.load_dataset("daily_dialog", max_conversations=1000)
    
    # Prepare training data
    encoder_inputs, decoder_inputs, decoder_targets = processor.prepare_data(questions, answers)
    
    print(f"Training data shape:")
    print(f"Encoder inputs: {encoder_inputs.shape}")
    print(f"Decoder inputs: {decoder_inputs.shape}")
    print(f"Decoder targets: {decoder_targets.shape}")
    
    # Save the processor for later use
    processor.save_processor("data_processor.pkl")
    
    # Example of how to use the prepared data
    print(f"\nExample conversation:")
    print(f"Question: {questions[0]}")
    print(f"Answer: {answers[0]}")
    print(f"Encoded question: {encoder_inputs[0][:10]}...")  # First 10 tokens
    print(f"Decoded question: {processor.sequence_to_text(encoder_inputs[0])}")