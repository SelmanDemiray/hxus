import numpy as np
import re
import os
import pickle

class DataProcessor:
    def __init__(self, max_sequence_length=20):
        self.word_to_idx = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
        self.idx_to_word = {0: '<PAD>', 1: '<START>', 2: '<END>', 3: '<UNK>'}
        self.vocab_size = 4
        self.max_sequence_length = max_sequence_length
    
    def preprocess_text(self, text):
        """Clean and tokenize text"""
        text = text.lower()
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
    
    def load_sample_data(self):
        """Load sample conversation data for testing"""
        questions = [
            "hello how are you",
            "what is your name",
            "how does this work",
            "tell me a joke",
            "what time is it"
        ]
        
        answers = [
            "i am doing well thank you",
            "my name is chat bot",
            "you ask questions and i try to answer them",
            "why did the chicken cross the road to get to the other side",
            "sorry i do not have access to the current time"
        ]
        
        # Build vocabulary from these samples
        self.build_vocabulary(questions + answers)
        
        return questions, answers
