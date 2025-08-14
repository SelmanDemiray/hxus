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
            "what time is it",
            "can you help me",
            "where are you from",
            "what can you do",
            "how old are you",
            "who created you",
            "what is the weather today",
            "do you like music",
            "can you speak other languages",
            "what is your favorite color",
            "how do i reset my password",
            "what is artificial intelligence",
            "how do i contact support",
            "can you recommend a book",
            "what is your purpose",
            "how do i update my profile",
            "what languages do you speak",
            "can you tell me a fun fact",
            "how do i delete my account",
            "what is machine learning",
            "can you set a reminder for me",
            "how do i change my email address",
            "what is your favorite food",
            "can you play games",
            "how do i subscribe to the newsletter",
            "what is deep learning",
            "can you translate text",
            "how do i log out",
            "what is your favorite movie",
            "can you tell me a story",
            "how do i enable notifications",
            "what is natural language processing",
            "can you answer math questions",
            "how do i recover my username",
            "what is your favorite animal",
            "can you send emails",
            "how do i change my password",
            # Additional training data
            "how do i make a reservation",
            "can you order food for me",
            "what is the capital of france",
            "how do i connect to wifi",
            "can you tell me the news",
            "how do i set an alarm",
            "what is your favorite sport",
            "can you check my schedule",
            "how do i turn off notifications",
            "can you tell me a riddle",
            # More training data
            "how do i book a flight",
            "can you check the weather tomorrow",
            "what is the meaning of life",
            "how do i cancel my subscription",
            "can you recommend a movie",
            "how do i change my username",
            "can you help me with homework",
            "what is your favorite book",
            "how do i set a timer",
            "can you tell me a joke about computers",
            # Even more training data
            "how do i create an account",
            "can you help me find a restaurant",
            "what is the tallest mountain",
            "how do i delete a message",
            "can you tell me a fun story",
            "how do i check my balance",
            "what is the fastest animal",
            "can you help me with directions",
            "how do i update my app",
            "can you tell me a science fact",
            "how do i send a photo",
            "can you help me with math homework",
            "what is the largest ocean",
            "how do i change my profile picture",
            "can you tell me a joke about animals",
            "how do i set my location",
            "can you help me with my schedule",
            "what is the smallest country",
            "how do i block someone",
            "can you tell me a historical fact",
            # Even more data for training
            "how do i check my email",
            "can you help me with programming",
            "what is the speed of light",
            "how do i print a document",
            "can you tell me a joke about robots",
            "how do i find nearby hotels",
            "can you help me with travel plans",
            "what is the population of japan",
            "how do i set up two factor authentication",
            "can you tell me a joke about science",
            "how do i change my notification settings",
            "can you help me with my homework",
            "what is the capital of germany",
            "how do i add a new contact",
            "can you tell me a joke about math",
            "how do i check my internet speed",
            "can you help me with a recipe",
            "what is the largest planet",
            "how do i update my billing information",
            "can you tell me a joke about computers",
            "how do i create a backup",
            "can you help me with my calendar",
            "what is the capital of italy",
            "how do i delete my browsing history",
            "can you tell me a joke about food",
            # Even more data for training (added)
            "how do i reset my device",
            "can you help me with coding",
            "what is the tallest building",
            "how do i change my language settings",
            "can you tell me a joke about animals",
            "how do i check my notifications",
            "can you help me with science homework",
            "what is the capital of spain",
            "how do i set a profile picture",
            "can you tell me a joke about sports",
            "how do i update my contact information",
            "can you help me with geography homework",
            "what is the largest desert",
            "how do i change my theme",
            "can you tell me a joke about school",
            "how do i check my reminders",
            "can you help me with history homework",
            "what is the smallest planet",
            "how do i block notifications",
            "can you tell me a joke about teachers",
            # Even more data for training (newly added)
            "how do i check my flight status",
            "can you help me with chemistry homework",
            "what is the capital of canada",
            "how do i set a meeting reminder",
            "can you tell me a joke about engineers",
            "how do i find a nearby pharmacy",
            "can you help me with my resume",
            "what is the population of australia",
            "how do i enable dark mode",
            "can you tell me a joke about doctors",
            "how do i change my privacy settings",
            "can you help me with my shopping list",
            "what is the capital of brazil",
            "how do i add an event to my calendar",
            "can you tell me a joke about lawyers",
            "how do i check my device storage",
            "can you help me with my travel itinerary",
            "what is the largest continent",
            "how do i update my payment method",
            "can you tell me a joke about programmers"
        ]

        answers = [
            "i am doing well thank you",
            "my name is chat bot",
            "you ask questions and i try to answer them",
            "why did the chicken cross the road to get to the other side",
            "sorry i do not have access to the current time",
            "yes i can help you with your questions",
            "i am a virtual assistant created by developers",
            "i can answer questions and help with information",
            "i do not have an age i am a program",
            "i was created by a team of engineers",
            "sorry i cannot check the weather",
            "yes i enjoy listening to music",
            "i can understand and reply in several languages",
            "i do not have preferences as i am an ai",
            "to reset your password follow the instructions on the website",
            "artificial intelligence is the simulation of human intelligence by machines",
            "you can contact support through the help section on the website",
            "i recommend reading to kill a mockingbird",
            "my purpose is to assist and provide information",
            "to update your profile go to the settings page",
            "i can speak english and understand several other languages",
            "here is a fun fact honey never spoils",
            "to delete your account follow the instructions in your account settings",
            "machine learning is a field of ai that enables computers to learn from data",
            "i can set reminders if you tell me the details",
            "to change your email address go to your account settings",
            "i do not eat food as i am a virtual assistant",
            "yes i can play simple games with you",
            "to subscribe to the newsletter visit the subscription page",
            "deep learning is a subset of machine learning using neural networks",
            "yes i can translate text between several languages",
            "to log out click the log out button in your account menu",
            "i do not watch movies but i know many popular ones",
            "sure here is a short story once upon a time there was a curious user",
            "to enable notifications go to your settings and turn them on",
            "natural language processing is a field of ai focused on understanding human language",
            "yes i can help with math questions just ask",
            "to recover your username use the forgot username option on the login page",
            "i do not have a favorite animal but i know many facts about animals",
            "yes i can help you send emails if you provide the details",
            "to change your password go to your account settings and follow the instructions",
            # Additional training data answers
            "to make a reservation please provide the details and i will assist you",
            "i can help you order food if you tell me what you would like",
            "the capital of france is paris",
            "to connect to wifi go to your device settings and select a network",
            "here is the latest news i found online",
            "to set an alarm tell me the time you want to wake up",
            "i do not have a favorite sport but i know many sports facts",
            "i can check your schedule if you give me access to your calendar",
            "to turn off notifications go to your settings and disable them",
            "here is a riddle what has keys but cant open locks a piano",
            # More training data answers
            "to book a flight please provide your destination and dates",
            "the weather tomorrow is expected to be sunny",
            "the meaning of life is a philosophical question with many answers",
            "to cancel your subscription go to your account settings and select cancel",
            "i recommend watching inception",
            "to change your username go to your profile settings",
            "yes i can help you with your homework just tell me the subject",
            "my favorite book is pride and prejudice",
            "to set a timer tell me the duration",
            "here is a computer joke why do programmers prefer dark mode because light attracts bugs",
            # Even more training data answers
            "to create an account go to the sign up page and follow the instructions",
            "i can help you find a restaurant if you tell me your location",
            "the tallest mountain is mount everest",
            "to delete a message select the message and choose delete",
            "once upon a time there was a clever fox who loved stories",
            "to check your balance log in to your account and view your dashboard",
            "the fastest animal is the cheetah",
            "i can help you with directions if you provide your destination",
            "to update your app go to the app store and check for updates",
            "here is a science fact water is made of hydrogen and oxygen",
            "to send a photo select the photo and choose send",
            "yes i can help you with math homework just tell me the problem",
            "the largest ocean is the pacific ocean",
            "to change your profile picture go to your profile and upload a new photo",
            "here is a joke why did the cow become an astronaut to see the moooon",
            "to set your location go to your settings and enable location services",
            "i can help you organize your schedule if you give me your appointments",
            "the smallest country is vatican city",
            "to block someone go to their profile and select block",
            "here is a historical fact the first computer was invented in the 1940s",
            # Even more data for training answers
            "to check your email open your email app or website",
            "yes i can help you with programming just tell me your question",
            "the speed of light is approximately three hundred thousand kilometers per second",
            "to print a document select print from the file menu",
            "here is a robot joke why did the robot go on vacation to recharge its batteries",
            "to find nearby hotels use a maps app or website",
            "i can help you with travel plans if you tell me your destination",
            "the population of japan is over one hundred million",
            "to set up two factor authentication go to your account security settings",
            "here is a science joke why did the scientist install a knocker on his door to win the no bell prize",
            "to change your notification settings go to your settings menu",
            "yes i can help you with your homework just tell me the subject",
            "the capital of germany is berlin",
            "to add a new contact go to your contacts app and select add",
            "here is a math joke why was the equal sign so humble because it knew it was not greater than or less than anyone else",
            "to check your internet speed use an online speed test",
            "i can help you with a recipe if you tell me what you want to cook",
            "the largest planet is jupiter",
            "to update your billing information go to your account settings",
            "here is a computer joke why do computers get cold because they have windows",
            "to create a backup use your device's backup feature",
            "i can help you with your calendar if you give me access",
            "the capital of italy is rome",
            "to delete your browsing history go to your browser settings",
            "here is a food joke why did the tomato turn red because it saw the salad dressing",
            # Even more data for training answers (added)
            "to reset your device go to your settings and select reset",
            "yes i can help you with coding just tell me your question",
            "the tallest building is burj khalifa",
            "to change your language settings go to your settings and select language",
            "here is an animal joke why did the cat sit on the computer to keep an eye on the mouse",
            "to check your notifications open your notification panel",
            "yes i can help you with science homework just tell me the topic",
            "the capital of spain is madrid",
            "to set a profile picture go to your profile and upload a photo",
            "here is a sports joke why did the football team go to the bank to get their quarterback",
            "to update your contact information go to your profile settings",
            "yes i can help you with geography homework just tell me the question",
            "the largest desert is the sahara",
            "to change your theme go to your settings and select theme",
            "here is a school joke why was the math book sad because it had too many problems",
            "to check your reminders open your reminders app",
            "yes i can help you with history homework just tell me the topic",
            "the smallest planet is mercury",
            "to block notifications go to your notification settings",
            "here is a teacher joke why did the teacher wear sunglasses because her students were so bright",
            # Even more data for training answers (newly added)
            "to check your flight status visit your airline's website or app",
            "yes i can help you with chemistry homework just tell me the topic",
            "the capital of canada is ottawa",
            "to set a meeting reminder tell me the date and time",
            "here is an engineer joke why did the engineer cross the road to get to the other side of the equation",
            "to find a nearby pharmacy use a maps app or website",
            "i can help you with your resume if you provide your work experience",
            "the population of australia is about twenty five million",
            "to enable dark mode go to your settings and select dark mode",
            "here is a doctor joke why did the doctor carry a red pen in case they needed to draw blood",
            "to change your privacy settings go to your account settings",
            "i can help you with your shopping list if you tell me what you need",
            "the capital of brazil is brasilia",
            "to add an event to your calendar tell me the details",
            "here is a lawyer joke why did the lawyer wear a neck brace to help his case",
            "to check your device storage go to your settings and select storage",
            "i can help you with your travel itinerary if you provide your plans",
            "the largest continent is asia",
            "to update your payment method go to your account billing settings",
            "here is a programmer joke why do programmers hate nature because it has too many bugs"
        ]

        # Build vocabulary from these samples
        self.build_vocabulary(questions + answers)

        return questions, answers
