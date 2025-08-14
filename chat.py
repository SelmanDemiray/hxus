import numpy as np
from neural_network import EncoderDecoderNN
from data_processor import DataProcessor

def chat_interface(model_path, processor_path):
    """
    Interactive chat interface for testing models
    
    Parameters:
    -----------
    model_path : str
        Path to the trained model file
    processor_path : str
        Path to the saved data processor
    """
    # Load model and processor
    model = EncoderDecoderNN.load_model(model_path)
    processor = DataProcessor.load_processor(processor_path)
    
    print("\n===== Chat Bot Interface =====")
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'help' for assistance")
    print("================================\n")
    
    while True:
        user_input = input("You: ").strip().lower()
        
        if user_input in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        if user_input == 'help':
            print("\n--- Help Menu ---")
            print("- This is a simple chat bot trained on a neural network")
            print("- Type your message and press Enter to get a response")
            print("- Type 'quit' or 'exit' to end the conversation")
            print("----------------\n")
            continue
        
        if not user_input:
            print("Please type something!")
            continue
        
        # Generate response
        response = model.predict(user_input, processor)
        print("Bot:", response)

def test_model(model_path, processor_path, test_questions):
    """
    Test model on predefined questions
    
    Parameters:
    -----------
    model_path : str
        Path to the trained model file
    processor_path : str
        Path to the saved data processor
    test_questions : list
        List of test questions
    """
    # Load model and processor
    model = EncoderDecoderNN.load_model(model_path)
    processor = DataProcessor.load_processor(processor_path)
    
    print("\n===== Model Testing =====")
    
    for i, question in enumerate(test_questions):
        response = model.predict(question, processor)
        print(f"Q{i+1}: {question}")
        print(f"A{i+1}: {response}")
        print()
    
    print("========================\n")
