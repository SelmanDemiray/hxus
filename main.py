import argparse
import os
import numpy as np
from data_processor import DataProcessor
from neural_network import EncoderDecoderNN
from training import train_model
from chat import chat_interface, test_model

def main():
    parser = argparse.ArgumentParser(description='Text Chat Neural Network')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'chat', 'test'],
                       help='Mode: train the model or chat with trained model')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to conversation data (if not provided, sample data will be used)')
    parser.add_argument('--model_path', type=str, default='models/model.pkl',
                       help='Path to save/load model')
    parser.add_argument('--processor_path', type=str, default='models/processor.pkl',
                       help='Path to save/load data processor')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--embedding_dim', type=int, default=128,
                       help='Word embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden layer dimension')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Starting training mode...")
        
        # Initialize data processor
        processor = DataProcessor(max_sequence_length=20)
        
        # Load data
        if args.data_path:
            # TODO: Implement loading custom data format
            # For now, we'll use sample data
            print(f"Custom data loading not implemented yet. Using sample data instead.")
            questions, answers = processor.load_sample_data()
        else:
            questions, answers = processor.load_sample_data()
            
        print(f"Loaded {len(questions)} conversation pairs")
        
        # Prepare data for training
        encoder_inputs, decoder_inputs, decoder_targets = processor.prepare_data(questions, answers)
        
        # Save the processor
        os.makedirs(os.path.dirname(args.processor_path), exist_ok=True)
        processor.save_processor(args.processor_path)
        
        # Initialize model
        model = EncoderDecoderNN(
            vocab_size=processor.vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim
        )
        
        # Train model
        train_model(
            encoder_inputs, decoder_inputs, decoder_targets,
            model, epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, save_dir=os.path.dirname(args.model_path)
        )
        
    elif args.mode == 'chat':
        print("Starting chat mode...")
        
        # Check if model and processor files exist
        if not os.path.exists(args.model_path):
            print(f"Model file not found: {args.model_path}")
            return
        
        if not os.path.exists(args.processor_path):
            print(f"Processor file not found: {args.processor_path}")
            return
        
        # Start chat interface
        chat_interface(args.model_path, args.processor_path)
        
    elif args.mode == 'test':
        print("Starting test mode...")
        
        # Check if model and processor files exist
        if not os.path.exists(args.model_path):
            print(f"Model file not found: {args.model_path}")
            return
        
        if not os.path.exists(args.processor_path):
            print(f"Processor file not found: {args.processor_path}")
            return
        
        # Test questions
        test_questions = [
            "hello how are you",
            "what is your name",
            "how does this work",
            "tell me a joke",
            "what time is it"
        ]
        
        # Start test
        test_model(args.model_path, args.processor_path, test_questions)

if __name__ == "__main__":
    main()
