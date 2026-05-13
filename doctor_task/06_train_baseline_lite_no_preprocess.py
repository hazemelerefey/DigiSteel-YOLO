"""
Step 6-LITE: Train Lightweight CNN WITHOUT Preprocessing (CPU Optimized)
Purpose: Fast baseline training on CPU
Changes:
- 20 epochs instead of 50 (faster iteration)
- Batch size 16 instead of 32 (less memory)
- Simplified metrics (just accuracy)
- Early stopping patience 5 (faster)
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from pathlib import Path
import json
from datetime import datetime

class LiteTrainer:
    def __init__(self, model_path, data_path, output_path):
        self.model_path = model_path
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.model = keras.models.load_model(model_path)
        self.history = {}
    
    def load_images_labels(self, split):
        """Load images and labels"""
        images_dir = self.data_path / split / 'images'
        labels_dir = self.data_path / split / 'labels'
        
        images, labels = [], []
        
        image_files = sorted(images_dir.glob('*.jpg'))
        print(f"Loading {split}: {len(image_files)} images...")
        
        for img_file in image_files:
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=-1)
            images.append(img)
            
            label_file = labels_dir / f"{img_file.stem}.txt"
            with open(label_file, 'r') as f:
                lines = f.readlines()
                class_id = int(lines[0].split()[0]) if lines else 0
                label = keras.utils.to_categorical(class_id, 6)
                labels.append(label)
        
        return np.array(images), np.array(labels)
    
    def train(self, epochs=20, batch_size=16):
        """Train model - LIGHTWEIGHT"""
        print("\n=== Loading Data (WITHOUT Preprocessing) ===")
        X_train, y_train = self.load_images_labels('train')
        X_val, y_val = self.load_images_labels('val')
        
        print(f"Train: {X_train.shape}, Val: {X_val.shape}")
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,  # Stop faster on CPU
                restore_best_weights=True
            )
        ]
        
        print(f"\n=== Training (Epochs: {epochs}, Batch: {batch_size}) ===")
        print("This will take 20-40 minutes on CPU...")
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history.history
        return history
    
    def evaluate_on_test(self):
        """Evaluate on test set"""
        print("\n=== Evaluating on Test Set ===")
        X_test, y_test = self.load_images_labels('test')
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        test_loss, test_accuracy = results
        
        metrics = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
        }
        
        return metrics
    
    def save_results(self, metrics, tag="lite_baseline_without_preprocessing"):
        """Save results"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'tag': tag,
            'training_history': self.history,
            'test_metrics': metrics
        }
        
        results_file = self.output_path / f"{tag}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results saved to {results_file}")

if __name__ == "__main__":
    print("=" * 70)
    print("Step 6-LITE: Train Lightweight CNN (WITHOUT Preprocessing)")
    print("=" * 70)
    
    trainer = LiteTrainer(
        model_path="doctor_task/models/cnn_baseline_lite.h5",
        data_path="doctor_task/data/NEU-DET-subset-lite",
        output_path="doctor_task/logs"
    )
    
    # Train (LIGHTWEIGHT: 20 epochs, batch 16)
    history = trainer.train(epochs=20, batch_size=16)
    
    # Evaluate
    metrics = trainer.evaluate_on_test()
    
    # Print results
    print("\n=== Test Metrics (WITHOUT Preprocessing) ===")
    print(f"Loss: {metrics['test_loss']:.4f}")
    print(f"Accuracy: {metrics['test_accuracy']:.4f}")
    
    # Save
    trainer.model.save("doctor_task/models/cnn_baseline_lite_no_preprocess.h5")
    trainer.save_results(metrics)
    
    print("\n✓ Training COMPLETE")
    print("✓ Time saved: 50-75% faster than full version!")
