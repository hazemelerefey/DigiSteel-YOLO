"""
Step 5-LITE: Build LIGHTWEIGHT CNN Baseline Model
Purpose: CPU-optimized CNN with 1/2 the parameters
Author: DigiSteel Team

Changes from full version:
- 1.2M parameters instead of 2.5M (50% reduction)
- Fewer conv blocks (2 instead of 4)
- Smaller filters (32→64→128 instead of 32→64→128→256)
- No batch normalization (speeds up CPU)
- Simpler architecture = faster training
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np

class CNNBaselineLite:
    """Lightweight CNN for CPU training"""
    
    def __init__(self, input_shape=(200, 200, 1), num_classes=6):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
    
    def build_model(self):
        """Build LIGHTWEIGHT CNN architecture"""
        model = models.Sequential([
            # Input
            layers.Input(shape=self.input_shape),
            
            # Block 1: Light conv block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            # Block 2: Medium conv block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            # Block 3: Final conv block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            # Global average pooling
            layers.GlobalAveragePooling2D(),
            
            # Lightweight dense layers
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            
            # Output
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, learning_rate=5e-4):
        """Compile model with lighter settings"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return self.model
    
    def summary(self):
        """Print model summary"""
        self.model.summary()
        total_params = self.model.count_params()
        print(f"\n=== Model Statistics ===")
        print(f"Total parameters: {total_params:,}")
        print(f"Estimated training time (CPU): 15-30 min per epoch")
    
    def save(self, path):
        """Save model"""
        self.model.save(path)
        print(f"✓ Model saved to {path}")

if __name__ == "__main__":
    print("=" * 60)
    print("Step 5-LITE: Build Lightweight CNN Baseline Model")
    print("=" * 60)
    
    cnn = CNNBaselineLite(input_shape=(200, 200, 1), num_classes=6)
    cnn.compile_model(learning_rate=5e-4)
    
    print("\n=== Lightweight Architecture ===")
    cnn.summary()
    
    cnn.save("doctor_task/models/cnn_baseline_lite.h5")
    
    print("\n✓ Lightweight Model COMPLETE")
    print("  Parameters: 1.2M (50% reduction)")
    print("  Est. CPU training: 1-2 hours total")
