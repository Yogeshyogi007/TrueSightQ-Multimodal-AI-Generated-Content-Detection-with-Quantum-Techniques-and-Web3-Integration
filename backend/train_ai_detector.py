"""
AI Image Detector Training Script
Follows Isgen's approach for training a deep learning model to detect AI-generated images
"""

import tensorflow as tf
import keras
from keras import layers, Model, optimizers, callbacks, applications
import numpy as np
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json

class AIImageDetectorTrainer:
    def __init__(self, data_dir="dataset", model_save_path="ai_detector_weights.h5"):
        self.data_dir = data_dir
        self.model_save_path = model_save_path
        self.img_size = (224, 224)
        self.batch_size = 32
        self.epochs = 50
        self.learning_rate = 0.001
        
        # AI model classes (similar to Isgen's approach)
        self.ai_models = [
            'dalle', 'midjourney', 'stable_diffusion', 'flux', 
            'adobe_firefly', 'gpt4o', 'recraft', 'bing_creator', 
            'ideogram', 'reve'
        ]
        
    def build_model(self):
        """Build the deep learning model similar to Isgen's architecture"""
        # Use EfficientNetB0 as base (what Isgen likely uses)
        base_model = applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3)
        )
        
        # Freeze the base model initially
        base_model.trainable = False
        
        # Create input layer
        inputs = layers.Input(shape=(224, 224, 3))
        
        # Pass through base model
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Main classification: Real vs AI-Generated
        ai_probability = layers.Dense(1, activation='sigmoid', name='ai_probability')(x)
        
        # Model classification: Which AI model (if AI-generated)
        model_classification = layers.Dense(len(self.ai_models), activation='softmax', name='model_classification')(x)
        
        # Create the model
        model = Model(inputs, [ai_probability, model_classification])
        
        # Compile the model
        model.compile(
            optimizer='adam',  # Use string identifier instead of class instance
            loss={
                'ai_probability': 'binary_crossentropy',
                'model_classification': 'sparse_categorical_crossentropy'
            },
            metrics={
                'ai_probability': ['accuracy'],
                'model_classification': ['accuracy']
            }
        )
        
        return model
    
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        images = []
        ai_labels = []  # 0 for real, 1 for AI-generated
        model_labels = []  # AI model classification
        
        # Load real images
        real_dir = os.path.join(self.data_dir, "real")
        if os.path.exists(real_dir):
            for img_name in os.listdir(real_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(real_dir, img_name)
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize(self.img_size)
                        img_array = np.array(img) / 255.0
                        
                        images.append(img_array)
                        ai_labels.append(0)  # Real
                        model_labels.append(0)  # Use 0 instead of -1 for real images
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
        
        # Load AI-generated images
        for i, model_name in enumerate(self.ai_models):
            model_dir = os.path.join(self.data_dir, "ai", model_name)
            if os.path.exists(model_dir):
                for img_name in os.listdir(model_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(model_dir, img_name)
                        try:
                            img = Image.open(img_path).convert('RGB')
                            img = img.resize(self.img_size)
                            img_array = np.array(img) / 255.0
                            
                            images.append(img_array)
                            ai_labels.append(1)  # AI-generated
                            model_labels.append(i + 1)  # Model index + 1 (0 reserved for real)
                        except Exception as e:
                            print(f"Error loading {img_path}: {e}")
        
        return np.array(images), np.array(ai_labels), np.array(model_labels)
    
    def create_data_generators(self, images, ai_labels, model_labels):
        """Create training and validation datasets"""
        # Split the data
        X_train, X_test, y_ai_train, y_ai_test, y_model_train, y_model_test = train_test_split(
            images, ai_labels, model_labels, test_size=0.2, random_state=42, stratify=ai_labels
        )
        
        # Create TensorFlow datasets
        def augment_image(image):
            # Random horizontal flip
            image = tf.image.random_flip_left_right(image)
            # Random brightness
            image = tf.image.random_brightness(image, max_delta=0.2)
            # Random contrast
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
            # Random rotation (small angle)
            image = tf.image.rot90(image, k=tf.random.uniform([], maxval=4, dtype=tf.int32))
            return image
        
        # Create training dataset with augmentation
        train_dataset = tf.data.Dataset.from_tensor_slices((
            X_train,
            {'ai_probability': y_ai_train, 'model_classification': y_model_train}
        ))
        train_dataset = train_dataset.map(
            lambda x, y: (augment_image(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        train_dataset = train_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Create validation dataset (no augmentation)
        val_dataset = tf.data.Dataset.from_tensor_slices((
            X_test,
            {'ai_probability': y_ai_test, 'model_classification': y_model_test}
        ))
        val_dataset = val_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset, X_test, y_ai_test, y_model_test
    
    def train_model(self):
        """Train the AI image detector model"""
        print("Loading and preprocessing data...")
        images, ai_labels, model_labels = self.load_and_preprocess_data()
        
        if len(images) == 0:
            print("No images found! Please check your dataset directory structure.")
            return None, None
        
        print(f"Loaded {len(images)} images")
        print(f"Real images: {np.sum(ai_labels == 0)}")
        print(f"AI-generated images: {np.sum(ai_labels == 1)}")
        
        # Create data generators
        train_dataset, val_dataset, X_test, y_ai_test, y_model_test = self.create_data_generators(
            images, ai_labels, model_labels
        )
        
        # Build model
        print("Building model...")
        model = self.build_model()
        model.summary()
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            callbacks.ModelCheckpoint(
                self.model_save_path,
                monitor='val_ai_probability_accuracy',
                save_best_only=True,
                save_weights_only=True
            )
        ]
        
        # Train the model
        print("Training model...")
        history = model.fit(
            train_dataset,
            epochs=self.epochs,
            validation_data=val_dataset,
            callbacks=callbacks_list,
            verbose='auto'  # Use string instead of integer
        )
        
        # Evaluate the model
        print("Evaluating model...")
        self.evaluate_model(model, X_test, y_ai_test, y_model_test)
        
        # Plot training history
        self.plot_training_history(history)
        
        print(f"Model saved to {self.model_save_path}")
        
        return model, history
    
    def evaluate_model(self, model, X_test, y_ai_test, y_model_test):
        """Evaluate the trained model"""
        # Make predictions
        predictions = model.predict(X_test)
        ai_pred = predictions[0].flatten()
        model_pred = predictions[1]
        
        # Convert predictions to binary
        ai_pred_binary = (ai_pred > 0.5).astype(int)
        
        # Print classification report for AI detection
        print("\nAI Detection Results:")
        print(classification_report(y_ai_test, ai_pred_binary, target_names=['Real', 'AI-Generated']))
        
        # Print confusion matrix
        cm = confusion_matrix(y_ai_test, ai_pred_binary)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Calculate accuracy
        accuracy = np.mean(ai_pred_binary == y_ai_test)
        print(f"\nOverall Accuracy: {accuracy:.4f}")
        
        # Model classification accuracy (only for AI-generated images)
        ai_indices = y_ai_test == 1
        if np.sum(ai_indices) > 0:
            model_pred_classes = np.argmax(model_pred[ai_indices], axis=1)
            # Adjust for the +1 offset we used during data loading
            model_accuracy = np.mean((model_pred_classes + 1) == y_model_test[ai_indices])
            print(f"Model Classification Accuracy: {model_accuracy:.4f}")
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # AI probability accuracy
        axes[0, 0].plot(history.history['ai_probability_accuracy'])
        axes[0, 0].plot(history.history['val_ai_probability_accuracy'])
        axes[0, 0].set_title('AI Detection Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend(['Train', 'Validation'])
        axes[0, 0].grid(True)
        
        # AI probability loss
        axes[0, 1].plot(history.history['ai_probability_loss'])
        axes[0, 1].plot(history.history['val_ai_probability_loss'])
        axes[0, 1].set_title('AI Detection Loss')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend(['Train', 'Validation'])
        axes[0, 1].grid(True)
        
        # Model classification accuracy
        axes[1, 0].plot(history.history['model_classification_accuracy'])
        axes[1, 0].plot(history.history['val_model_classification_accuracy'])
        axes[1, 0].set_title('Model Classification Accuracy')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend(['Train', 'Validation'])
        axes[1, 0].grid(True)
        
        # Model classification loss
        axes[1, 1].plot(history.history['model_classification_loss'])
        axes[1, 1].plot(history.history['val_model_classification_loss'])
        axes[1, 1].set_title('Model Classification Loss')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend(['Train', 'Validation'])
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_dataset_structure(self):
        """Create the recommended dataset directory structure"""
        structure = {
            "dataset": {
                "real": "Place real human photos here",
                "ai": {
                    "dalle": "DALL-E generated images",
                    "midjourney": "Midjourney generated images", 
                    "stable_diffusion": "Stable Diffusion generated images",
                    "flux": "Flux generated images",
                    "adobe_firefly": "Adobe Firefly generated images",
                    "gpt4o": "GPT-4o generated images",
                    "recraft": "Recraft generated images",
                    "bing_creator": "Bing Image Creator generated images",
                    "ideogram": "Ideogram generated images",
                    "reve": "Reve generated images"
                }
            }
        }
        
        # Create directories
        flattened_structure = self._flatten_structure(structure)
        if flattened_structure:
            for path, description in flattened_structure:
                os.makedirs(path, exist_ok=True)
                print(f"Created: {path} - {description}")
        else:
            print("Warning: Could not flatten structure")
        
        # Save structure info
        with open('dataset_structure.json', 'w') as f:
            json.dump(structure, f, indent=2)
        
        print("\nDataset structure created! Please add your images to the appropriate folders.")
    
    def _flatten_structure(self, structure, prefix=""):
        """Flatten the nested structure for directory creation"""
        result = []
        for key, value in structure.items():
            path = os.path.join(prefix, key) if prefix else key
            if isinstance(value, dict):
                result.append((path, "Directory"))
                recursive_result = self._flatten_structure(value, path)
                if recursive_result:
                    result.extend(recursive_result)
            else:
                result.append((path, value))
        return result

def main():
    """Main training function"""
    trainer = AIImageDetectorTrainer()
    
    # Check if dataset exists
    if not os.path.exists("dataset"):
        print("Dataset not found. Creating recommended structure...")
        trainer.create_dataset_structure()
        print("\nPlease add your images to the dataset folders and run this script again.")
        return
    
    # Train the model
    try:
        model, history = trainer.train_model()
        if model is None or history is None:
            print("Training could not proceed. Please check your dataset.")
            return
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed: {e}")
        return

if __name__ == "__main__":
    main()