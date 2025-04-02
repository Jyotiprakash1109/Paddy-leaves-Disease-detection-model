import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import json
import shutil
from pathlib import Path

class PaddyDiseaseClassifierTF:
    def __init__(self, img_size=(224, 224), batch_size=32, epochs=20):
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.class_indices = None
        self.history = None
    
    def build_model(self, num_classes):
        """Build a CNN model for image classification"""
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Fourth Convolutional Block
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Flatten and Dense Layers
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, train_dir):
        """Train the CNN model using data augmentation"""
        # Data augmentation for training set
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Load training data
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        # Store class indices for prediction
        self.class_indices = {v: k for k, v in train_generator.class_indices.items()}
        
        # Calculate number of classes
        num_classes = len(train_generator.class_indices)
        print(f"Number of classes: {num_classes}")
        print(f"Class mapping: {self.class_indices}")
        
        # Build the model
        self.model = self.build_model(num_classes)
        
        # Print model summary
        self.model.summary()
        
        # Callbacks for training
        callbacks = [
            EarlyStopping(monitor='loss', patience=7, restore_best_weights=True),
            ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=1e-6),
            ModelCheckpoint('paddy_disease_best_model.keras', save_best_only=True, monitor='accuracy')
        ]
        
        # Train the model
        print("\nTraining CNN model...")
        self.history = self.model.fit(
            train_generator,
            epochs=self.epochs,
            callbacks=callbacks
        )
        
        return {
            'accuracy': self.history.history['accuracy'][-1],
            'loss': self.history.history['loss'][-1]
        }
    
    def plot_training_history(self):
        """Plot training accuracy/loss curves"""
        if self.history is None:
            print("Model hasn't been trained yet.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'])
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train'], loc='upper left')
        
        # Loss plot
        ax2.plot(self.history.history['loss'])
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
    
    def plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix as a heatmap"""
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()
    
    def predict(self, image_path):
        """Predict disease for a single image"""
        if self.model is None:
            raise ValueError("Model is not trained or loaded yet")
        
        if self.class_indices is None:
            raise ValueError("Class indices are not available")
            
        # Print class indices for debugging
        print("Available class indices:", self.class_indices)
        
        # Load and preprocess the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        prediction = self.model.predict(img)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        probability = np.max(prediction) * 100
        
        print(f"Predicted class index: {predicted_class_index}")
        
        # Get the class label from the index
        try:
            predicted_class = self.class_indices[predicted_class_index]
        except KeyError:
            # If class index is not found, display a list of available classes
            print(f"KeyError: Class index {predicted_class_index} not found in class_indices")
            
            # If the indices are string keys instead of integers, try converting
            if all(isinstance(k, str) for k in self.class_indices.keys()):
                predicted_class = self.class_indices.get(str(predicted_class_index), f"Unknown-{predicted_class_index}")
            # If indices are integers but in wrong order
            elif isinstance(list(self.class_indices.keys())[0], int):
                # Get class names sorted by their indices
                class_list = [name for idx, name in sorted(self.class_indices.items())]
                if 0 <= predicted_class_index < len(class_list):
                    predicted_class = class_list[predicted_class_index]
                else:
                    predicted_class = f"Unknown-{predicted_class_index}"
            # Last resort: use first class if indices are messed up
            else:
                # Find closest index as fallback
                closest_idx = min(self.class_indices.keys(), key=lambda x: abs(x - predicted_class_index))
                predicted_class = f"{self.class_indices[closest_idx]} (using closest index)"
        
        return predicted_class, probability
    
    def save_model(self, model_dir):
        """Save the trained model and metadata"""
        if self.model is None:
            raise ValueError("Model is not trained yet")
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model architecture and weights
        model_path = os.path.join(model_dir, 'paddy_disease_model.keras')
        self.model.save(model_path)
        
        # Save metadata
        metadata = {
            'img_size': self.img_size,
            'class_indices': self.class_indices,
            'batch_size': self.batch_size,
            'epochs': self.epochs
        }
        
        metadata_path = os.path.join(model_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        print(f"Model saved to {model_dir}")
    
    @classmethod
    def load_model(cls, model_dir):
        """Load a trained model and metadata"""
        # Load metadata
        metadata_path = os.path.join(model_dir, 'model_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create instance with loaded parameters
        instance = cls(
            img_size=tuple(metadata['img_size']),
            batch_size=metadata['batch_size'],
            epochs=metadata['epochs']
        )
        
        # Load class indices
        # Convert string keys back to integers if needed
        class_indices = metadata['class_indices']
        if all(k.isdigit() for k in class_indices.keys()):
            instance.class_indices = {int(k): v for k, v in class_indices.items()}
        else:
            instance.class_indices = class_indices
        
        print(f"Loaded class indices: {instance.class_indices}")
        
        # Load model
        model_path = os.path.join(model_dir, 'paddy_disease_model.keras')
        instance.model = load_model(model_path)
        
        print(f"Model loaded from {model_dir}")
        return instance
    
    def evaluate_test_data(self, test_dir):
        """Evaluate model on a test dataset"""
        if self.model is None:
            raise ValueError("Model is not trained or loaded yet")
        
        # Create test data generator
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Evaluate the model
        print("Evaluating model on test data...")
        test_loss, test_acc = self.model.evaluate(test_generator)
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        
        # Get predictions
        test_generator.reset()
        y_true = test_generator.classes
        y_pred_prob = self.model.predict(test_generator)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Calculate classification report
        print("\nClassification Report:")
        class_names = list(test_generator.class_indices.keys())
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm, class_names)
        
        return {
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'confusion_matrix': cm.tolist()
        }

# Usage example
if __name__ == "__main__":
    # Define paths
    data_train_path = '/Users/jyotiprakash/Documents/PaddyLeaves/Paddyleaves_Train'
    data_test_path = '/Users/jyotiprakash/Documents/PaddyLeaves/Paddyleaves_Test'
    model_dir = 'PaddyLeaves/tf_model'
    
    # Create and train model or load existing one
    if os.path.exists(os.path.join(model_dir, 'paddy_disease_model.keras')):
        print("Loading existing model...")
        classifier = PaddyDiseaseClassifierTF.load_model(model_dir)
    else:
        print("Training new model...")
        classifier = PaddyDiseaseClassifierTF()
        
        # Train the model
        classifier.train(data_train_path)
        
        # Plot training history
        classifier.plot_training_history()
        
        # Save the model
        classifier.save_model(model_dir)
    
    # Evaluate on test data if available
    if os.path.exists(data_test_path):
        test_metrics = classifier.evaluate_test_data(data_test_path)
    
    # Make prediction on a single image
    test_image = '/Users/jyotiprakash/Documents/PaddyLeaves/Paddyleaves_Test/Brownspot/brown-leaf-spot-infestation-on-260nw-2523213965.jpg'
    if os.path.exists(test_image):
        disease, confidence = classifier.predict(test_image)
        print(f'\nPrediction for test image:')
        print(f'Disease: {disease}')
        print(f'Confidence: {confidence:.2f}%')
        
        # Display the image with prediction
        img = cv2.imread(test_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f'Prediction: {disease} ({confidence:.2f}%)')
        plt.axis('off')
        plt.savefig('prediction_result.png')
        plt.show()