"""
Waste Classification System - Modular Implementation
Optimized for Google Colab with GPU support - FIXED VERSION
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import kagglehub
from PIL import Image
import cv2
import warnings
import json
import zipfile
from datetime import datetime
import tempfile
import shutil

warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)


class CustomModelCheckpoint(keras.callbacks.Callback):
   
    
    def __init__(self, filepath, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.verbose = verbose
        self.best_value = -np.inf if mode == 'max' else np.inf
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        current_value = logs.get(self.monitor, None)
        if current_value is None:
            if self.verbose:
                print(f"Warning: {self.monitor} not found in logs")
            return
        
        if self.save_best_only:
            if self.mode == 'max' and current_value > self.best_value:
                self.best_value = current_value
                self._save_model(epoch, current_value)
            elif self.mode == 'min' and current_value < self.best_value:
                self.best_value = current_value
                self._save_model(epoch, current_value)
        else:
            self._save_model(epoch, current_value)
    
    def _save_model(self, epoch, value):
        try:
            self.model.save(self.filepath)
            if self.verbose:
                print(f"Model saved to {self.filepath} (epoch {epoch+1}, {self.monitor}: {value:.4f})")
        except Exception as e:
            print(f"Error saving model: {e}")


class MultiModelWasteClassifier:

    
    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path
        self.models = {}
        self.histories = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.img_size = (224, 224)
        self.batch_size = 32
        self.epochs = 15

        self.in_colab = self._check_colab()
        if self.in_colab:
            print("Running in Google Colab - GPU acceleration enabled!")
            self._setup_colab()

        self.available_models = {
            'ResNet50V2': {
                'model': ResNet50V2,
                'input_shape': (224, 224, 3),
                'description': 'Strong performance, moderate size - Modern TensorFlow optimized'
            }
        }

    def _check_colab(self):
        """Check if running in Google Colab"""
        try:
            import google.colab
            return True
        except ImportError:
            return False

    def _setup_colab(self):
        """Setup Google Colab environment"""
        if self.in_colab:
            from google.colab import drive
            drive.mount('/content/drive')

            self.save_dir = '/content/drive/MyDrive/WasteClassifierModels'
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"Models will be saved to: {self.save_dir}")

            gpu_info = tf.config.list_physical_devices('GPU')
            if gpu_info:
                print(f"GPU detected: {len(gpu_info)} GPU(s) available")
                for gpu in gpu_info:
                    tf.config.experimental.set_memory_growth(gpu, True)
            else:
                print("No GPU detected, using CPU")
        else:
            self.save_dir = './models'
            os.makedirs(self.save_dir, exist_ok=True)

    def download_dataset(self):
        """Download the dataset from Kaggle"""
        print("Downloading dataset...")
        if self.in_colab:
            os.environ['KAGGLE_CONFIG_DIR'] = '/content'
            print("Please upload your kaggle.json file to authenticate")

        path = kagglehub.dataset_download("alistairking/recyclable-and-household-waste-classification")
        self.dataset_path = path
        print(f"Dataset downloaded to: {path}")
        return path

    def explore_dataset(self, data_dir=None):
        """Explore and analyze the dataset structure"""
        if data_dir is None:
            if self.dataset_path:
                data_dir = self.dataset_path
            else:
                possible_paths = [
                    '/content/recyclable-and-household-waste-classification',
                    './recyclable-and-household-waste-classification',
                    './data'
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        data_dir = path
                        break
                
                if not data_dir:
                    print("Dataset not found. Please download it first.")
                    return None, None

        if not os.path.exists(data_dir):
            print(f"Dataset directory not found: {data_dir}")
            return None, None

        print(f"Exploring dataset at: {data_dir}")
        
        categories = []
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):

                has_default = os.path.exists(os.path.join(item_path, 'default'))
                has_real_world = os.path.exists(os.path.join(item_path, 'real_world'))
                if has_default or has_real_world:
                    categories.append(item)

        if not categories:
            print("No valid categories found with expected structure")
            return None, None

        print(f"Found {len(categories)} categories:")

        category_counts = {}
        total_images = 0
        for category in categories:
            category_path = os.path.join(data_dir, category)
            count = 0
            
            default_path = os.path.join(category_path, 'default')
            if os.path.exists(default_path):
                count += len([f for f in os.listdir(default_path)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
            
            real_world_path = os.path.join(category_path, 'real_world')
            if os.path.exists(real_world_path):
                count += len([f for f in os.listdir(real_world_path)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
            
            category_counts[category] = count
            total_images += count
            print(f"   {category}: {count:,} images")

        print(f"\nTotal Images: {total_images:,}")

        plt.figure(figsize=(14, 8))
        bars = plt.bar(category_counts.keys(), category_counts.values(),
                      color=plt.cm.Set3(np.linspace(0, 1, len(category_counts))))
        plt.title('Dataset Distribution by Category', fontsize=16, fontweight='bold')
        plt.xlabel('Categories', fontsize=12)
        plt.ylabel('Number of Images', fontsize=12)
        plt.xticks(rotation=45, ha='right')

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

        return data_dir, categories

    def show_model_info(self):
        """Display information about available models"""
        print("\nAvailable Models:")
        print("=" * 50)
        for name, info in self.available_models.items():
            print(f"{name}")
            print(f"   Input Shape: {info['input_shape']}")
            print(f"   Description: {info['description']}")
            print()

    def create_binary_labels(self, categories):
        """Create binary labels mapping based on waste categories"""
        label_mapping = {}

        recyclable_keywords = [
            'plastic', 'paper', 'cardboard', 'glass', 'metal', 'aluminum', 'steel',
            'bottle', 'can', 'container', 'packaging', 'magazine', 'newspaper'
        ]

        non_recyclable_keywords = [
            'organic', 'food waste', 'eggshell', 'coffee grounds', 'tea bag',
            'textile', 'clothing', 'shoe', 'straw', 'cutlery', 'disposable'
        ]

        print("\nAutomatic Binary Classification:")
        print("-" * 50)

        recyclable_count = 0
        non_recyclable_count = 0
        ambiguous_categories = []

        for category in sorted(categories):
            category_lower = category.lower()

            is_recyclable = any(keyword in category_lower for keyword in recyclable_keywords)
            is_non_recyclable = any(keyword in category_lower for keyword in non_recyclable_keywords)

            if is_recyclable and not is_non_recyclable:
                label_mapping[category] = 1
                status = "Recyclable"
                recyclable_count += 1
            elif is_non_recyclable and not is_recyclable:
                label_mapping[category] = 0 
                status = "Non-recyclable"
                non_recyclable_count += 1
            else:
                ambiguous_categories.append(category)
                continue

            print(f"   {category}: {status}")

        if ambiguous_categories:
            print(f"\nFound {len(ambiguous_categories)} ambiguous categories:")
            print("Using heuristics for classification:")

            for category in ambiguous_categories:
                category_lower = category.lower()

                if any(word in category_lower for word in ['bag', 'lid', 'straw']):

                    label_mapping[category] = 0
                    status = "Non-recyclable (heuristic)"
                elif 'textile' in category_lower or 'clothing' in category_lower:

                    label_mapping[category] = 0
                    status = "Non-recyclable (textile)"
                else:

                    label_mapping[category] = 1
                    status = "Recyclable (default)"

                print(f"   {category}: {status}")

                if label_mapping[category] == 1:
                    recyclable_count += 1
                else:
                    non_recyclable_count += 1

        print(f"\nFinal Classification Summary:")
        print(f"   Recyclable categories: {recyclable_count}")
        print(f"   Non-recyclable categories: {non_recyclable_count}")
        print(f"   Total categories: {len(categories)}")

        return label_mapping

    def prepare_data_generators(self, images_dir, label_mapping):
        """Prepare data generators for training with the hierarchical structure"""
        print("Preparing data generators...")
        print("Processing hierarchical dataset structure (default + real_world)")

        temp_dir = tempfile.mkdtemp()
        train_dir = os.path.join(temp_dir, 'train')
        val_dir = os.path.join(temp_dir, 'validation')

        for split in ['train', 'validation']:
            for class_name in ['recyclable', 'non_recyclable']:
                os.makedirs(os.path.join(temp_dir, split, class_name), exist_ok=True)

        print("Organizing data for binary classification...")
        total_train = 0
        total_val = 0

        recyclable_train = 0
        recyclable_val = 0
        non_recyclable_train = 0
        non_recyclable_val = 0

        for category in os.listdir(images_dir):
            category_path = os.path.join(images_dir, category)
            if not os.path.isdir(category_path) or category not in label_mapping:
                continue

            binary_label = 'recyclable' if label_mapping[category] == 1 else 'non_recyclable'

            all_images = []

            default_path = os.path.join(category_path, 'default')
            if os.path.exists(default_path):
                default_images = [('default', f) for f in os.listdir(default_path)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
                all_images.extend(default_images)

            real_world_path = os.path.join(category_path, 'real_world')
            if os.path.exists(real_world_path):
                real_world_images = [('real_world', f) for f in os.listdir(real_world_path)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
                all_images.extend(real_world_images)

            if len(all_images) == 0:
                print(f"No images found for category: {category}")
                continue

            print(f"   {category} ({binary_label}): {len(all_images)} images")

            train_images, val_images = train_test_split(
                all_images, test_size=0.2, random_state=42, shuffle=True
            )

            for img_type, img_name in train_images:
                src_path = os.path.join(category_path, img_type, img_name)
                dst_name = f"{category}_{img_type}_{img_name}"
                dst_path = os.path.join(train_dir, binary_label, dst_name)

                try:
                    shutil.copy2(src_path, dst_path)
                    total_train += 1
                    if binary_label == 'recyclable':
                        recyclable_train += 1
                    else:
                        non_recyclable_train += 1
                except Exception as e:
                    print(f"Error copying {src_path}: {e}")

            for img_type, img_name in val_images:
                src_path = os.path.join(category_path, img_type, img_name)
                dst_name = f"{category}_{img_type}_{img_name}"
                dst_path = os.path.join(val_dir, binary_label, dst_name)

                try:
                    shutil.copy2(src_path, dst_path)
                    total_val += 1
                    if binary_label == 'recyclable':
                        recyclable_val += 1
                    else:
                        non_recyclable_val += 1
                except Exception as e:
                    print(f"Error copying {src_path}: {e}")

        print(f"\nData organization completed:")
        print(f"   Training set: {total_train:,} images")
        print(f"      Recyclable: {recyclable_train:,}")
        print(f"      Non-recyclable: {non_recyclable_train:,}")
        print(f"   Validation set: {total_val:,} images")
        print(f"      Recyclable: {recyclable_val:,}")
        print(f"      Non-recyclable: {non_recyclable_val:,}")

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )

        val_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=True
        )

        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )

        print(f"\nData generators created:")
        print(f"   Training batches: {len(train_generator)}")
        print(f"   Validation batches: {len(val_generator)}")

        return train_generator, val_generator, temp_dir

    def create_model(self, model_name):
        """Create a specific model architecture"""
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available")

        model_info = self.available_models[model_name]
        base_model = model_info['model'](
            weights='imagenet',
            include_top=False,
            input_shape=model_info['input_shape'],
            pooling='avg' 
        )


        base_model.trainable = False

        model = keras.Sequential([
            base_model,
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def debug_model_creation(self):
        """Debug model creation"""
        try:
            model = self.create_model('ResNet50V2')
            print("Model creation successful")
            print(model.summary())
            return model
        except Exception as e:
            print(f"Model creation failed: {e}")
            return None

    def load_model_safely(self, model_path, model_name='ResNet50V2'):
        """Load a saved model by recreating architecture first"""
        try:
            model = self.create_model(model_name)
            
            try:
                loaded_model = keras.models.load_model(model_path)
                print(f"Successfully loaded full model from {model_path}")
                return loaded_model
            except Exception as e:
                print(f"Full model loading failed: {e}")
                print("Attempting to load weights only...")

                weights_path = model_path.replace('.keras', '_weights.h5')
                if os.path.exists(weights_path):
                    model.load_weights(weights_path)
                    print(f"Successfully loaded weights from {weights_path}")
                    return model
                else:
                    print(f"Weights file not found: {weights_path}")
                    return None
                    
        except Exception as e:
            print(f"Model loading failed: {e}")
            return None

    def create_deployment_package(self):
        """Create a deployment package with the best model"""
        if not self.best_model:
            print("No trained model available for deployment")
            return None
            
        try:

            deployment_dir = os.path.join(self.save_dir, 'deployment')
            os.makedirs(deployment_dir, exist_ok=True)
            

            model_path = os.path.join(deployment_dir, 'best_model.keras')
            self.best_model.save(model_path)

            config = {
                'model_name': self.best_model_name,
                'input_shape': self.img_size,
                'classes': ['non_recyclable', 'recyclable'],
                'created_at': datetime.now().isoformat()
            }
            
            config_path = os.path.join(deployment_dir, 'model_config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"Deployment package created at: {deployment_dir}")
            print(f"Model: {model_path}")
            print(f"Config: {config_path}")
            
            return deployment_dir
            
        except Exception as e:
            print(f"Failed to create deployment package: {e}")
            return None

    def train_model(self, model_name, train_gen, val_gen):
        """Train a specific model"""
        print(f"\nTraining {model_name}...")
        print("=" * 40)


        model = self.create_model(model_name)
        

        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        

        try:
            if os.path.exists(self.save_dir) and os.access(self.save_dir, os.W_OK):

                checkpoint_callback = CustomModelCheckpoint(
                    filepath=os.path.join(self.save_dir, f'{model_name}_best.keras'),
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max',
                    verbose=1
                )
                callbacks.append(checkpoint_callback)
                print(f"Model checkpoints will be saved to: {self.save_dir}")
            else:
                print("Warning: Cannot save model checkpoints - directory not writable")
        except Exception as e:
            print(f"Warning: Custom checkpoint disabled due to error: {e}")
            print("Training will continue without saving checkpoints")

        try:
            history = model.fit(
                train_gen,
                epochs=self.epochs,
                validation_data=val_gen,
                callbacks=callbacks,
                verbose=1
            )
        except Exception as e:
            print(f"Error during training with callbacks: {e}")
            print("Retrying with minimal callbacks...")

            minimal_callbacks = [
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=5,
                    restore_best_weights=True,
                    verbose=1
                )
            ]
            
            history = model.fit(
                train_gen,
                epochs=self.epochs,
                validation_data=val_gen,
                callbacks=minimal_callbacks,
                verbose=1
            )


        self.models[model_name] = model
        self.histories[model_name] = history.history


        val_loss, val_accuracy = model.evaluate(val_gen, verbose=0)
        self.results[model_name] = {
            'val_accuracy': val_accuracy,
            'val_loss': val_loss
        }

        print(f"{model_name} training completed!")
        print(f"   Validation Accuracy: {val_accuracy:.4f}")
        print(f"   Validation Loss: {val_loss:.4f}")

    def simple_train_model(self, model_name, train_gen, val_gen):
        """Simple training method without problematic callbacks"""
        print(f"\nSimple training {model_name}...")
        print("=" * 40)


        model = self.create_model(model_name)
        
        print("Training without callbacks to avoid compatibility issues...")
        history = model.fit(
            train_gen,
            epochs=self.epochs,
            validation_data=val_gen,
            verbose=1
        )

        self.models[model_name] = model
        self.histories[model_name] = history.history

        val_loss, val_accuracy = model.evaluate(val_gen, verbose=0)
        self.results[model_name] = {
            'val_accuracy': val_accuracy,
            'val_loss': val_loss
        }

        try:
            model_save_path = os.path.join(self.save_dir, f'{model_name}_final.keras')
            model.save(model_save_path)
            print(f"Model saved to: {model_save_path}")
        except Exception as e:
            print(f"Warning: Could not save model: {e}")

        print(f"{model_name} training completed!")
        print(f"   Validation Accuracy: {val_accuracy:.4f}")
        print(f"   Validation Loss: {val_loss:.4f}")

        return model, history

    def train_all_models(self, train_gen, val_gen, selected_models=None):
        """Train all or selected models"""
        if selected_models is None:
            selected_models = list(self.available_models.keys())

        print(f"\nTraining {len(selected_models)} models...")
        print("=" * 50)

        for model_name in selected_models:
            try:
                # Try regular training first
                self.train_model(model_name, train_gen, val_gen)
            except Exception as e:
                print(f"Regular training failed for {model_name}: {e}")
                print("Attempting simple training method...")
                try:
                    # Try simple training as backup
                    self.simple_train_model(model_name, train_gen, val_gen)
                except Exception as e2:
                    print(f"Simple training also failed for {model_name}: {e2}")
                    continue

        # Find best model
        if self.results:
            best_model_name = max(self.results.keys(), 
                                key=lambda x: self.results[x]['val_accuracy'])
            self.best_model = self.models[best_model_name]
            self.best_model_name = best_model_name
            
            print(f"\nBest Model: {best_model_name}")
            print(f"   Accuracy: {self.results[best_model_name]['val_accuracy']:.4f}")

        return self.best_model_name

    def debug_tensorflow_version(self):
        """Debug TensorFlow version and available callbacks"""
        print("\n🔧 TENSORFLOW DEBUG INFO")
        print("-" * 30)
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Keras version: {tf.keras.__version__}")
        
        try:
            test_checkpoint = ModelCheckpoint(
                filepath='test_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=0
            )
            print("ModelCheckpoint works with basic parameters")
            
            try:
                test_checkpoint2 = ModelCheckpoint(
                    filepath='test_model2.keras',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=0,
                    save_weights_only=False
                )
                print("ModelCheckpoint works with extended parameters")
            except Exception as e:
                print(f"ModelCheckpoint fails with extended parameters: {e}")
                
        except Exception as e:
            print(f"ModelCheckpoint fails with basic parameters: {e}")
        

        print(f"Available optimizers: {dir(tf.keras.optimizers)}")
        
        return True

    def plot_training_histories(self):
        """Plot training histories for all models"""
        if not self.histories:
            print("No training histories to plot")
            return

        n_models = len(self.histories)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for i, (model_name, history) in enumerate(self.histories.items()):
            if i >= 4:
                break
                
            ax = axes[i]
            
            ax.plot(history['accuracy'], label='Training Accuracy', color='blue')
            ax.plot(history['val_accuracy'], label='Validation Accuracy', color='red')
            ax.set_title(f'{model_name} - Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend()
            ax.grid(True)

        for i in range(len(self.histories), 4):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for i, (model_name, history) in enumerate(self.histories.items()):
            if i >= 4:
                break
                
            ax = axes[i]
            
            ax.plot(history['loss'], label='Training Loss', color='blue')
            ax.plot(history['val_loss'], label='Validation Loss', color='red')
            ax.set_title(f'{model_name} - Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True)

        for i in range(len(self.histories), 4):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()

    def predict_image(self, image_path):
        """Predict if an image is recyclable or not"""
        if not self.best_model:
            print("No trained model available")
            return None

        try:
            img = Image.open(image_path)
            img = img.convert('RGB')
            img = img.resize(self.img_size)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            prediction = self.best_model.predict(img_array, verbose=0)
            confidence = float(prediction[0][0])
            is_recyclable = confidence > 0.5

            return {
                'recyclable': is_recyclable,
                'confidence': confidence,
                'prediction': 'Recyclable' if is_recyclable else 'Non-recyclable'
            }

        except Exception as e:
            print(f"Error predicting image: {e}")
            return None