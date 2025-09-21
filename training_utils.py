"""
Training utilities for waste classification
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from waste_classifier import MultiModelWasteClassifier


def analyze_dataset_balance(images_dir, label_mapping):
    """Analyze the balance between recyclable and non-recyclable items"""
    print("\nDataset Balance Analysis")
    print("=" * 40)

    recyclable_counts = {'default': 0, 'real_world': 0, 'total': 0}
    non_recyclable_counts = {'default': 0, 'real_world': 0, 'total': 0}

    for category in os.listdir(images_dir):
        category_path = os.path.join(images_dir, category)
        if not os.path.isdir(category_path) or category not in label_mapping:
            continue

        is_recyclable = label_mapping[category] == 1

        # Count default images
        default_path = os.path.join(category_path, 'default')
        if os.path.exists(default_path):
            default_count = len([f for f in os.listdir(default_path)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
            if is_recyclable:
                recyclable_counts['default'] += default_count
            else:
                non_recyclable_counts['default'] += default_count

        # Count real_world images
        real_world_path = os.path.join(category_path, 'real_world')
        if os.path.exists(real_world_path):
            real_world_count = len([f for f in os.listdir(real_world_path)
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
            if is_recyclable:
                recyclable_counts['real_world'] += real_world_count
            else:
                non_recyclable_counts['real_world'] += real_world_count

    # Calculate totals
    recyclable_counts['total'] = recyclable_counts['default'] + recyclable_counts['real_world']
    non_recyclable_counts['total'] = non_recyclable_counts['default'] + non_recyclable_counts['real_world']

    total_images = recyclable_counts['total'] + non_recyclable_counts['total']

    print(f"Recyclable Items:")
    print(f"   Default: {recyclable_counts['default']:,} images")
    print(f"   Real-world: {recyclable_counts['real_world']:,} images")
    print(f"   Total: {recyclable_counts['total']:,} images ({recyclable_counts['total']/total_images:.1%})")

    print(f"\nNon-recyclable Items:")
    print(f"   Default: {non_recyclable_counts['default']:,} images")
    print(f"   Real-world: {non_recyclable_counts['real_world']:,} images")
    print(f"   Total: {non_recyclable_counts['total']:,} images ({non_recyclable_counts['total']/total_images:.1%})")

    print(f"\nOverall Balance:")
    print(f"   Total Images: {total_images:,}")
    print(f"   Balance Ratio: {recyclable_counts['total']/non_recyclable_counts['total']:.2f}:1 (recyclable:non-recyclable)")

    return recyclable_counts, non_recyclable_counts


def create_sample_predictions_demo(classifier, images_dir, label_mapping, num_samples=5):
    """Create a demo showing predictions on sample images from each class"""
    if not classifier.best_model:
        print("No trained model available for demo")
        return

    print(f"\nSAMPLE PREDICTIONS DEMO")
    print("=" * 40)

    # Get sample images from each category
    recyclable_samples = []
    non_recyclable_samples = []

    for category in list(label_mapping.keys())[:10]:  # Limit to first 10 categories
        category_path = os.path.join(images_dir, category)
        if not os.path.isdir(category_path):
            continue

        # Try to get one sample from real_world first, then default
        sample_path = None
        sample_type = None

        real_world_path = os.path.join(category_path, 'real_world')
        if os.path.exists(real_world_path):
            images = [f for f in os.listdir(real_world_path)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            if images:
                sample_path = os.path.join(real_world_path, images[0])
                sample_type = 'real_world'

        if not sample_path:
            default_path = os.path.join(category_path, 'default')
            if os.path.exists(default_path):
                images = [f for f in os.listdir(default_path)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
                if images:
                    sample_path = os.path.join(default_path, images[0])
                    sample_type = 'default'

        if sample_path:
            sample_info = {
                'path': sample_path,
                'category': category,
                'type': sample_type,
                'true_label': label_mapping[category]
            }

            if label_mapping[category] == 1:
                recyclable_samples.append(sample_info)
            else:
                non_recyclable_samples.append(sample_info)

    print("Testing model on sample images...")

    all_samples = recyclable_samples[:num_samples] + non_recyclable_samples[:num_samples]

    correct_predictions = 0
    total_predictions = 0

    for sample in all_samples:
        try:
            result = classifier.predict_image(sample['path'])
            if result:
                predicted_recyclable = result['recyclable']
                true_recyclable = sample['true_label'] == 1
                is_correct = predicted_recyclable == true_recyclable

                if is_correct:
                    correct_predictions += 1
                    status = "Correct"
                else:
                    status = "Incorrect"

                total_predictions += 1

                print(f"\nSample: {sample['category']} ({sample['type']})")
                print(f"   True: {'Recyclable' if true_recyclable else 'Non-recyclable'}")
                print(f"   Predicted: {'Recyclable' if predicted_recyclable else 'Non-recyclable'}")
                print(f"   Confidence: {result['confidence']:.1%}")
                print(f"   Result: {status}")

        except Exception as e:
            print(f"Error predicting {sample['category']}: {e}")

    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"\nDemo Results: {correct_predictions}/{total_predictions} correct ({accuracy:.1%})")


def quick_train_best_models(classifier, train_gen, val_gen):
    """Quick training pipeline - trains ResNet50V2 only for modern TensorFlow"""
    print("\nQUICK TRAINING - ResNet50V2 Only")
    print("=" * 50)
    
    best_models = ['ResNet50V2']
    
    classifier.train_all_models(train_gen, val_gen, best_models)
    
    return classifier.best_model_name


def comprehensive_train_all_models(classifier, train_gen, val_gen):
    """Comprehensive training pipeline - trains ResNet50V2 only (modern TensorFlow optimized)"""
    print("\nCOMPREHENSIVE TRAINING - ResNet50V2 Only")
    print("=" * 50)
    
    all_models = list(classifier.available_models.keys())
    classifier.train_all_models(train_gen, val_gen, all_models)
    
    return classifier.best_model_name


def main_quick():
    """Quick training pipeline - trains ResNet50V2 only (modern TensorFlow)"""
    print("QUICK WASTE CLASSIFIER TRAINING - ResNet50V2 Only")
    print("=" * 50)

    classifier = MultiModelWasteClassifier()

    classifier.show_model_info()

    images_dir, categories = classifier.explore_dataset()
    if not images_dir:
        print("Could not find dataset structure")
        return None

    label_mapping = classifier.create_binary_labels(categories)
    analyze_dataset_balance(images_dir, label_mapping)

    train_gen, val_gen, temp_dir = classifier.prepare_data_generators(images_dir, label_mapping)

    best_model = quick_train_best_models(classifier, train_gen, val_gen)

    classifier.plot_training_histories()

    create_sample_predictions_demo(classifier, images_dir, label_mapping)

    classifier.create_deployment_package()

    classifier.export_for_mobile()

    import shutil
    shutil.rmtree(temp_dir)

    print(f"\nQuick training completed! Best model: {best_model}")
    return classifier


def main_comprehensive():
    """Comprehensive training pipeline - trains ResNet50V2 only (modern TensorFlow)"""
    print("COMPREHENSIVE WASTE CLASSIFIER TRAINING - ResNet50V2 Only")
    print("=" * 50)

    classifier = MultiModelWasteClassifier()

    images_dir, categories = classifier.explore_dataset()
    if not images_dir:
        print("Could not find dataset structure")
        return None
    
    label_mapping = classifier.create_binary_labels(categories)
    analyze_dataset_balance(images_dir, label_mapping)

    train_gen, val_gen, temp_dir = classifier.prepare_data_generators(images_dir, label_mapping)

    best_model = comprehensive_train_all_models(classifier, train_gen, val_gen)


    classifier.plot_training_histories()


    create_sample_predictions_demo(classifier, images_dir, label_mapping)


    classifier.create_deployment_package()


    classifier.export_for_mobile()


    import shutil
    shutil.rmtree(temp_dir)

    print(f"\nComprehensive training completed! Best model: {best_model}")
    return classifier


def main_custom(selected_models=None):
    """Custom training with ResNet50V2 only (modern TensorFlow optimized)"""
    print("CUSTOM WASTE CLASSIFIER TRAINING - ResNet50V2 Only")
    print("=" * 50)


    classifier = MultiModelWasteClassifier()
    classifier.show_model_info()


    if selected_models is None:
        available = list(classifier.available_models.keys())
        print(f"\nAvailable models: {', '.join(available)}")
        print("Only ResNet50V2 is available for modern TensorFlow training")
        selected_models = ['ResNet50V2']
    else:
        selected_models = ['ResNet50V2']
        print(f"Selected models: {selected_models} (ResNet50V2 only for modern TensorFlow)")

    images_dir, categories = classifier.explore_dataset()
    if not images_dir:
        return None

    label_mapping = classifier.create_binary_labels(categories)
    analyze_dataset_balance(images_dir, label_mapping)

    train_gen, val_gen, temp_dir = classifier.prepare_data_generators(images_dir, label_mapping)

    classifier.train_all_models(train_gen, val_gen, selected_models)

    classifier.plot_training_histories()
    create_sample_predictions_demo(classifier, images_dir, label_mapping)
    classifier.create_deployment_package()
    classifier.export_for_mobile()

    import shutil
    shutil.rmtree(temp_dir)

    return classifier
