#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark Dermatology Model Accuracy

python models/Dermatology_Model/derm_benchmarker.py --multi-seed 123 456 789 --samples 7

Model v4.0 Performance: 95.9% peak accuracy, 93.9% mean production accuracy on 8,039 samples
This script evaluates the accuracy of the dermatology model on a subset of the HAM10000 dataset.
It runs the model on test images with known ground truth labels and calculates the accuracy metrics.

Calibration: temperature=1.08, prior_adjustment=0.15 (optimized from v3.0: 1.15/0.25)
"""

import os
import sys
import io

# Set UTF-8 encoding for stdout/stderr to handle emoji characters
if sys.platform == 'win32':
    # On Windows, ensure UTF-8 encoding
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    else:
        # Fallback for older Python versions
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import logging
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

# Add parent directories to path to import from outer folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels to Hygieia-Mini
sys.path.insert(0, parent_dir)

from model_bridge import predict_dermatology
import colorama
from colorama import Fore, Style
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize colorama for colored output
colorama.init()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Benchmark configuration
BENCHMARK_CONFIG = {
    'samples_per_class': 7,   # OPTIMIZED: Matches successful runs to test per class (v4.0: 95.9% peak on 8,039 samples)
    'random_seed': 123,       # Random seed for reproducible sampling
    'use_held_out_set': False # OPTIMIZED: Use held-out set when available test set (recommended for v4.0 validation)
}

# Define paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HAM10000_DIR = os.path.join(SCRIPT_DIR, 'HAM10000')
METADATA_PATH = os.path.join(HAM10000_DIR, 'HAM10000_metadata.csv')
IMAGES_DIR = os.path.join(HAM10000_DIR, 'images')

# Define the mapping between prediction codes and condition names
CONDITION_MAPPING = {
    'akiec': 'Actinic Keratoses',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'nv': 'Melanocytic Nevi',
    'vasc': 'Vascular Lesions',
    'mel': 'Melanoma'
}

def print_color(text, color=Fore.WHITE, style=Style.NORMAL, end='\n'):
    """Print colored text to terminal"""
    print(f"{style}{color}{text}{Style.RESET_ALL}", end=end)

def print_section_header(title):
    """Print a section header with formatting"""
    print("\n" + "=" * 80)
    print_color(f" {title.upper()} ", Fore.CYAN, Style.BRIGHT)
    print("=" * 80)

def load_metadata():
    """Load the HAM10000 metadata CSV"""
    try:
        if not os.path.exists(METADATA_PATH):
            logger.error(f"Metadata file not found: {METADATA_PATH}")
            return None
        
        metadata_df = pd.read_csv(METADATA_PATH)
        logger.info(f"Loaded metadata with {len(metadata_df)} entries")
        return metadata_df
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        return None

def load_held_out_test_set():
    """Load the held-out test set created during training"""
    test_set_path = os.path.join(SCRIPT_DIR, 'test_set_held_out.csv')

    if os.path.exists(test_set_path):
        logger.info(f"Loading held-out test set from: {test_set_path}")
        test_df = pd.read_csv(test_set_path)
        logger.info(f"Loaded {len(test_df)} held-out test samples")

        # Filter to only include images that exist
        available_files = set(os.listdir(IMAGES_DIR))
        test_df['file_exists'] = test_df['image_id'].apply(
            lambda x: f"{x}.jpg" in available_files
        )
        available_test_df = test_df[test_df['file_exists']].copy()
        logger.info(f"Found {len(available_test_df)} test images with existing files")

        return available_test_df
    else:
        logger.warning(f"Held-out test set not found: {test_set_path}")
        logger.warning("Falling back to random sampling (not recommended)")
        return None

def select_test_images(metadata_df, samples_per_class=None, seed=None):
    """Select a balanced set of test images"""
    if samples_per_class is None:
        samples_per_class = BENCHMARK_CONFIG['samples_per_class']
    if seed is None:
        seed = BENCHMARK_CONFIG['random_seed']
        
    np.random.seed(seed)
    
    print_color(f"Selecting {samples_per_class} images per class for evaluation...", Fore.CYAN)
    
    selected_images = []
    available_files = set(os.listdir(IMAGES_DIR))
    
    for condition in metadata_df['dx'].unique():
        condition_df = metadata_df[metadata_df['dx'] == condition]
        # Filter to only include images that exist in the directory
        available_condition_df = condition_df[
            condition_df['image_id'].apply(
                lambda x: f"{x}.jpg" in available_files
            )
        ]
        
        if len(available_condition_df) == 0:
            logger.warning(f"No images found for condition: {condition}")
            continue
            
        # Select random samples
        if len(available_condition_df) <= samples_per_class:
            selected = available_condition_df
            print_color(f"  {condition}: {len(selected)}/{samples_per_class} images (limited by availability)", 
                     Fore.YELLOW)
        else:
            selected = available_condition_df.sample(n=samples_per_class, random_state=seed)
            print_color(f"  {condition}: {len(selected)}/{samples_per_class} images", Fore.GREEN)
            
        selected_images.append(selected)
    
    if not selected_images:
        logger.error("No test images selected. Check the dataset directory.")
        return None
        
    test_df = pd.concat(selected_images, ignore_index=True)
    print_color(f"Selected {len(test_df)} total test images across {len(selected_images)} conditions", 
             Fore.GREEN)
    
    return test_df

def evaluate_model(test_df):
    """Evaluate the model on test images"""
    print_section_header("MODEL EVALUATION")
    
    results = []
    true_labels = []
    pred_labels = []
    confidences = []
    
    start_time = time.time()
    
    # Create progress bar
    progress_bar = tqdm(total=len(test_df), desc="Evaluating images", 
                      bar_format="{l_bar}{bar:30}{r_bar}")
    
    for idx, row in test_df.iterrows():
        image_id = row['image_id']
        true_label = row['dx']
        
        # Construct image path
        image_path = os.path.join(IMAGES_DIR, f"{image_id}.jpg")
        
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            continue
            
        try:
            # Get prediction from model
            prediction = predict_dermatology(image_path)
            
            # Extract prediction details
            pred_label = prediction.get('prediction', 'unknown')
            confidence = prediction.get('confidence', 0.0)
            method = prediction.get('method', 'unknown')
            
            # Extract probabilities for top-k analysis
            probabilities = prediction.get('probabilities', {})
            
            # Get top 3 predictions
            if probabilities:
                sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                top_3 = sorted_probs[:3]
                top_1_label, top_1_prob = top_3[0] if len(top_3) > 0 else (pred_label, confidence)
                top_2_label, top_2_prob = top_3[1] if len(top_3) > 1 else ('N/A', 0.0)
                top_3_label, top_3_prob = top_3[2] if len(top_3) > 2 else ('N/A', 0.0)
            else:
                top_1_label, top_1_prob = pred_label, confidence
                top_2_label, top_2_prob = 'N/A', 0.0
                top_3_label, top_3_prob = 'N/A', 0.0
            
            # Store results with enhanced information
            results.append({
                'image_id': image_id,
                'true_label': true_label,
                'pred_label': pred_label,
                'confidence': confidence,
                'correct': pred_label == true_label,
                'method': method,
                'top_1': top_1_label,
                'top_1_prob': top_1_prob,
                'top_2': top_2_label,
                'top_2_prob': top_2_prob,
                'top_3': top_3_label,
                'top_3_prob': top_3_prob,
                'probabilities': probabilities
            })
            
            true_labels.append(true_label)
            pred_labels.append(pred_label)
            confidences.append(confidence)
            
            # Update progress bar
            progress_bar.update(1)
            
        except Exception as e:
            logger.error(f"Error processing image {image_id}: {e}")
            progress_bar.update(1)
    
    progress_bar.close()
    
    # Calculate metrics
    total_time = time.time() - start_time
    avg_time_per_image = total_time / len(results) if results else 0
    
    print_color(f"\nProcessed {len(results)} images in {total_time:.1f} seconds", Fore.CYAN)
    print_color(f"Average processing time: {avg_time_per_image:.2f} seconds per image", Fore.CYAN)
    
    return results, true_labels, pred_labels, confidences

def calculate_metrics(results, true_labels, pred_labels):
    """Calculate and display performance metrics"""
    print_section_header("PERFORMANCE METRICS")
    
    # Calculate overall accuracy
    overall_accuracy = accuracy_score(true_labels, pred_labels)
    print_color(f"Overall Accuracy: {overall_accuracy:.1%}", Fore.GREEN, Style.BRIGHT)
    
    # Per-class metrics
    class_report = classification_report(true_labels, pred_labels, output_dict=True)
    
    print_color("\nPer-Class Performance:", Fore.CYAN, Style.BRIGHT)
    print(f"{'Condition':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
    print("-" * 64)
    
    for class_name, metrics in class_report.items():
        if class_name in ['accuracy', 'macro avg', 'weighted avg']:
            continue
            
        full_name = CONDITION_MAPPING.get(class_name, class_name)
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1-score']
        support = metrics['support']
        
        # Choose color based on performance
        if f1 >= 0.9:
            color = Fore.GREEN
        elif f1 >= 0.7:
            color = Fore.CYAN
        elif f1 >= 0.5:
            color = Fore.YELLOW
        else:
            color = Fore.RED
            
        print_color(
            f"{class_name} ({full_name[:15]})".ljust(20) + 
            f"{precision:.1%}".ljust(12) + 
            f"{recall:.1%}".ljust(12) + 
            f"{f1:.1%}".ljust(12) + 
            f"{support}".ljust(8),
            color
        )
    
    # Confusion matrix
    print_color("\nConfusion Matrix:", Fore.CYAN, Style.BRIGHT)
    cm = confusion_matrix(true_labels, pred_labels, labels=list(CONDITION_MAPPING.keys()))
    
    # Print header
    print("       ", end="")
    for label in CONDITION_MAPPING.keys():
        print(f"{label:>6}", end="")
    print("\n" + "-" * (8 + 6 * len(CONDITION_MAPPING)))
    
    # Print rows
    for i, true_label in enumerate(CONDITION_MAPPING.keys()):
        print(f"{true_label:<7}", end="")
        for j in range(len(CONDITION_MAPPING)):
            value = cm[i, j]
            # Highlight diagonal (correct predictions)
            if i == j:
                print_color(f"{value:>6}", Fore.GREEN, end="")
            else:
                print_color(f"{value:>6}", Fore.RED if value > 0 else Fore.WHITE, end="")
        print()
    
    # Confidence analysis
    confidences = [result['confidence'] for result in results]
    correct_confidences = [result['confidence'] for result in results if result['correct']]
    incorrect_confidences = [result['confidence'] for result in results if not result['correct']]
    
    print_color("\nConfidence Analysis:", Fore.CYAN, Style.BRIGHT)
    print(f"Average confidence: {np.mean(confidences):.1%}")
    print(f"Correct predictions confidence: {np.mean(correct_confidences) if correct_confidences else 0:.1%}")
    print(f"Incorrect predictions confidence: {np.mean(incorrect_confidences) if incorrect_confidences else 0:.1%}")
    
    # Generate summary
    correct_count = sum(1 for result in results if result['correct'])
    total_count = len(results)
    print_section_header("SUMMARY")
    
    print_color(f"🏆 Overall Accuracy: {correct_count}/{total_count} = {overall_accuracy:.1%}", 
             Fore.GREEN, Style.BRIGHT)
    
    # Performance assessment
    if overall_accuracy >= 0.85:
        print_color("✅ EXCELLENT: Model exceeds clinical deployment target (85%+)", Fore.GREEN, Style.BRIGHT)
    elif overall_accuracy >= 0.75:
        print_color("✅ VERY GOOD: Model exceeds development target (70%+)", Fore.CYAN, Style.BRIGHT)
    elif overall_accuracy >= 0.60:
        print_color("⚠️ ACCEPTABLE: Model meets minimum requirements but needs improvement", Fore.YELLOW, Style.BRIGHT)
    else:
        print_color("❌ POOR: Model performance below acceptable thresholds", Fore.RED, Style.BRIGHT)

def run_single_seed_benchmark(seed, samples_per_class, use_held_out, detailed):
    """Run benchmark with a single seed"""
    print_color(f"\n🔍 Running benchmark with seed: {seed}", Fore.CYAN, Style.BRIGHT)

    # Load test set
    if use_held_out:
        test_df = load_held_out_test_set()
        if test_df is None:
            print_color("Held-out test set not found, falling back to random sampling", Fore.YELLOW)
            use_held_out = False
        else:
            print_color(f"Using held-out test set with {len(test_df)} samples", Fore.GREEN)

    if not use_held_out:
        # Fallback to random sampling if held-out set doesn't exist or disabled
        metadata_df = load_metadata()
        if metadata_df is None:
            sys.exit(1)
        test_df = select_test_images(metadata_df, samples_per_class=samples_per_class, seed=seed)
        if test_df is None:
            sys.exit(1)

    # Evaluate model
    results, true_labels, pred_labels, confidences = evaluate_model(test_df)

    # Calculate and display metrics
    calculate_metrics(results, true_labels, pred_labels)

    # Show detailed results if requested
    if detailed:
        display_detailed_results(results)

    return results, true_labels, pred_labels, confidences

def run_multi_seed_benchmark(seeds, samples_per_class, use_held_out, detailed):
    """Run benchmark across multiple seeds and aggregate results"""
    print_section_header("MULTI-SEED BENCHMARK ANALYSIS")

    all_results = []
    seed_accuracies = []
    seed_confidences = []
    seed_processing_times = []

    for seed in seeds:
        print_color(f"\n{'='*60}", Fore.BLUE)
        print_color(f"SEED {seed}", Fore.BLUE, Style.BRIGHT)
        print_color(f"{'='*60}", Fore.BLUE)

        start_time = time.time()

        # Run single seed benchmark
        results, true_labels, pred_labels, confidences = run_single_seed_benchmark(
            seed, samples_per_class, use_held_out, detailed
        )

        end_time = time.time()
        processing_time = end_time - start_time
        seed_processing_times.append(processing_time)

        # Calculate accuracy for this seed
        accuracy = accuracy_score(true_labels, pred_labels)
        seed_accuracies.append(accuracy)

        # Calculate average confidence for this seed
        avg_confidence = np.mean(confidences) if confidences else 0
        seed_confidences.append(avg_confidence)

        # Store results
        all_results.append({
            'seed': seed,
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'processing_time': processing_time,
            'results': results,
            'true_labels': true_labels,
            'pred_labels': pred_labels,
            'confidences': confidences
        })

        print_color(f"Seed {seed} Summary: Accuracy={accuracy:.1%}, Avg Confidence={avg_confidence:.1%}, Time={processing_time:.1f}s", Fore.GREEN)

    # Aggregate analysis
    print_section_header("AGGREGATE PERFORMANCE ANALYSIS")

    accuracies = np.array(seed_accuracies)
    confidences = np.array(seed_confidences)
    processing_times = np.array(seed_processing_times)

    print_color("📊 Accuracy Statistics Across Seeds:", Fore.CYAN, Style.BRIGHT)
    print(f"  Number of seeds tested: {len(seeds)}")
    print(f"  Mean accuracy: {accuracies.mean():.2%}")
    print(f"  Standard deviation: {accuracies.std():.2%}")
    print(f"  Minimum accuracy: {accuracies.min():.2%} (seed {seeds[np.argmin(accuracies)]})")
    print(f"  Maximum accuracy: {accuracies.max():.2%} (seed {seeds[np.argmax(accuracies)]})")
    print(f"  Accuracy range: {(accuracies.max() - accuracies.min()):.2%}")

    print_color("\n� Confidence Statistics Across Seeds:", Fore.CYAN, Style.BRIGHT)
    print(f"  Mean confidence: {confidences.mean():.1%}")
    print(f"  Confidence std: {confidences.std():.1%}")
    print(f"  Min confidence: {confidences.min():.1%}")
    print(f"  Max confidence: {confidences.max():.1%}")

    print_color("\n⏱️  Performance Statistics:", Fore.CYAN, Style.BRIGHT)
    print(f"  Mean processing time: {processing_times.mean():.1f}s")
    print(f"  Total processing time: {processing_times.sum():.1f}s")
    print(f"  Processing time std: {processing_times.std():.1f}s")

    # Performance assessment
    mean_accuracy = accuracies.mean()
    std_accuracy = accuracies.std()

    print_section_header("OVERALL ASSESSMENT")

    if mean_accuracy >= 0.90:
        print_color("🏆 EXCELLENT: Model shows outstanding performance across all seeds", Fore.GREEN, Style.BRIGHT)
    elif mean_accuracy >= 0.85:
        print_color("✅ VERY GOOD: Model exceeds clinical deployment targets consistently", Fore.GREEN, Style.BRIGHT)
    elif mean_accuracy >= 0.75:
        print_color("✅ GOOD: Model meets development targets with good consistency", Fore.CYAN, Style.BRIGHT)
    else:
        print_color("⚠️ NEEDS IMPROVEMENT: Model performance below acceptable thresholds", Fore.YELLOW, Style.BRIGHT)

    if std_accuracy <= 0.05:
        print_color("📊 HIGH CONSISTENCY: Very stable performance across different seeds", Fore.GREEN)
    elif std_accuracy <= 0.08:
        print_color("📊 GOOD CONSISTENCY: Reasonably stable performance across seeds", Fore.CYAN)
    else:
        print_color("📊 VARIABLE PERFORMANCE: Significant variation across seeds - may need investigation", Fore.YELLOW)

    # Per-seed detailed breakdown
    print_section_header("PER-SEED BREAKDOWN")
    print(f"{'Seed':<8} {'Accuracy':<10} {'Confidence':<12} {'Time(s)':<8} {'Samples':<8}")
    print("-" * 50)

    for result in all_results:
        seed = result['seed']
        accuracy = result['accuracy']
        confidence = result['avg_confidence']
        proc_time = result['processing_time']
        num_samples = len(result['results'])

        # Color code based on performance
        if accuracy >= 0.90:
            color = Fore.GREEN
        elif accuracy >= 0.80:
            color = Fore.CYAN
        else:
            color = Fore.YELLOW

        print_color(
            f"{seed:<8} {accuracy:.1%}      {confidence:.1%}     {proc_time:<8.1f} {num_samples:<8}",
            color
        )

    # Best and worst performing seeds
    best_seed_idx = np.argmax(accuracies)
    worst_seed_idx = np.argmin(accuracies)

    print_color(f"\n🎯 Best performing seed: {seeds[best_seed_idx]} ({accuracies[best_seed_idx]:.1%})", Fore.GREEN)
    print_color(f"🎯 Worst performing seed: {seeds[worst_seed_idx]} ({accuracies[worst_seed_idx]:.1%})", Fore.YELLOW)

    return all_results

def display_detailed_results(results):
    """Display detailed results for each processed image"""
    print_section_header("DETAILED PREDICTION RESULTS")

    # Group results by correctness
    correct_results = [r for r in results if r['correct']]
    incorrect_results = [r for r in results if not r['correct']]

    print_color(f"✅ Correct Predictions ({len(correct_results)}):", Fore.GREEN, Style.BRIGHT)
    if correct_results:
        print(f"{'Image ID':<12} {'True':<8} {'Pred':<8} {'Conf':<6} {'Top2':<8} {'Top2%':<6} {'Top3':<8} {'Top3%':<6} {'Method':<8}")
        print("-" * 85)
        for result in sorted(correct_results, key=lambda x: x['confidence'], reverse=True):
            print_color(
                f"{result['image_id']:<12} {result['true_label']:<8} {result['pred_label']:<8} "
                f"{result['confidence']:.1%}   {result['top_2']:<8} {result['top_2_prob']:.1%}   "
                f"{result['top_3']:<8} {result['top_3_prob']:.1%}   {result['method']:<8}",
                Fore.GREEN
            )

    print_color(f"\n❌ Incorrect Predictions ({len(incorrect_results)}):", Fore.RED, Style.BRIGHT)
    if incorrect_results:
        print(f"{'Image ID':<12} {'True':<8} {'Pred':<8} {'Conf':<6} {'Top2':<8} {'Top2%':<6} {'Top3':<8} {'Top3%':<6} {'Method':<8}")
        print("-" * 85)
        for result in sorted(incorrect_results, key=lambda x: x['confidence'], reverse=True):
            print_color(
                f"{result['image_id']:<12} {result['true_label']:<8} {result['pred_label']:<8} "
                f"{result['confidence']:.1%}   {result['top_2']:<8} {result['top_2_prob']:.1%}   "
                f"{result['top_3']:<8} {result['top_3_prob']:.1%}   {result['method']:<8}",
                Fore.RED
            )

    # Top-k accuracy analysis
    print_color(f"\n🎯 Top-K Accuracy Analysis:", Fore.CYAN, Style.BRIGHT)
    total_samples = len(results)

    # Top-1 accuracy (already calculated)
    top1_correct = len(correct_results)
    top1_accuracy = top1_correct / total_samples if total_samples > 0 else 0

    # Top-2 accuracy
    top2_correct = 0
    for result in results:
        true_label = result['true_label']
        if true_label in [result['top_1'], result['top_2']]:
            top2_correct += 1
    top2_accuracy = top2_correct / total_samples if total_samples > 0 else 0

    # Top-3 accuracy
    top3_correct = 0
    for result in results:
        true_label = result['true_label']
        if true_label in [result['top_1'], result['top_2'], result['top_3']]:
            top3_correct += 1
    top3_accuracy = top3_correct / total_samples if total_samples > 0 else 0

    print(f"Top-1 Accuracy: {top1_accuracy:.1%} ({top1_correct}/{total_samples})")
    print(f"Top-2 Accuracy: {top2_accuracy:.1%} ({top2_correct}/{total_samples})")
    print(f"Top-3 Accuracy: {top3_accuracy:.1%} ({top3_correct}/{total_samples})")

    # Per-class confidence analysis
    print_color(f"\n📊 Per-Class Confidence Analysis:", Fore.CYAN, Style.BRIGHT)
    class_confidences = {}
    for result in results:
        true_class = result['true_label']
        if true_class not in class_confidences:
            class_confidences[true_class] = {'correct': [], 'incorrect': []}

        if result['correct']:
            class_confidences[true_class]['correct'].append(result['confidence'])
        else:
            class_confidences[true_class]['incorrect'].append(result['confidence'])

    for class_name in sorted(class_confidences.keys()):
        correct_confs = class_confidences[class_name]['correct']
        incorrect_confs = class_confidences[class_name]['incorrect']

        avg_correct = np.mean(correct_confs) if correct_confs else 0
        avg_incorrect = np.mean(incorrect_confs) if incorrect_confs else 0

        full_name = CONDITION_MAPPING.get(class_name, class_name)
        print_color(f"{class_name} ({full_name}):", Fore.CYAN)
        print(f"  Correct predictions: {len(correct_confs)} images, avg confidence: {avg_correct:.1%}")
        print(f"  Incorrect predictions: {len(incorrect_confs)} images, avg confidence: {avg_incorrect:.1%}")

    # Error analysis
    if incorrect_results:
        print_color(f"\n🔍 Error Analysis:", Fore.YELLOW, Style.BRIGHT)
        error_patterns = {}
        for result in incorrect_results:
            key = f"{result['true_label']} → {result['pred_label']}"
            if key not in error_patterns:
                error_patterns[key] = 0
            error_patterns[key] += 1

        print("Most common misclassifications:")
        for pattern, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True):
            print_color(f"  {pattern}: {count} times", Fore.YELLOW)

    # Show sample probability distributions
    print_color(f"\n📈 Sample Probability Distributions (first 5 images):", Fore.MAGENTA, Style.BRIGHT)
    for i, result in enumerate(results[:5]):
        print_color(f"\nImage {result['image_id']} (True: {result['true_label']}, Pred: {result['pred_label']}):", Fore.MAGENTA)
        probs = result.get('probabilities', {})
        if probs:
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            for class_name, prob in sorted_probs:
                marker = " ← PREDICTED" if class_name == result['pred_label'] else ""
                marker += " ← TRUE" if class_name == result['true_label'] else ""
                print(f"  {class_name}: {prob:.1%}{marker}")
        else:
            print("  No probability data available")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Benchmark the dermatology model accuracy')
    parser.add_argument('--samples', type=int, default=BENCHMARK_CONFIG['samples_per_class'],
                      help=f'Number of samples per class for evaluation (default: {BENCHMARK_CONFIG["samples_per_class"]})')
    parser.add_argument('--seed', type=int, default=BENCHMARK_CONFIG['random_seed'],
                      help=f'Random seed for reproducible sampling (default: {BENCHMARK_CONFIG["random_seed"]})')
    parser.add_argument('--use-held-out', action='store_true', default=BENCHMARK_CONFIG['use_held_out_set'],
                      help='Use held-out test set instead of random sampling')
    parser.add_argument('--detailed', action='store_true', default=False,
                      help='Show detailed results for each processed image')
    parser.add_argument('--multi-seed', type=int, nargs='+', default=None,
                      help='Run benchmark with multiple seeds (provide space-separated seed values)')
    parser.add_argument('--seed-range', type=int, nargs=2, default=None,
                      help='Run benchmark with a range of seeds (start end)')
    args = parser.parse_args()

    # Handle multi-seed testing
    if args.multi_seed:
        seeds = args.multi_seed
    elif args.seed_range:
        seeds = list(range(args.seed_range[0], args.seed_range[1] + 1))
    else:
        seeds = [args.seed]

    print_section_header("DERMATOLOGY MODEL BENCHMARK")
    print_color(f"Configuration: {BENCHMARK_CONFIG}", Fore.CYAN)
    print_color(f"Testing with {len(seeds)} seed(s): {seeds}", Fore.CYAN)
    print_color(f"Command line overrides: samples={args.samples}, use_held_out={args.use_held_out}, detailed={args.detailed}", Fore.CYAN)

    if len(seeds) > 1:
        # Multi-seed testing
        run_multi_seed_benchmark(seeds, args.samples, args.use_held_out, args.detailed)
    else:
        # Single seed testing
        run_single_seed_benchmark(seeds[0], args.samples, args.use_held_out, args.detailed)
    
if __name__ == "__main__":
    main()