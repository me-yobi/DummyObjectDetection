#!/usr/bin/env python3
"""
Simple pipeline runner for DummyObjectDetection project
Provides single-line commands for the complete workflow
"""

import argparse
import subprocess
import sys
import os
import re
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}")
    print(f"Running: {cmd}")
    print("-" * 50)
    
    try:
        # Don't capture output to show real-time progress
        result = subprocess.run(cmd, shell=True, check=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"Exit code: {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Simple pipeline runner for DummyObjectDetection')
    parser.add_argument('command', choices=[
        'setup', 'grayscale', 'colored', 'both', 
        'evaluate', 'analyze', 'demo', 'full', 'clean'
    ], help='Command to run')
    
    parser.add_argument('--num-images', type=int, default=500, 
                       help='Number of images to generate (default: 500)')
    parser.add_argument('--dataset', choices=['grayscale', 'colored'], default='colored',
                       help='Dataset type for evaluation (default: colored)')
    
    args = parser.parse_args()
    
    # Ensure we're in the project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    if args.command == 'setup':
        """Setup project with both datasets"""
        success = True
        success &= run_command(
            f"python src/data/DataGenerator.py",
            "📦 Generating both grayscale and colored datasets"
        )
        if success:
            print("\n✅ Setup complete! Both datasets generated.")
        else:
            print("\n❌ Setup failed!")
            sys.exit(1)
    
    elif args.command == 'grayscale':
        """Generate grayscale dataset only"""
        success = run_command(
            f"python -m src.scripts.prepare_data --output-dir datasets/rectangles_grayscale --dataset-type grayscale --num-images {args.num_images}",
            f"🎨 Generating {args.num_images} grayscale images"
        )
        if success:
            print(f"\n✅ Grayscale dataset generated with {args.num_images} images")
        else:
            print("\n❌ Grayscale dataset generation failed!")
            sys.exit(1)
    
    elif args.command == 'colored':
        """Generate colored dataset only"""
        success = run_command(
            f"python -m src.scripts.prepare_data --output-dir datasets/rectangles_colored --dataset-type colored --num-images {args.num_images}",
            f"🌈 Generating {args.num_images} colored images"
        )
        if success:
            print(f"\n✅ Colored dataset generated with {args.num_images} images")
        else:
            print("\n❌ Colored dataset generation failed!")
            sys.exit(1)
    
    elif args.command == 'both':
        """Generate both datasets"""
        success = True
        success &= run_command(
            f"python -m src.scripts.prepare_data --output-dir datasets/rectangles_grayscale --dataset-type grayscale --num-images {args.num_images}",
            f"🎨 Generating {args.num_images} grayscale images"
        )
        success &= run_command(
            f"python -m src.scripts.prepare_data --output-dir datasets/rectangles_colored --dataset-type colored --num-images {args.num_images}",
            f"🌈 Generating {args.num_images} colored images"
        )
        if success:
            print(f"\n✅ Both datasets generated with {args.num_images} images each")
        else:
            print("\n❌ Dataset generation failed!")
            sys.exit(1)
    
    elif args.command == 'evaluate':
        """Evaluate model on specified dataset"""
        # Update config first
        dataset_path = f"datasets/rectangles_{args.dataset}"
        
        # Read and update config
        config_path = "src/config.py"
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Update dataset path using regex for more robust matching
        new_config = re.sub(
            r'DATASET_DIR = os\.path\.join\(BASE_DIR, "[^"]*"\).*',
            f'DATASET_DIR = os.path.join(BASE_DIR, "{dataset_path}")  # {args.dataset.title()} dataset',
            config_content
        )
        
        with open(config_path, 'w') as f:
            f.write(new_config)
        
        print(f"🔧 Config updated to use {args.dataset} dataset")
        
        # Run evaluation
        success = run_command(
            "python -m src.evaluate_model",
            f"📊 Evaluating model on {args.dataset} dataset"
        )
        
        if success:
            print(f"\n✅ Evaluation complete for {args.dataset} dataset")
        else:
            print(f"\n❌ Evaluation failed for {args.dataset} dataset!")
            sys.exit(1)
    
    elif args.command == 'analyze':
        """Analyze results"""
        success = run_command(
            "python -m src.scripts.analyze_results",
            "📈 Analyzing results and generating visualizations"
        )
        if success:
            print("\n✅ Analysis complete! Check generated PNG files.")
        else:
            print("\n❌ Analysis failed!")
            sys.exit(1)
    
    elif args.command == 'demo':
        """Quick demo with small datasets"""
        print("🚀 Running quick demo with small datasets...")
        
        # Generate small datasets
        success = True
        success &= run_command(
            "python -m src.scripts.prepare_data --output-dir datasets/demo_grayscale --dataset-type grayscale --num-images 50",
            "🎨 Generating 50 grayscale demo images"
        )
        success &= run_command(
            "python -m src.scripts.prepare_data --output-dir datasets/demo_colored --dataset-type colored --num-images 50",
            "🌈 Generating 50 colored demo images"
        )
        
        if not success:
            print("\n❌ Demo dataset generation failed!")
            sys.exit(1)
        
        # Evaluate on colored dataset
        success &= run_command(
            "python -m src.scripts.prepare_data --output-dir datasets/demo_colored --dataset-type colored --num-images 50 --force",
            "🔧 Updating config for colored demo dataset"
        )
        
        # Update config for demo
        config_path = "src/config.py"
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        new_config = re.sub(
            r'DATASET_DIR = os\.path\.join\(BASE_DIR, "[^"]*"\).*',
            'DATASET_DIR = os.path.join(BASE_DIR, "datasets/demo_colored")  # Demo colored dataset',
            config_content
        )
        
        with open(config_path, 'w') as f:
            f.write(new_config)
        
        success &= run_command(
            "python -m src.evaluate_model",
            "📊 Evaluating on demo colored dataset"
        )
        
        success &= run_command(
            "python -m src.scripts.analyze_results",
            "📈 Analyzing demo results"
        )
        
        if success:
            print("\n🎉 Demo complete! Check the generated visualizations.")
        else:
            print("\n❌ Demo failed!")
            sys.exit(1)
    
    elif args.command == 'full':
        """Run complete pipeline"""
        print("🚀 Running complete pipeline...")
        
        # Generate datasets
        success = True
        success &= run_command(
            f"python src/data/DataGenerator.py",
            "📦 Generating both datasets"
        )
        
        # Evaluate on colored dataset
        config_path = "src/config.py"
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        new_config = re.sub(
            r'DATASET_DIR = os\.path\.join\(BASE_DIR, "[^"]*"\).*',
            'DATASET_DIR = os.path.join(BASE_DIR, "datasets/rectangles_colored")  # Colored dataset',
            config_content
        )
        
        with open(config_path, 'w') as f:
            f.write(new_config)
        
        success &= run_command(
            "python -m src.evaluate_model",
            "📊 Evaluating on colored dataset"
        )
        
        success &= run_command(
            "python -m src.scripts.analyze_results",
            "📈 Analyzing results"
        )
        
        if success:
            print("\n🎉 Full pipeline complete!")
            print("📁 Check generated files:")
            print("   - datasets/rectangles_grayscale/")
            print("   - datasets/rectangles_colored/")
            print("   - regression_predictions.png")
            print("   - validation_predictions.png")
            print("   - metrics_distribution.png")
        else:
            print("\n❌ Full pipeline failed!")
            sys.exit(1)
    
    elif args.command == 'clean':
        """Clean generated datasets and results"""
        import shutil
        
        dirs_to_clean = [
            "datasets/rectangles_grayscale",
            "datasets/rectangles_colored", 
            "datasets/rectangles",
            "datasets/demo_grayscale",
            "datasets/demo_colored"
        ]
        
        files_to_clean = [
            "regression_predictions.png",
            "validation_predictions.png", 
            "metrics_distribution.png",
            "dataset_comparison.png"
        ]
        
        print("🧹 Cleaning generated files...")
        
        for dir_path in dirs_to_clean:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                print(f"  📁 Removed: {dir_path}")
        
        for file_path in files_to_clean:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"  📄 Removed: {file_path}")
        
        print("\n✅ Clean complete!")

if __name__ == "__main__":
    main()
