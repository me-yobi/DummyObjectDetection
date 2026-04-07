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


def check_datasets_exist(required_datasets=None):
    """Check if required datasets exist. Returns (bool, list of missing datasets)."""
    if required_datasets is None:
        required_datasets = ['grayscale', 'colored', 'grayscale_border', 'colored_border']
    
    missing = []
    for dataset in required_datasets:
        dataset_path = Path(f"datasets/rectangles_{dataset}")
        if not dataset_path.exists() or not any(dataset_path.iterdir()):
            missing.append(dataset)
    
    return len(missing) == 0, missing


def print_missing_datasets_error(missing_datasets):
    """Print helpful error message for missing datasets."""
    print("\n" + "="*80)
    print("❌ ERROR: Missing Required Datasets")
    print("="*80)
    print(f"\nThe following datasets are missing or empty:")
    for ds in missing_datasets:
        print(f"   • datasets/rectangles_{ds}")
    
    print(f"\n💡 To generate the missing datasets, run one of these commands:")
    print(f"   ./run generate_all              # Generate all 4 datasets")
    for ds in missing_datasets:
        print(f"   ./run {ds:<25}# Generate only {ds}")
    
    print(f"\n   You can also specify a custom number of images:")
    print(f"   ./run generate_all --num-images 100")
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Simple pipeline runner for DummyObjectDetection')
    parser.add_argument('command', choices=[
        'setup', 'grayscale', 'colored', 'grayscale_border', 'colored_border', 'generate_all', 
        'evaluate', 'analyze', 'kernel_analysis', 'demo', 'full', 'clean'
    ], help='Command to run')
    
    parser.add_argument('--num-images', type=int, default=500, 
                       help='Number of images to generate (default: 500)')
    parser.add_argument('--dataset', choices=['grayscale', 'colored', 'grayscale_border', 'colored_border', 'all'], default='all',
                       help='Dataset type for evaluation (default: all - evaluates all datasets)')
    
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
    
    elif args.command == 'grayscale_border':
        """Generate grayscale border dataset only"""
        success = run_command(
            f"python -m src.scripts.prepare_data --output-dir datasets/rectangles_grayscale_border --dataset-type grayscale_border --num-images {args.num_images}",
            f"📐 Generating {args.num_images} grayscale border images"
        )
        if success:
            print(f"\n✅ Grayscale border dataset generated with {args.num_images} images")
        else:
            print("\n❌ Grayscale border dataset generation failed!")
            sys.exit(1)
    
    elif args.command == 'colored_border':
        """Generate colored border dataset only"""
        success = run_command(
            f"python -m src.scripts.prepare_data --output-dir datasets/rectangles_colored_border --dataset-type colored_border --num-images {args.num_images}",
            f"🎨 Generating {args.num_images} colored border images"
        )
        if success:
            print(f"\n✅ Colored border dataset generated with {args.num_images} images")
        else:
            print("\n❌ Colored border dataset generation failed!")
            sys.exit(1)
    
    elif args.command == 'generate_all':
        """Generate all four datasets"""
        success = True
        success &= run_command(
            f"python -m src.scripts.prepare_data --output-dir datasets/rectangles_grayscale --dataset-type grayscale --num-images {args.num_images}",
            f"🎨 Generating {args.num_images} grayscale images"
        )
        success &= run_command(
            f"python -m src.scripts.prepare_data --output-dir datasets/rectangles_colored --dataset-type colored --num-images {args.num_images}",
            f"🌈 Generating {args.num_images} colored images"
        )
        success &= run_command(
            f"python -m src.scripts.prepare_data --output-dir datasets/rectangles_grayscale_border --dataset-type grayscale_border --num-images {args.num_images}",
            f"📐 Generating {args.num_images} grayscale border images"
        )
        success &= run_command(
            f"python -m src.scripts.prepare_data --output-dir datasets/rectangles_colored_border --dataset-type colored_border --num-images {args.num_images}",
            f"🎨 Generating {args.num_images} colored border images"
        )
        if success:
            print(f"\n✅ All four datasets generated with {args.num_images} images each")
        else:
            print("\n❌ Dataset generation failed!")
            sys.exit(1)
    
    elif args.command == 'evaluate':
        """Evaluate model on specified dataset(s)"""
        
        # Check if required datasets exist
        if args.dataset == 'all':
            datasets_to_check = ['grayscale', 'colored', 'grayscale_border', 'colored_border']
        else:
            datasets_to_check = [args.dataset]
        
        datasets_exist, missing = check_datasets_exist(datasets_to_check)
        if not datasets_exist:
            print_missing_datasets_error(missing)
            sys.exit(1)
        
        if args.dataset == 'all':
            """Evaluate all four datasets"""
            datasets = ['grayscale', 'colored', 'grayscale_border', 'colored_border']
            print("🔄 Evaluating all four datasets...\n")
            
            for dataset in datasets:
                # Update config for this dataset
                dataset_path = f"datasets/rectangles_{dataset}"
                config_path = "src/config.py"
                
                with open(config_path, 'r') as f:
                    config_content = f.read()
                
                new_config = re.sub(
                    r'DATASET_DIR = os\.path\.join\(BASE_DIR, "[^"]*"\).*',
                    f'DATASET_DIR = os.path.join(BASE_DIR, "{dataset_path}")  # {dataset.title()} dataset',
                    config_content
                )
                
                with open(config_path, 'w') as f:
                    f.write(new_config)
                
                print(f"🔧 Config updated to use {dataset} dataset")
                
                # Run evaluation
                success = run_command(
                    "python -m src.evaluate_model",
                    f"📊 Evaluating model on {dataset} dataset"
                )
                
                if success:
                    print(f"✅ Evaluation complete for {dataset} dataset\n")
                else:
                    print(f"❌ Evaluation failed for {dataset} dataset!\n")
            
            print("🎉 All evaluations complete!")
        else:
            """Evaluate single dataset"""
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
        """Analyze results for all datasets by default"""
        
        # Determine if we should analyze all datasets or just current
        if args.dataset == 'all':
            datasets = ['grayscale', 'colored', 'grayscale_border', 'colored_border']
            print("🔄 Analyzing all four datasets...\n")
            
            for dataset in datasets:
                dataset_path = f"datasets/rectangles_{dataset}"
                config_path = "src/config.py"
                
                with open(config_path, 'r') as f:
                    config_content = f.read()
                
                new_config = re.sub(
                    r'DATASET_DIR = os\.path\.join\(BASE_DIR, "[^"]*"\).*',
                    f'DATASET_DIR = os.path.join(BASE_DIR, "{dataset_path}")',
                    config_content
                )
                
                with open(config_path, 'w') as f:
                    f.write(new_config)
                
                print(f"🔧 Config updated to use {dataset} dataset")
                
                success = run_command(
                    "python -m src.scripts.analyze_results",
                    f"📈 Analyzing {dataset} dataset"
                )
                
                if success:
                    print(f"✅ Analysis complete for {dataset} dataset\n")
                else:
                    print(f"❌ Analysis failed for {dataset} dataset!\n")
            
            print("🎉 All analyses complete!")
        else:
            success = run_command(
                "python -m src.scripts.analyze_results",
                "📈 Analyzing results and generating visualizations"
            )
            if success:
                print("\n✅ Analysis complete! Check generated PNG files.")
            else:
                print("\n❌ Analysis failed!")
            sys.exit(1)
    
    elif args.command == 'kernel_analysis':
        """Run edge detection kernel analysis"""
        
        # Check if all required datasets exist
        datasets_exist, missing = check_datasets_exist(['grayscale', 'colored', 'grayscale_border', 'colored_border'])
        if not datasets_exist:
            print_missing_datasets_error(missing)
            sys.exit(1)
        
        success = run_command(
            "python -m src.scripts.kernel_analysis",
            "🔬 Running edge detection kernel analysis (Sobel X/Y)"
        )
        
        if success:
            success &= run_command(
                "python -m src.scripts.quantitative_kernel_analysis",
                "📊 Running quantitative edge analysis"
            )
        
        if success:
            print("\n✅ Kernel analysis complete! Generated files:")
            print("   • edge_detection_analysis.png - Full kernel response visualization")
            print("   • edge_profile_comparison.png - Cross-section edge profiles")
            print("   • quantitative_edge_analysis.png - Statistical comparison")
        else:
            print("\n❌ Kernel analysis failed!")
            sys.exit(1)
    
    elif args.command == 'demo':
        """Quick demo with 10 images per dataset"""
        print("🚀 Running demo with 10 images per dataset...")
        
        # Generate all four datasets with 10 images each
        success = True
        success &= run_command(
            "python -m src.scripts.prepare_data --output-dir datasets/demo_grayscale --dataset-type grayscale --num-images 10",
            "🎨 Generating 10 grayscale demo images"
        )
        success &= run_command(
            "python -m src.scripts.prepare_data --output-dir datasets/demo_colored --dataset-type colored --num-images 10",
            "🌈 Generating 10 colored demo images"
        )
        success &= run_command(
            "python -m src.scripts.prepare_data --output-dir datasets/demo_grayscale_border --dataset-type grayscale_border --num-images 10",
            "📐 Generating 10 grayscale border demo images"
        )
        success &= run_command(
            "python -m src.scripts.prepare_data --output-dir datasets/demo_colored_border --dataset-type colored_border --num-images 10",
            "🎨 Generating 10 colored border demo images"
        )
        
        if not success:
            print("\n❌ Demo dataset generation failed!")
            sys.exit(1)
        
        # Evaluate and analyze each dataset
        datasets_to_demo = [
            ('grayscale', 'demo_grayscale'),
            ('colored', 'demo_colored'), 
            ('grayscale_border', 'demo_grayscale_border'),
            ('colored_border', 'demo_colored_border')
        ]
        
        for dataset_type, dataset_path in datasets_to_demo:
            print(f"\n📊 Evaluating {dataset_type} dataset...")
            
            # Update config for this dataset
            config_path = "src/config.py"
            with open(config_path, 'r') as f:
                config_content = f.read()
            
            new_config = re.sub(
                r'DATASET_DIR = os\.path\.join\(BASE_DIR, "[^"]*"\).*',
                f'DATASET_DIR = os.path.join(BASE_DIR, "datasets/{dataset_path}")  # Demo {dataset_type} dataset',
                config_content
            )
            
            with open(config_path, 'w') as f:
                f.write(new_config)
            
            # Run evaluation
            success &= run_command(
                "python -m src.evaluate_model",
                f"📊 Evaluating on demo {dataset_type} dataset"
            )
            
            # Run analysis
            success &= run_command(
                "python -m src.scripts.analyze_results",
                f"📈 Analyzing demo {dataset_type} results"
            )
        
        # Run kernel analysis on demo datasets
        print(f"\n🔬 Running edge detection kernel analysis on demo datasets...")
        
        # Update kernel analysis to use demo datasets
        kernel_analysis_path = "src/scripts/kernel_analysis.py"
        with open(kernel_analysis_path, 'r') as f:
            kernel_content = f.read()
        
        # Replace dataset paths with demo paths
        kernel_content_demo = kernel_content.replace(
            "datasets/rectangles_grayscale", "datasets/demo_grayscale"
        ).replace(
            "datasets/rectangles_colored", "datasets/demo_colored"
        ).replace(
            "datasets/rectangles_grayscale_border", "datasets/demo_grayscale_border"
        ).replace(
            "datasets/rectangles_colored_border", "datasets/demo_colored_border"
        )
        
        # Write temporary demo kernel analysis file
        with open("src/scripts/kernel_analysis_demo.py", 'w') as f:
            f.write(kernel_content_demo)
        
        success &= run_command(
            "python -m src.scripts.kernel_analysis_demo",
            "🔬 Running edge detection kernel analysis (Sobel X/Y)"
        )
        
        # Update quantitative kernel analysis to use demo datasets
        quantitative_path = "src/scripts/quantitative_kernel_analysis.py"
        with open(quantitative_path, 'r') as f:
            quantitative_content = f.read()
        
        # Replace dataset paths with demo paths
        quantitative_content_demo = quantitative_content.replace(
            "datasets/rectangles_grayscale", "datasets/demo_grayscale"
        ).replace(
            "datasets/rectangles_colored", "datasets/demo_colored"
        ).replace(
            "datasets/rectangles_grayscale_border", "datasets/demo_grayscale_border"
        ).replace(
            "datasets/rectangles_colored_border", "datasets/demo_colored_border"
        )
        
        # Write temporary demo quantitative kernel analysis file
        with open("src/scripts/quantitative_kernel_analysis_demo.py", 'w') as f:
            f.write(quantitative_content_demo)
        
        success &= run_command(
            "python -m src.scripts.quantitative_kernel_analysis_demo",
            "📊 Running quantitative edge analysis"
        )
        
        # Clean up temporary files
        if os.path.exists("src/scripts/kernel_analysis_demo.py"):
            os.remove("src/scripts/kernel_analysis_demo.py")
        if os.path.exists("src/scripts/quantitative_kernel_analysis_demo.py"):
            os.remove("src/scripts/quantitative_kernel_analysis_demo.py")
        
        if success:
            print("\n🎉 Demo complete! Check the generated visualizations:")
            print("   • analysis_grayscale_predictions_*.png")
            print("   • analysis_colored_predictions_*.png") 
            print("   • analysis_grayscale_border_predictions_*.png")
            print("   • analysis_colored_border_predictions_*.png")
            print("   • edge_detection_analysis.png - Kernel response visualization")
            print("   • edge_profile_comparison.png - Edge strength profiles")
            print("   • quantitative_edge_analysis.png - Statistical comparison")
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
            "datasets/rectangles_grayscale_border",
            "datasets/rectangles_colored_border",
            "datasets/rectangles",
            "datasets/demo_grayscale",
            "datasets/demo_colored"
        ]
        
        files_to_clean = [
            "validation_predictions.png", 
            "metrics_distribution.png",
            "dataset_comparison.png",
            "edge_detection_analysis.png",
            "edge_profile_comparison.png",
            "quantitative_edge_analysis.png",
            "analysis_*.png"  # Catches all analysis files
        ]
        
        print("🧹 Cleaning generated files...")
        
        for dir_path in dirs_to_clean:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                print(f"  📁 Removed: {dir_path}")
        
        for file_pattern in files_to_clean:
            if '*' in file_pattern:
                # Handle wildcards
                import glob
                for file_path in glob.glob(file_pattern):
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"  📄 Removed: {file_path}")
            else:
                # Handle exact filenames
                if os.path.exists(file_pattern):
                    os.remove(file_pattern)
                    print(f"  📄 Removed: {file_pattern}")
        
        print("\n✅ Clean complete!")

if __name__ == "__main__":
    main()
