import os
import numpy as np
import camylanet

# 🚨 CRITICAL: Dataset Configuration - MUST BE CHANGED ACCORDING TO EXPERIMENT!
# ⚠️  WARNING: DO NOT use the default values below!
# ✅ REQUIRED: Use the dataset_id and configuration specified in the experiment description!

# Default example configuration - MUST BE REPLACED with actual experiment configuration
dataset_id = None  # 🔥 MUST BE REPLACED: Use the dataset_id from experiment description (e.g., 1, 2, 5, 8)
configuration = None  # 🔥 MUST BE REPLACED: Use configuration from experiment description (e.g., '3d_fullres', '2d')
exp_name = "xxx"  # 🔥 MUST BE REPLACED: Use the exp_name provided in experiment context


def main():
    """
    Main function to run the experiment.
    This function is called when the script is executed directly.
    """
    # Create working directory for storing experiment data
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    
    # Initialize experiment data structure
    experiment_data = {
        'dataset': {
            'metrics': {'train': [], 'val': []},
            'result_folder': None,
            'epochs': [],
            'dice_scores': [],
            'hd95_scores': []
        }
    }
    
    try:
        # Step 1: Data preprocessing
        print("🔄 Step 1: Data Preprocessing...")
        plans_identifier = camylanet.plan_and_preprocess(
            dataset_id=dataset_id,
            configurations=[configuration]
        )
        print(f"✅ Preprocessing completed. Plans identifier: {plans_identifier}")

        # Step 2: Train using default trainer
        print("\n🚀 Step 2: Model Training...")
        result_folder, training_log = camylanet.training_network(
            dataset_id=dataset_id,
            configuration=configuration,
            plans_identifier=plans_identifier,
            exp_name=exp_name
        )
        
        # Save result folder path
        experiment_data['dataset']['result_folder'] = result_folder
        experiment_data['dataset']['epochs'] = training_log['epochs']

        # Print training log information
        print(f"\n✅ Training completed!")
        print(f"Number of epochs: {len(training_log['epochs'])}")
        if training_log['train_losses']:
            print(f"Final training loss: {training_log['train_losses'][-1]:.4f}")
            print(f"Final validation loss: {training_log['val_losses'][-1]:.4f}")

        # Step 3: Evaluate results
        print("\n📊 Step 3: Model Evaluation...")
        results = camylanet.evaluate(
            dataset_id=dataset_id,
            result_folder=result_folder,
            exp_name=exp_name
        )

        # Extract and store metrics
        dice_score = results['foreground_mean']['Dice']
        hd95_score = results['foreground_mean']['HD95']
        
        experiment_data['dataset']['dice_scores'].append(dice_score)
        experiment_data['dataset']['hd95_scores'].append(hd95_score)
        experiment_data['dataset']['metrics']['val'].append({
            'dice': dice_score,
            'hd95': hd95_score
        })

        # Print evaluation results
        print(f"\n✅ Evaluation completed!")
        print(f"Mean Dice score: {dice_score:.4f}")
        print(f"Mean HD95 score: {hd95_score:.4f}")
        
        # Save experiment data
        experiment_data_path = os.path.join(working_dir, 'experiment_data.npy')
        np.save(experiment_data_path, experiment_data)
        print(f"\n💾 Experiment data saved to: {experiment_data_path}")
        
        print("\n🎉 Experiment completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during experiment: {str(e)}")
        # Save partial data if available
        if experiment_data['dataset']['result_folder'] is not None:
            experiment_data_path = os.path.join(working_dir, 'experiment_data_partial.npy')
            np.save(experiment_data_path, experiment_data)
            print(f"📝 Partial data saved to: {experiment_data_path}")
        raise e


# 🔒 CRITICAL: This guard prevents code execution when imported
# This is REQUIRED to prevent OpenHands from accidentally executing the experiment
# when running syntax checks like `python -c "import experiment"`
if __name__ == "__main__":
    main()