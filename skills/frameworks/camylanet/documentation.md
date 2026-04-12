# CamylaNet - Medical Image Segmentation Framework

> **🚨 KEY REQUIREMENTS FOR CUSTOM TRAINERS:**
> 
> **When creating custom trainers, you MUST:**
> 1. ✅ **Inherit from `nnUNetTrainerNoDeepSupervision`** (NOT `nnUNetTrainer`)
> 2. ✅ **Use the correct import path:**
>    ```python
>    from camylanet.training.nnUNetTrainer.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
>    ```
> 3. ✅ **Only override `build_network_architecture` method** (do NOT override `__init__`)
>
> ❌ **Common mistake:** Using `nnUNetTrainer` as base class will cause errors!

CamylaNet is a wrapper framework based on nnUNet v2 for medical image segmentation. It provides a simplified API for data preprocessing, model training, and evaluation with support for custom network architectures.

## Basic Usage

```python
import camylanet

# Dataset ID (following nnUNet dataset format)
dataset_id = xxx
configuration = '2d'

# Step 1: Data preprocessing
plans_identifier = camylanet.plan_and_preprocess(
    dataset_id=dataset_id,
    configurations=[configuration]  # Options: ['2d', '3d_fullres']
)

# Step 2: Model training
# Use exp_name to organize experiments into separate folders
result_folder, training_log = camylanet.training_network(
    dataset_id=dataset_id,
    configuration=configuration,
    plans_identifier=plans_identifier,
    exp_name="experiment_v1", # Optional: distinct folder for this run
    initial_lr=0.01  # Optional: initial learning rate, default is 0.01
)

# Step 3: Result evaluation
results = camylanet.evaluate(
    dataset_id=dataset_id,
    result_folder=result_folder,
    exp_name="experiment_v1" # Must match training exp_name
)

# View results
print(f"Number of epochs: {len(training_log['epochs'])}")
if training_log['train_losses']:
    print(f"Final training loss: {training_log['train_losses'][-1]:.4f}")
print(f"Mean Dice Score: {results['foreground_mean']['Dice']:.4f}")
```

## Result Organization

Using `exp_name` creates a separate subdirectory structure:
`$camylanet_results/Datasetxxx_xxxxx/[exp_name]/nnUNetTrainer__nnUNetPlans__2d/fold_0/`

## Custom Network Architecture

### 1. Create Custom Network Class

```python
import torch
from torch import nn
from typing import List

class PlainConvUNet(nn.Module):
    """Simple PlainConvUNet implementation example"""
    def __init__(self, input_channels: int, num_classes: int,
                 features: List[int] = [32, 64, 128, 256], is_3d: bool = False):
        super().__init__()
        self.features = features
        self.is_3d = is_3d

        # Choose convolution operations based on 2D/3D
        if is_3d:
            self.conv_op = nn.Conv3d
            self.norm_op = nn.InstanceNorm3d
            self.pool_op = nn.MaxPool3d
            self.upsample_op = nn.ConvTranspose3d
        else:
            self.conv_op = nn.Conv2d
            self.norm_op = nn.InstanceNorm2d
            self.pool_op = nn.MaxPool2d
            self.upsample_op = nn.ConvTranspose2d

        # ... (rest of implementation similar to standard U-Net)
        
        self.final_conv = self.conv_op(features[0], num_classes, 1)

    def forward(self, x):
        # ... (forward pass implementation)
        return self.final_conv(x)
```

### 2. Create Custom Trainer

```python
# 🚨 CRITICAL: Inherit from nnUNetTrainerNoDeepSupervision
from camylanet.training.nnUNetTrainer.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from typing import Union, List, Tuple
import torch.nn as nn

class PlainConvUNetTrainer(nnUNetTrainerNoDeepSupervision):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                  arch_init_kwargs: dict,
                                  arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                  num_input_channels: int,
                                  num_output_channels: int,
                                  enable_deep_supervision: bool = True) -> nn.Module:

        # Detect 2D/3D from architecture parameters
        is_3d = 'Conv3d' in str(arch_init_kwargs.get('conv_op', ''))

        return PlainConvUNet(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            features=[32, 64, 128, 256],
            is_3d=is_3d
        )
```

### 3. Usage with Custom Trainer

```python
result_folder, training_log = camylanet.training_network(
    dataset_id=dataset_id,
    configuration=configuration,
    trainer_class=PlainConvUNetTrainer, # Pass class directly
    plans_identifier=plans_identifier,
    initial_lr=0.01
)
```

## API Reference

### `plan_and_preprocess`
```python
plans_identifier = camylanet.plan_and_preprocess(
    dataset_id: Union[int, List[int]],
    preprocessor_name: str = 'DefaultPreprocessor',
    configurations: List[str] = ['2d', '3d_fullres']
)
```

### `training_network`
```python
result_folder, training_log = camylanet.training_network(
    dataset_id: Union[int, str],
    configuration: str,
    trainer_class: Union[Type[nnUNetTrainer], str] = 'nnUNetTrainer',
    plans_identifier: str = 'nnUNetPlans',
    exp_name: Optional[str] = None,
    initial_lr: float = 0.01 
)
```

**Parameters:**
- `initial_lr` (float, optional): Initial learning rate for training. The learning rate can be adjusted based on your dataset and model requirements. Default value is 0.01.

### `evaluate`
```python
results = camylanet.evaluate(
    dataset_id: Union[int, str],
    result_folder: str,
    output_file: Optional[str] = None,
    exp_name: Optional[str] = None
)
```

### `training_network_1epoch` (For Quick Testing)

Runs a single epoch of training for unit testing or quick validation.

```python
result_folder, training_log = camylanet.training_network_1epoch(
    dataset_id: Union[int, str],
    configuration: str,
    trainer_class: Union[Type[nnUNetTrainer], str] = 'nnUNetTrainer',
    # ... other args same as training_network
)
```
**Returns**: `Tuple[str, dict]` - Same as `training_network`, but only runs 1 epoch. Saves temporary checkpoints.

## Configuration Types
- `2d`: 2D U-Net for 2D images or slice-by-slice processing
- `3d_fullres`: 3D U-Net for full resolution 3D processing

## Evaluation Metrics
- **Dice Coefficient**: Segmentation overlap [0,1], higher is better
- **HD95**: 95th percentile Hausdorff distance (mm), lower is better