# SAM-FNet

This repository contains the implementation of the following paper:

SAM-FNet: SAM-Guided Fusion Network for Laryngo-Pharyngeal Tumor Detection



## Fine-tune SAM with LoRA

To fine-tune SAM using LoRA, we recommend following the guidelines provided in the original repository: [SAMed](https://github.com/hitachinsk/SAMed/tree/main).

### Steps:

1. **Fine-tune SAM:**
   - Follow the instructions in the [SAMed repository](https://github.com/hitachinsk/SAMed/tree/main) to fine-tune SAM with LoRA.
2. **Generate Local Images:**
   - After fine-tuning, modify the `crop_image.py` script to suit your requirements.
   - Run the script to generate local images:

```markdown
python crop_image.py
```



## Dataset

Organize your datasets in the following manner:

```markdown
datasets/
├── dataset1/
│   ├── global/
│   │   ├── train/
│   │   │   ├── benign/
│   │   │   ├── normal/
│   │   │   └── tumor/
│   │   ├── val/
│   │   │   ├── benign/
│   │   │   ├── normal/
│   │   │   └── tumor/
│   │   └── test/
│   │       ├── benign/
│   │       ├── normal/
│   │       └── tumor/
│   └── local_seg/
│       ├── train/
│       │   ├── benign/
│       │   ├── normal/
│       │   └── tumor/
│       ├── val/
│       │   ├── benign/
│       │   ├── normal/
│       │   └── tumor/
│       └── test/
│           ├── benign/
│           ├── normal/
│           └── tumor/
├── dataset6/
│   └── ...
```



## Training

1. Modify the `class_labels` variable in the `dataset.py` file to reflect the classes in your dataset.
2. Run this command to train SAM-FNet.

```markdown
python train.py --data_dir <Your folder> --save_path <Your output path> --num_classes <Your number of categories for your tasks> --pretrained True --encoder ResNet50
```

- Replace `<Your folder>` with the path to your dataset.
- Replace `<Your output path>` with the directory where you want to save the model checkpoints.
- Replace `<Your number of categories for your tasks>` with the number of classes in your classification task.



## Testing

1. Change "classes" in the val.py
2. Run this command to test.

```markdown
python val.py --model_path <Your model path> --encoder ResNet50 --dataset <Your dataset name> --save_path <Your output path>
```



## Acknowledgement

The code of SAM-FNet is built upon  [SAMed](https://github.com/hitachinsk/SAMed/tree/main) and [DLGNet](https://github.com/soleilssss/DLGNet), and we express our gratitude to these awesome projects.
