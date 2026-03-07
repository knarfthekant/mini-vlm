from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class TrainConfig:
    # Dataset
    train_dataset_path: str = "HuggingFaceM4/FineVision_concat_shuffled_2"
    train_dataset_names: tuple[str, ...] = ("default", )  #('allava_laion', 'allava_vflan', 'cambrian(filtered)_processed', 'LLaVA_Instruct_150K', 'mmevol', 'sharegpt4o', 'sharegpt4v(coco)', 'sharegpt4v(knowledge)', 'sharegpt4v(llava)', 'sharegpt4v(sam)') # 'vision_flan(filtered)', 'lvis_instruct4v',
    stream_dataset: bool = True
    val_size: float = 1000
    max_sample_length: int = 3072 # 4096 originally
    batch_size: int = 4 # 1 originally
    max_images_per_example: int = 4
    max_images_per_knapsack: int = 12 # 18 originally

    # Training
    max_training_steps: int = 12000 # originally 40000
    resume_from_checkpoint: bool = False
    gradient_checkpointing: bool = True
    gradient_accumulation_steps: int = 4 # 2 originally
    max_grad_norm: float = 1.0
    compile: bool = False

    # Learning rate
    lr_vision_backbone: float = 1e-5
    lr_language_backbone: float = 1e-5
    lr_mp: float = 1e-3

    # Rating thresholds
    relevance_min_rating: int = 1
    image_correspondence_min_rating: int = 1
    visual_dependency_min_rating: int = 1
    formatting_min_rating: int = 1

    # Logging
    log_wandb: bool = True
    wandb_entity: str = None

    # Evaluation
    eval_in_epochs: bool = True
    eval_interval: int = 250 # originally 500
    stats_log_interval: int = 100
    use_lmms_eval: bool = False
    lmms_eval_limit: float = None
    lmms_eval_tasks: str = 'mmstar,mmmu_val,ocrbench,textvqa_val,docvqa_val,scienceqa,mme,infovqa_val,chartqa'
    lmms_eval_batch_size: int = 32
