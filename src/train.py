from data import DETRData
from model import DETR
from loss import DETRLoss, HungarianMatcher
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch import optim, load, save
from colorama import Fore  
from utils.logger import get_logger
from utils.rich_handlers import TrainingHandler, rich_training_context
from utils.training_metrics import save_training_summary
import sys 
import torch
from utils.boxes import stacker
import time
import os
import argparse
import shutil
import json
from word_classes import WORD_CLASSES


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Train SignDETR")
    parser.add_argument("--mode", choices=["alphabet", "words"], default="alphabet")
    args = parser.parse_args()

    # Initialize logger and handlers
    logger = get_logger("training")
    logger.print_banner()
    total_start_time = time.time()
    
    # Setup device - use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️ Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    
    is_words_mode = args.mode == "words"
    # Keep the accuracy-first Words profile identical whether training is
    # launched through this script or directly through train_words.py.
    batch_size = int(os.getenv("BATCH_SIZE", "8" if is_words_mode else "8"))
    grad_accum_steps = int(os.getenv("GRAD_ACCUM_STEPS", "1"))
    test_every = int(os.getenv("TEST_EVERY", "2" if is_words_mode else "5"))
    epochs = int(os.getenv("EPOCHS", "400" if is_words_mode else "200"))
    early_patience = int(os.getenv("EARLY_STOP_PATIENCE", "25" if is_words_mode else "15"))
    early_min_delta = float(os.getenv("EARLY_STOP_MIN_DELTA", "0.0"))
    freeze_backbone_epochs = int(os.getenv("FREEZE_BACKBONE_EPOCHS", "10" if is_words_mode and torch.cuda.is_available() else "5"))
    default_workers = "2" if is_words_mode and os.name == "nt" else "4"
    num_workers = int(os.getenv("NUM_WORKERS", default_workers))
    prefetch_factor = int(os.getenv("PREFETCH_FACTOR", "1" if is_words_mode else "2"))
    backbone_lr = float(os.getenv("BACKBONE_LR", "1e-5"))
    head_lr = float(os.getenv("HEAD_LR", "1e-4"))
    weight_decay = float(os.getenv("WEIGHT_DECAY", "1e-4"))
    scheduler_t0 = int(os.getenv("SCHEDULER_T0", "100" if is_words_mode else "60"))
    max_grad_norm = float(os.getenv("MAX_GRAD_NORM", "0.1"))
    persistent_workers = num_workers > 0

    if args.mode == "words":
        num_classes = len(WORD_CLASSES)
        train_dir = "data/words/train"
        test_dir = "data/words/test"
        best_path = "checkpoints/words/best_model.pt"
        final_path = "checkpoints/words/words_model.pt"
        checkpoint_prefix = "checkpoints/words/words"
    else:
        num_classes = 26
        train_dir = "data/train"
        test_dir = "data/test"
        best_path = "checkpoints/best_alphabet_model.pt"
        final_path = "checkpoints/alphabet_model.pt"
        checkpoint_prefix = "checkpoints/alphabet"
    weights= {'class_weighting': 1, 'bbox_weighting': 5, 'giou_weighting': 2}
    word_box_head_layers = int(os.getenv("WORDS_BOX_HEAD_LAYERS", "1")) if is_words_mode else 1
    words_image_size = int(os.getenv("WORDS_IMAGE_SIZE", "320")) if is_words_mode else 224

    train_dataset = DETRData(train_dir, words_mode=is_words_mode,
                             words_image_size=words_image_size)
    train_sampler = None
    if is_words_mode:
        sample_class_ids = []
        for filename in train_dataset.labels:
            with open(os.path.join(train_dataset.labels_path, filename), "r", encoding="utf-8") as label_file:
                first_label = next((line.split()[0] for line in label_file if line.strip()), None)
            if first_label is None:
                raise ValueError(f"Empty label file: {filename}")
            sample_class_ids.append(int(first_label))
        class_counts = torch.bincount(torch.tensor(sample_class_ids), minlength=num_classes).float()
        missing = [WORD_CLASSES[i] for i, count in enumerate(class_counts) if count == 0]
        if missing:
            raise ValueError(f"Words training data has no samples for: {', '.join(missing)}")
        sample_weights = torch.tensor(
            [1.0 / class_counts[class_id] for class_id in sample_class_ids], dtype=torch.double)
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=stacker,
        sampler=train_sampler,
        shuffle=not is_words_mode,
        drop_last=False,
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers,
    )

    test_dataset = DETRData(test_dir, train=False, words_mode=is_words_mode,
                            words_image_size=words_image_size)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=stacker,
        shuffle=False,
        drop_last=False,
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers,
    )

    model = DETR(num_classes=num_classes, box_head_layers=word_box_head_layers)
    model = model.to(device)
    model.log_model_info()
    model.train() 

    backbone_params = list(model.backbone.parameters())
    backbone_param_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_param_ids]
    opt = optim.AdamW(
        [{"params": backbone_params, "lr": backbone_lr},
         {"params": head_params, "lr": head_lr}],
        weight_decay=weight_decay,
    )
    # This scheduler advances once per epoch, so T_0 is expressed in epochs.
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=scheduler_t0, T_mult=2)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    matcher = HungarianMatcher(weights)
    criterion = DETRLoss(num_classes=num_classes, matcher=matcher, weight_dict=weights, eos_coef=0.1)
    criterion = criterion.to(device)  # Move loss function to GPU

    train_batches = len(train_dataloader)
    test_batches = len(test_dataloader)
    
    # Log training configuration
    training_config = {
        "Total Epochs": epochs,
        "Mode": args.mode,
        "Num Classes": num_classes,
        "Train Dir": train_dir,
        "Test Dir": test_dir,
        "Final Checkpoint": final_path,
        "Batch Size": batch_size,
        "Train Batches": train_batches,
        "Test Batches": test_batches,
        "Backbone / Head LR": f"{backbone_lr:g} / {head_lr:g}",
        "Weight Decay": weight_decay,
        "Optimizer": "AdamW",
        "Scheduler": "CosineAnnealingWarmRestarts",
        "Scheduler First Cycle (epochs)": scheduler_t0,
        "Max Gradient Norm": max_grad_norm,
        "Grad Accum Steps": grad_accum_steps,
        "Test Every (epochs)": test_every,
        "Early Stop Patience": early_patience,
        "Freeze Backbone (epochs)": freeze_backbone_epochs,
        "Num Workers": num_workers,
        "Prefetch Factor": prefetch_factor
    }
    logger.print_table("🏋️ Training Configuration", list(training_config.keys()), [list(training_config.values())])

    def _summarize_targets(targets):
        summary = []
        for t in targets:
            item = {}
            for k, v in t.items():
                if isinstance(v, torch.Tensor):
                    item[k] = list(v.shape)
                else:
                    item[k] = type(v).__name__
            summary.append(item)
        return summary
    
    def _set_backbone_trainable(is_trainable: bool):
        for p in model.backbone.parameters():
            p.requires_grad = is_trainable

    # Freeze backbone for initial epochs to speed up convergence
    if freeze_backbone_epochs > 0:
        _set_backbone_trainable(False)

    best_val_loss = float("inf")
    early_stop_count = 0
    best_checkpoint_saved = False
    completed_epochs = 0

    # Start training with rich context
    with rich_training_context() as training_handler:
        for epoch in range(epochs): 
            completed_epochs = epoch + 1
            if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs:
                _set_backbone_trainable(True)
            with training_handler.create_training_progress() as epoch_progress:
                epoch_task = epoch_progress.add_task(f"[bold blue] Progress {epoch+1}/{epochs}", train_loss=0.0, test_loss=0.0, total=train_batches)
                # Training phase
                model.train()
                train_epoch_loss = 0.0 
            
                # Create progress bar for current epoch
                opt.zero_grad(set_to_none=True)
                for batch_idx, batch in enumerate(train_dataloader): 
                    X, y = batch
                    X = X.to(device)  # Move input to GPU
                    # Move targets to GPU
                    y = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in y]
                    try: 
                        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                            yhat = model(X) 
                            loss_dict = criterion(yhat, y) 
                            weight_dict = criterion.weight_dict
                            
                            # Ensure we sum exactly over the expected weighted keys, and keep tensor dtype
                            losses = loss_dict['labels']['loss_ce']*weight_dict['class_weighting'] + loss_dict['boxes']['loss_bbox']*weight_dict['bbox_weighting'] + loss_dict['boxes']['loss_giou']*weight_dict['giou_weighting']
                        
                        # Calculate loss 
                        train_epoch_loss += losses.item() 

                        loss_scaled = losses / max(1, grad_accum_steps)
                        if torch.cuda.is_available():
                            scaler.scale(loss_scaled).backward()
                        else:
                            loss_scaled.backward()

                        if (batch_idx + 1) % max(1, grad_accum_steps) == 0:
                            if torch.cuda.is_available():
                                scaler.unscale_(opt)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                                scaler.step(opt)
                                scaler.update()
                            else:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                                opt.step()
                            opt.zero_grad(set_to_none=True)
                        
                        # Update progress
                        epoch_progress.update(epoch_task, advance=1, train_loss=round(train_epoch_loss/train_batches,5))
                        
                    except Exception as e: 
                        if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower():
                            logger.error(f"CUDA OOM at epoch {epoch}, batch {batch_idx}. Try lowering BATCH_SIZE or GRAD_ACCUM_STEPS.")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        else:
                            logger.error(f"Training error at epoch {epoch}, batch {batch_idx}: {str(e)}")
                        logger.error(f"Batch targets summary: {str(_summarize_targets(y))}")
                        sys.exit()

                # Do not discard gradients from a final partial accumulation.
                if train_batches % max(1, grad_accum_steps) != 0:
                    if torch.cuda.is_available():
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        scaler.step(opt)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        opt.step()
                    opt.zero_grad(set_to_none=True)
            
                # Progress lr 
                scheduler.step()
            
                # Test phase (every N epochs)
                if test_every > 0 and ((epoch + 1) % test_every == 0 or epoch == epochs - 1):
                    model.eval()
                    test_epoch_loss = 0.0
                    with torch.no_grad():
                        for batch_idx, batch in enumerate(test_dataloader):
                            X, y = batch
                            X = X.to(device)  # Move input to GPU
                            # Move targets to GPU
                            y = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in y]
                            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                                yhat = model(X)
                                loss_dict = criterion(yhat, y) 
                                weight_dict = criterion.weight_dict
                                losses = loss_dict['labels']['loss_ce']*weight_dict['class_weighting'] + loss_dict['boxes']['loss_bbox']*weight_dict['bbox_weighting'] + loss_dict['boxes']['loss_giou']*weight_dict['giou_weighting']
                            
                            # Calculate loss 
                            test_epoch_loss += losses.item() 
                            epoch_progress.update(epoch_task, advance=0, test_loss=round(test_epoch_loss/test_batches,5))

                    avg_val_loss = test_epoch_loss / max(1, test_batches)
                    if avg_val_loss + early_min_delta < best_val_loss:
                        best_val_loss = avg_val_loss
                        early_stop_count = 0
                        save(model.state_dict(), best_path)
                        best_checkpoint_saved = True
                        training_handler.save_checkpoint_status(best_path, epoch)
                    else:
                        early_stop_count += 1
                        if early_stop_count >= early_patience:
                            logger.info(f"🛑 Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.5f}")
                            break
                
                # Save checkpoints
                if epoch % 10 == 0 and epoch != 0: 
                    checkpoint_path = f"{checkpoint_prefix}_{epoch}_model.pt"
                    save(model.state_dict(), checkpoint_path)
                    training_handler.save_checkpoint_status(checkpoint_path, epoch)

            if early_stop_count >= early_patience:
                break
            
    # The application loads final_path; make it the best validated model.
    if best_checkpoint_saved:
        shutil.copy2(best_path, final_path)
        logger.info(f"Saved best validation checkpoint to {final_path}")
    else:
        save(model.state_dict(), final_path)
        logger.warning("No validation checkpoint was created; saved the final epoch instead.")
    if args.mode == "words":
        with open(os.path.join(os.path.dirname(final_path), "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"mode": "words", "classes": WORD_CLASSES,
                       "box_head_layers": word_box_head_layers,
                       "words_image_size": words_image_size}, f, indent=2)
    total_seconds = time.time() - total_start_time
    summary_path = save_training_summary(
        "results/training_runs.csv",
        {
            "mode": args.mode,
            "duration_seconds": round(total_seconds, 2),
            "duration_minutes": round(total_seconds / 60, 2),
            "epochs_completed": completed_epochs,
            "epochs_requested": epochs,
            "best_validation_loss": "" if best_val_loss == float("inf") else round(best_val_loss, 6),
            "train_samples": len(train_dataset), "validation_samples": len(test_dataset),
            "batch_size": batch_size, "gradient_accumulation_steps": grad_accum_steps,
            "backbone_learning_rate": backbone_lr, "head_learning_rate": head_lr,
            "weight_decay": weight_decay, "scheduler_first_cycle_epochs": scheduler_t0,
            "freeze_backbone_epochs": freeze_backbone_epochs,
        },
    )
    logger.info(f"Training summary saved to {summary_path}")
    logger.info(f"⏱️ Total training time: {total_seconds/60:.2f} minutes")
