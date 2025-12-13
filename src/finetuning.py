import torch
import os
import wandb
import numpy as np
import torch.nn.functional as F
from SpatioTemporalGNN import SpatioTemporalGNN
from dataset import HDF5ScenarioDataset
from config import (device, batch_size, num_workers, num_layers, num_gru_layers,
                    radius, input_dim, output_dim, sequence_length, hidden_channels,
                    dropout, learning_rate, project_name, dataset_name,
                    loss_alpha, loss_beta, loss_gamma, loss_delta, 
                    use_edge_weights, checkpoint_dir, use_gat)

# Use stronger gradient clipping for fine-tuning (autoregressive mode is less stable)
gradient_clip_value = 5.0
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from helper_functions.graph_creation_functions import collate_graph_sequences_to_batch
from torch.nn.utils import clip_grad_norm_
from helper_functions.visualization_functions import visualize_epoch
from helper_functions.helpers import advanced_directional_loss
from testing import update_graph_features

# PS:>> $env:PYTHONWARNINGS="ignore"; $env:TF_CPP_MIN_LOG_LEVEL="3"; python ./src/finetuning.py

"""
Fine-tuning with Scheduled Sampling and Multi-Step Rollouts

This addresses the exposure bias problem:
- Training: model sees ground truth features at every step (teacher forcing)
- Testing: model uses its own predictions (autoregressive)
- Problem: model never learns to recover from its own errors

Solution strategies implemented:
1. Scheduled Sampling: gradually mix model predictions with ground truth during training
2. Multi-step Rollout: train model to predict multiple steps ahead using its own predictions
3. Curriculum Learning: start with short rollouts, gradually increase length
"""


def scheduled_sampling_step(model, graph_sequence, teacher_forcing_ratio, 
                            optimizer, loss_fn, loss_alpha, loss_beta, loss_gamma, loss_delta,
                            device, max_rollout_steps_per_iteration=5, clamp_predictions=True):
    """training step with scheduled sampling: probabilistically mix ground truth features (with probability teacher_forcing_ratio) and model's own predictions to update features (autoregressive)"""
    model.train()
    T = len(graph_sequence)
    for t in range(T):
        graph_sequence[t] = graph_sequence[t].to(device)
    
    optimizer.zero_grad()
    accumulated_loss = 0.0
    
    total_mse = 0.0
    total_cosine_sim = 0.0
    total_angular = 0.0
    metric_count = 0
    num_teacher_forced = 0
    num_autoregressive = 0
    
    # process sequence with scheduled sampling:
    current_graph = graph_sequence[0]
    for t in range(T - 1):
        edge_w = current_graph.edge_attr if use_edge_weights else None
        out_predictions = model(        # predict next step
            current_graph.x, 
            current_graph.edge_index,
            edge_weight=edge_w,
            batch=current_graph.batch,
            batch_size=1,
            batch_num=0,
            timestep=t
        )
        
        # get ground truth for loss computation
        gt_graph = graph_sequence[t + 1]
        if gt_graph.y is None or torch.all(gt_graph.y == 0):
            continue
        
        # Handle case where number of nodes changes between timesteps
        # Only compute loss for nodes that exist in both timesteps
        num_pred_nodes = out_predictions.size(0)
        num_gt_nodes = gt_graph.y.size(0)
        
        if num_pred_nodes != num_gt_nodes:
            # Use minimum number of nodes
            num_nodes = min(num_pred_nodes, num_gt_nodes)
            out_predictions_matched = out_predictions[:num_nodes]
            gt_targets_matched = gt_graph.y[:num_nodes].to(out_predictions.dtype)
            current_features_matched = current_graph.x[:num_nodes]
        else:
            out_predictions_matched = out_predictions
            gt_targets_matched = gt_graph.y.to(out_predictions.dtype)
            current_features_matched = current_graph.x
        
        loss_t = loss_fn(          # compute loss
            out_predictions_matched, 
            gt_targets_matched,
            current_features_matched, 
            alpha=loss_alpha, 
            beta=loss_beta,
            gamma=loss_gamma, 
            delta=loss_delta
        )
        
        # Safety check: skip this step if loss is NaN or infinite
        if torch.isnan(loss_t) or torch.isinf(loss_t):
            print(f"    WARNING: Invalid loss at step {step}: {loss_t.item():.4f}, skipping")
            continue
            
        accumulated_loss += loss_t
        
        with torch.no_grad():   # track metrics
            mse = F.mse_loss(out_predictions_matched, gt_targets_matched)
            pred_norm = F.normalize(out_predictions_matched, p=2, dim=1, eps=1e-6)
            target_norm = F.normalize(gt_targets_matched, p=2, dim=1, eps=1e-6)
            cos_sim = F.cosine_similarity(pred_norm, target_norm, dim=1).mean()
            
            pred_angle = torch.atan2(out_predictions_matched[:, 1], out_predictions_matched[:, 0])
            target_angle = torch.atan2(gt_targets_matched[:, 1], gt_targets_matched[:, 0])
            angle_diff = pred_angle - target_angle
            angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
            mean_angle_error = torch.abs(angle_diff).mean()
            
            total_mse += mse.item()
            total_cosine_sim += cos_sim.item()
            total_angular += mean_angle_error.item()
            metric_count += 1
        
        # scheduled sampling: decide whether to use ground truth or prediction for next step
        use_teacher_forcing = np.random.rand() < teacher_forcing_ratio
        
        if use_teacher_forcing or t + 1 >= T - 1:
            # use ground truth for next step
            current_graph = graph_sequence[t + 1]
            num_teacher_forced += 1
        else:
            # use model prediction to update features (autoregressive mode)
            # IMPORTANT: Use ground truth graph structure but update features with predictions
            with torch.no_grad():
                next_gt_graph = graph_sequence[t + 1]
                # Only update features, keep the same graph structure as ground truth
                current_graph = next_gt_graph.clone()
                
                # Update features based on predictions (only for nodes that exist in both)
                num_nodes = min(out_predictions.size(0), current_graph.x.size(0))
                if num_nodes > 0:
                    dt = 0.1
                    # Calculate new velocity from predicted displacement
                    new_vx = out_predictions[:num_nodes, 0] / dt
                    new_vy = out_predictions[:num_nodes, 1] / dt
                    
                    # Update velocity features
                    current_graph.x[:num_nodes, 0] = new_vx
                    current_graph.x[:num_nodes, 1] = new_vy
                    current_graph.x[:num_nodes, 2] = torch.sqrt(new_vx**2 + new_vy**2)
                    current_graph.x[:num_nodes, 3] = torch.atan2(new_vy, new_vx) / np.pi
            num_autoregressive += 1
    
    if accumulated_loss > 0:    # backpropagate
        accumulated_loss.backward()
        clip_grad_norm_(model.parameters(), gradient_clip_value)
        optimizer.step()
    
    metrics = {
        'loss': accumulated_loss.item() if metric_count > 0 else 0.0,
        'mse': total_mse / max(1, metric_count),
        'cosine_sim': total_cosine_sim / max(1, metric_count),
        'angle_error': total_angular / max(1, metric_count),
        'teacher_forced_ratio': num_teacher_forced / max(1, num_teacher_forced + num_autoregressive)
    }

    return accumulated_loss.item() if metric_count > 0 else 0.0, metrics


def multistep_rollout_step(model, graph_sequence, rollout_length, 
                           optimizer, loss_fn, loss_alpha, loss_beta, loss_gamma, loss_delta,
                           device, detach_interval=3):
    """training step with multi-step rollout: predict K steps ahead using own predictions:
        - start from timestep t
        - predict t+1, t+2, ..., t+K using autoregressive rollout
        - compute loss across all K predictions
        - backpropagate through the rollout (with optional detachment for memory efficiency)
    (detach_interval: Detach gradients every N steps to save memory (0 = no detach))"""
    model.train()
    T = len(graph_sequence)
    
    for t in range(T):
        graph_sequence[t] = graph_sequence[t].to(device)
    
    optimizer.zero_grad()
    
    # select random starting point (leave room for rollout)
    if T - rollout_length <= 1:
        return 0.0, {'loss': 0.0, 'mse': 0.0, 'cosine_sim': 0.0, 'angle_error': 0.0, 'rollout_length': 0}
    start_t = np.random.randint(0, T - rollout_length - 1)
    
    # initialize with ground truth at start_t
    current_graph = graph_sequence[start_t].clone()
    
    accumulated_loss = 0.0
    total_mse = 0.0
    total_cosine_sim = 0.0
    total_angular = 0.0
    valid_steps = 0
    
    # autoregressive rollout for K steps
    for step in range(rollout_length):
        t = start_t + step
        
        edge_w = current_graph.edge_attr if use_edge_weights else None
        out_predictions = model(    # predict next displacement
            current_graph.x,
            current_graph.edge_index,
            edge_weight=edge_w,
            batch=current_graph.batch,
            batch_size=1,
            batch_num=0,
            timestep=t
        )
        
        gt_graph = graph_sequence[t + 1]        # get ground truth
        if gt_graph.y is None or torch.all(gt_graph.y == 0):
            continue
        
        # Handle node count mismatch
        num_pred_nodes = out_predictions.size(0)
        num_gt_nodes = gt_graph.y.size(0)
        
        if num_pred_nodes != num_gt_nodes:
            num_nodes = min(num_pred_nodes, num_gt_nodes)
            out_predictions_matched = out_predictions[:num_nodes]
            gt_targets_matched = gt_graph.y[:num_nodes].to(out_predictions.dtype)
            current_features_matched = current_graph.x[:num_nodes]
        else:
            out_predictions_matched = out_predictions
            gt_targets_matched = gt_graph.y.to(out_predictions.dtype)
            current_features_matched = current_graph.x
        
        loss_t = loss_fn(       # compute loss
            out_predictions_matched,
            gt_targets_matched,
            current_features_matched,
            alpha=loss_alpha,
            beta=loss_beta,
            gamma=loss_gamma,
            delta=loss_delta
        )
        
        # Safety check: skip this step if loss is NaN or infinite
        if torch.isnan(loss_t) or torch.isinf(loss_t):
            print(f"    WARNING: Invalid loss at rollout step {step}: {loss_t.item():.4f}, skipping")
            continue
        
        # weight early predictions more than later ones (optional)
        weight = 1.0 / (step + 1)  # could also use uniform weight = 1.0
        accumulated_loss += weight * loss_t
        
        # track metrics
        with torch.no_grad():
            mse = F.mse_loss(out_predictions_matched, gt_targets_matched)
            pred_norm = F.normalize(out_predictions_matched, p=2, dim=1, eps=1e-6)
            target_norm = F.normalize(gt_targets_matched, p=2, dim=1, eps=1e-6)
            cos_sim = F.cosine_similarity(pred_norm, target_norm, dim=1).mean()
            
            pred_angle = torch.atan2(out_predictions_matched[:, 1], out_predictions_matched[:, 0])
            target_angle = torch.atan2(gt_targets_matched[:, 1], gt_targets_matched[:, 0])
            angle_diff = pred_angle - target_angle
            angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
            mean_angle_error = torch.abs(angle_diff).mean()
            
            total_mse += mse.item()
            total_cosine_sim += cos_sim.item()
            total_angular += mean_angle_error.item()
            valid_steps += 1
        
        # update graph for next prediction using model's output
        if step < rollout_length - 1:
            # option 1: Backprop through entire rollout (memory intensive)
            # option 2: Detach periodically to save memory
            if detach_interval > 0 and (step + 1) % detach_interval == 0:
                current_graph = update_graph_features(current_graph, out_predictions.detach(), device)
            else:
                current_graph = update_graph_features(current_graph, out_predictions, device)
    
    
    if accumulated_loss > 0 and valid_steps > 0:
        accumulated_loss.backward()         # backpropagate
        clip_grad_norm_(model.parameters(), gradient_clip_value)
        optimizer.step()
    
    metrics = {
        'loss': accumulated_loss.item() if valid_steps > 0 else 0.0,
        'mse': total_mse / max(1, valid_steps),
        'cosine_sim': total_cosine_sim / max(1, valid_steps),
        'angle_error': total_angular / max(1, valid_steps),
        'rollout_length': valid_steps
    }
    
    return accumulated_loss.item() if valid_steps > 0 else 0.0, metrics


def finetune_epoch(model, dataloader, optimizer, loss_fn, loss_alpha, loss_beta, loss_gamma, loss_delta,
                   device, epoch, mode='scheduled_sampling', teacher_forcing_ratio=0.5, rollout_length=5):
    """one epoch of fine-tuning with either scheduled sampling or multi-step rollout
    (teacher_forcing_ratio: For scheduled sampling (0.0 = full autoregressive, 1.0 = full teacher forcing))"""
    model.train()
    
    total_loss = 0.0
    total_mse = 0.0
    total_cosine_sim = 0.0
    total_angular = 0.0
    metric_count = 0
    steps = 0
    
    for batch_idx, batch_dict in enumerate(dataloader):
        batched_graph_sequence = batch_dict["batch"]
        B = batch_dict["B"]
        T = batch_dict["T"]
        
        assert B == 1, f"batch_size must be 1 for sequential processing! Got B={B}"
        
        # Reset GRU hidden states for this scenario (per-agent)
        num_nodes = batched_graph_sequence[0].num_nodes
        model.reset_gru_hidden_states(num_agents=num_nodes)
        
        if mode == 'scheduled_sampling':
            loss, metrics = scheduled_sampling_step(
                model, batched_graph_sequence, teacher_forcing_ratio,
                optimizer, loss_fn, loss_alpha, loss_beta, loss_gamma, loss_delta, device
            )
        elif mode == 'rollout':
            loss, metrics = multistep_rollout_step(
                model, batched_graph_sequence, rollout_length,
                optimizer, loss_fn, loss_alpha, loss_beta, loss_gamma, loss_delta, device
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        total_loss += metrics['loss']
        total_mse += metrics['mse']
        total_cosine_sim += metrics['cosine_sim']
        total_angular += metrics['angle_error']
        metric_count += 1
        steps += 1
        
        if batch_idx % 5 == 0:
            if mode == 'scheduled_sampling':
                print(f"  Batch {batch_idx}: Loss={metrics['loss']:.4f}, TF_ratio={metrics.get('teacher_forced_ratio', 0):.2f}")
            else:
                print(f"  Batch {batch_idx}: Loss={metrics['loss']:.4f}, Rollout={metrics.get('rollout_length', 0)}")
    
    avg_metrics = {
        'loss': total_loss / max(1, steps),
        'mse': total_mse / max(1, metric_count),
        'cosine_sim': total_cosine_sim / max(1, metric_count),
        'angle_error': total_angular / max(1, metric_count)
    }
    
    return avg_metrics


def validate_epoch(model, dataloader, loss_fn, loss_alpha, loss_beta, loss_gamma, loss_delta, device):
    """validation with standard teacher forcing (ground truth at each step)"""
    model.eval()
    
    total_loss = 0.0
    total_mse = 0.0
    total_cosine_sim = 0.0
    total_angular = 0.0
    metric_count = 0
    
    with torch.no_grad():
        for batch_dict in dataloader:
            batched_graph_sequence = batch_dict["batch"]
            B = batch_dict["B"]
            T = batch_dict["T"]
            
            for t in range(T):
                batched_graph_sequence[t] = batched_graph_sequence[t].to(device)
            
            # Reset GRU for validation (per-agent)
            num_nodes = batched_graph_sequence[0].num_nodes
            model.reset_gru_hidden_states(num_agents=num_nodes)
            
            accumulated_loss = 0.0
            
            for t in range(T - 1):
                graph = batched_graph_sequence[t]
                gt_graph = batched_graph_sequence[t + 1]
                
                if gt_graph.y is None or torch.all(gt_graph.y == 0):
                    continue
                
                edge_w = graph.edge_attr if use_edge_weights else None
                out_predictions = model(
                    graph.x, graph.edge_index, edge_weight=edge_w,
                    batch=graph.batch, batch_size=1, batch_num=0, timestep=t
                )
                
                # Handle node count mismatch
                num_pred_nodes = out_predictions.size(0)
                num_gt_nodes = gt_graph.y.size(0)
                
                if num_pred_nodes != num_gt_nodes:
                    num_nodes = min(num_pred_nodes, num_gt_nodes)
                    out_predictions_matched = out_predictions[:num_nodes]
                    gt_targets_matched = gt_graph.y[:num_nodes].to(out_predictions.dtype)
                    graph_features_matched = graph.x[:num_nodes]
                else:
                    out_predictions_matched = out_predictions
                    gt_targets_matched = gt_graph.y.to(out_predictions.dtype)
                    graph_features_matched = graph.x
                
                loss_t = loss_fn(
                    out_predictions_matched, gt_targets_matched,
                    graph_features_matched, alpha=loss_alpha, beta=loss_beta,
                    gamma=loss_gamma, delta=loss_delta
                )
                accumulated_loss += loss_t
                
                mse = F.mse_loss(out_predictions_matched, gt_targets_matched)
                pred_norm = F.normalize(out_predictions_matched, p=2, dim=1, eps=1e-6)
                target_norm = F.normalize(gt_targets_matched, p=2, dim=1, eps=1e-6)
                cos_sim = F.cosine_similarity(pred_norm, target_norm, dim=1).mean()
                
                pred_angle = torch.atan2(out_predictions_matched[:, 1], out_predictions_matched[:, 0])
                target_angle = torch.atan2(gt_targets_matched[:, 1], gt_targets_matched[:, 0])
                angle_diff = pred_angle - target_angle
                angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
                mean_angle_error = torch.abs(angle_diff).mean()
                
                total_mse += mse.item()
                total_cosine_sim += cos_sim.item()
                total_angular += mean_angle_error.item()
                metric_count += 1
            
            total_loss += accumulated_loss.item()
    
    avg_metrics = {
        'loss': total_loss / max(1, len(dataloader)),
        'mse': total_mse / max(1, metric_count),
        'cosine_sim': total_cosine_sim / max(1, metric_count),
        'angle_error': total_angular / max(1, metric_count)
    }
    
    return avg_metrics


def run_finetuning(base_checkpoint_path=None,
                  train_dataset_path="./data/graphs/training/training.hdf5",
                  val_dataset_path="./data/graphs/validation/validation.hdf5",
                  mode='scheduled_sampling',  # 'scheduled_sampling' or 'rollout'
                  finetune_epochs=10,
                  finetune_lr=None):
    """fine-tune a pre-trained model with scheduled sampling or multi-step rollout:
        - scheduled sampling: gradually reduce teacher forcing ratio from 1.0 to 0.0
            - epochs 1-3: 80% teacher forcing (learning to handle some errors)
            - epochs 4-7: 50% teacher forcing (half ground truth, half predictions)
            - epochs 8-10: 20% teacher forcing (mostly autoregressive)
        - multi-step rollout: gradually increase rollout length from 3 to 10 steps
            - epochs 1-3: 3-step rollout
            - epochs 4-7: 5-step rollout
            - epochs 8-10: 10-step rollout
    (base_checkpoint_path: Path to pre-trained model (default: best_model.pt),
    finetune_lr: Learning rate (default: 0.1 * base learning rate))"""
    
    # load pre-trained model
    if base_checkpoint_path is None:
        base_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
    
    if not os.path.exists(base_checkpoint_path):
        raise FileNotFoundError(f"Base checkpoint not found: {base_checkpoint_path}\n"
                              f"Please run training.py first to get a baseline model.")
    
    print(f"Loading pre-trained model from {base_checkpoint_path}...")
    checkpoint = torch.load(base_checkpoint_path, map_location=device)
    
    config = checkpoint['config']
    model = SpatioTemporalGNN(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_channels'],
        output_dim=config['output_dim'],
        num_gcn_layers=config['num_layers'],
        num_gru_layers=1,
        dropout=config['dropout'],
        use_gat=False
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # setup optimizer with lower learning rate for fine-tuning
    if finetune_lr is None:
        finetune_lr = learning_rate * 0.05  # Even more conservative: 5% of base LR
    
    optimizer = torch.optim.Adam(model.parameters(), lr=finetune_lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True, min_lr=1e-7)
    loss_fn = advanced_directional_loss
    
    # load datasets
    try:
        train_dataset = HDF5ScenarioDataset(train_dataset_path, seq_len=sequence_length)
        val_dataset = HDF5ScenarioDataset(val_dataset_path, seq_len=sequence_length)
        print(f"\nLoaded datasets: {len(train_dataset)} train, {len(val_dataset)} val scenarios")
    except FileNotFoundError as e:
        print(f"ERROR: Dataset not found - {e}")
        exit(1)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_graph_sequences_to_batch,
        drop_last=True, persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_graph_sequences_to_batch,
        drop_last=True, persistent_workers=True if num_workers > 0 else False
    )
    
    # initialize wandb
    wandb.login()
    run = wandb.init(
        project=project_name,
        config={
            "mode": mode,
            "finetune_epochs": finetune_epochs,
            "finetune_lr": finetune_lr,
            "base_checkpoint": base_checkpoint_path,
            "batch_size": batch_size,
            "sequence_length": sequence_length
        },
        name=f"Finetune_{mode}_lr{finetune_lr}",
        dir="../wandb"
    )
    
    # create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"FINE-TUNING WITH {mode.upper()}")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Fine-tune epochs: {finetune_epochs}")
    print(f"Fine-tune learning rate: {finetune_lr}")
    print(f"Mode: {mode}")
    print(f"{'='*80}\n")
    
    best_val_loss = float('inf')
    
    for epoch in range(finetune_epochs):
        print(f"\nEpoch {epoch+1}/{finetune_epochs}")
        
        # Curriculum: adjust difficulty over epochs
        if mode == 'scheduled_sampling':
            # More gradual teacher forcing schedule to prevent instability
            if epoch < 2:
                teacher_forcing_ratio = 0.9  # Start higher
            elif epoch < 4:
                teacher_forcing_ratio = 0.8
            elif epoch < 6:
                teacher_forcing_ratio = 0.7  # Add intermediate step
            elif epoch < 8:
                teacher_forcing_ratio = 0.6  # Add intermediate step
            else:
                teacher_forcing_ratio = 0.5  # Don't go below 0.5 for stability
            rollout_length = None
            print(f"  Teacher forcing ratio: {teacher_forcing_ratio:.2f}")
        else:  # rollout mode
            teacher_forcing_ratio = None
            # Gradually increase rollout length (3 -> 10)
            if epoch < 3:
                rollout_length = 3
            elif epoch < 7:
                rollout_length = 5
            else:
                rollout_length = 10
            print(f"  Rollout length: {rollout_length} steps")
        
        # training
        train_metrics = finetune_epoch(
            model, train_loader, optimizer, loss_fn,
            loss_alpha, loss_beta, loss_gamma, loss_delta,
            device, epoch, mode=mode,
            teacher_forcing_ratio=teacher_forcing_ratio,
            rollout_length=rollout_length
        )
        
        # validation
        val_metrics = validate_epoch(
            model, val_loader, loss_fn,
            loss_alpha, loss_beta, loss_gamma, loss_delta, device
        )
        
        # update scheduler
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # log to wandb
        log_dict = {
            "epoch": epoch,
            "train_loss": train_metrics['loss'],
            "train_mse": train_metrics['mse'],
            "train_cosine_sim": train_metrics['cosine_sim'],
            "train_angle_error": train_metrics['angle_error'] * 180 / np.pi,
            "val_loss": val_metrics['loss'],
            "val_mse": val_metrics['mse'],
            "val_cosine_sim": val_metrics['cosine_sim'],
            "val_angle_error": val_metrics['angle_error'] * 180 / np.pi,
            "learning_rate": current_lr
        }
        
        if mode == 'scheduled_sampling':
            log_dict["teacher_forcing_ratio"] = teacher_forcing_ratio
        else:
            log_dict["rollout_length"] = rollout_length
        
        wandb.log(log_dict)
        
        # print epoch summary
        print(f"  Train Loss: {train_metrics['loss']:.6f} | Val Loss: {val_metrics['loss']:.6f}")
        print(f"  Train Cos: {train_metrics['cosine_sim']:.4f} | Val Cos: {val_metrics['cosine_sim']:.4f}")
        print(f"  Train Angle: {train_metrics['angle_error']*180/np.pi:.1f}° | "
              f"Val Angle: {val_metrics['angle_error']*180/np.pi:.1f}°")
        
        # save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_path = os.path.join(checkpoint_dir, f'finetuned_{mode}_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_metrics['loss'],
                'train_loss': train_metrics['loss'],
                'mode': mode,
                'config': config
            }, save_path)
            print(f"  -> Saved best fine-tuned model to {save_path}")
    
    # save final model
    final_path = os.path.join(checkpoint_dir, f'finetuned_{mode}_final.pt')
    torch.save({
        'epoch': finetune_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_metrics['loss'],
        'train_loss': train_metrics['loss'],
        'mode': mode,
        'config': config
    }, final_path)
    
    print(f"\n{'='*80}")
    print(f"FINE-TUNING COMPLETE!")
    print(f"{'='*80}")
    print(f"Best model saved to: {os.path.join(checkpoint_dir, f'finetuned_{mode}_best.pt')}")
    print(f"Final model saved to: {final_path}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"\nNext step: Run testing.py with the fine-tuned model for autoregressive evaluation")
    print(f"{'='*80}\n")
    
    wandb.finish()
    return model


if __name__ == '__main__':
    # fine-tune with scheduled sampling (recommended default)
    print("Starting fine-tuning with SCHEDULED SAMPLING...")
    print("This gradually reduces teacher forcing to prepare model for autoregressive inference.\n")
    
    model = run_finetuning(
        base_checkpoint_path=None,  # Uses best_model.pt
        mode='scheduled_sampling',  # or 'rollout'
        finetune_epochs=10,
        finetune_lr=None  # Auto: 0.1 * base learning rate
    )
    
    # Uncomment to also try rollout-based fine-tuning:
    # print("\n\nStarting fine-tuning with MULTI-STEP ROLLOUT...")
    # print("This trains on multi-step predictions with curriculum learning.\n")
    # model_rollout = run_finetuning(
    #     base_checkpoint_path=None,
    #     mode='rollout',
    #     finetune_epochs=10,
    #     finetune_lr=None
    # )
