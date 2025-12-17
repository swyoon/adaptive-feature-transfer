import torch
import wandb
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
from prior import UniformPrior, init_from_pretrained
from tqdm import tqdm
from functools import partial
import time
from utils import MovingAverage


def train_epoch(loader, model, criterion, prior, optimizer, stop_prior_grad=False, freeze_model=False):
    criterion = criterion()
    model.train()
    total_ce = 0
    total_prior_loss = 0
    total_batches = 0

    correct_predictions = 0
    total_predictions = 0

    # Initialize dictionary to store total values of each prior metric
    total_prior_metrics = {}
    for d in loader:
        if len(d) == 2:
            inputs, targets = d
            pretrained_feats = None
            inputs, targets = inputs.cuda(), targets.cuda()
        else:
            inputs, pretrained_feats, targets = d
            if hasattr(inputs, 'items'):
                inputs = {k: v.cuda() for k, v in inputs.items()}
            else:
                inputs = inputs.cuda()
            pretrained_feats, targets = pretrained_feats.cuda(), targets.cuda()
        optimizer.zero_grad()
            
        outputs, feats = model(inputs, return_feat=True)
        if freeze_model:
            outputs = outputs.detach()
            feats = feats.detach()
        # get batch indices of targets whose value is not -1
        valid_indices = (targets != -1).nonzero(as_tuple=True)[0]
        if len(valid_indices) == 0:
            continue
        outputs = outputs[valid_indices]
        targets = targets[valid_indices]
        
        loss = criterion(outputs, targets)
        prior_loss = -prior.prec * prior.log_prob(feats, pretrained_feats, grad_indices=valid_indices)
        total_ce += loss.item()
        if stop_prior_grad:
            prior_loss = prior_loss.detach()
        total_prior_loss += prior_loss.item() if isinstance(prior_loss, torch.Tensor) else prior_loss
        
        # Add each prior metric to the total
        for metric in prior.metrics:
            if metric not in total_prior_metrics:
                total_prior_metrics[metric] = 0
            total_prior_metrics[metric] += prior.metrics[metric]
        
        loss += prior_loss
        loss.backward()
        optimizer.step()
        total_batches += 1
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += targets.size(0)
        correct_predictions += (predicted == targets).sum().item()
    avg_ce = total_ce / total_batches
    avg_prior_loss = total_prior_loss / total_batches
    accuracy = 100 * correct_predictions / total_predictions

    # Compute the average of each prior metric over all batches
    avg_prior_metrics = {metric: total / total_batches for metric, total in total_prior_metrics.items()}

    return {'train_ce': avg_ce, 'prior_loss': avg_prior_loss, 'train_acc': accuracy, **avg_prior_metrics}

def evaluate_model(loader, model, criterion=nn.CrossEntropyLoss, max_test_points=None):
    criterion = criterion(reduction='none')
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_ent = 0
    correct_predictions = 0
    total_predictions = 0
    points_seen = 0
    with torch.no_grad():
        for d in loader:
            if len(d) == 2:
                inputs, targets = d
                pretrained_feats = None
                inputs, targets = inputs.cuda(), targets.cuda()
            else:
                inputs, pretrained_feats, targets = d
                if hasattr(inputs, 'items'):
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                else:
                    inputs = inputs.cuda()
                pretrained_feats, targets = pretrained_feats.cuda(), targets.cuda()
            outputs = model(inputs)
            p = nn.functional.softmax(outputs, dim=1)
            total_ent += -torch.sum(p * torch.log(p + 1e-8)).item()
            loss = criterion(outputs, targets)
            total_loss += loss.sum().item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
            points_seen += len(targets)
            if max_test_points is not None and points_seen >= max_test_points:
                break
    avg_loss = total_loss / total_predictions
    accuracy = 100 * correct_predictions / total_predictions
    ent = total_ent / total_predictions
    return avg_loss, accuracy, ent


def train_model(model, loaders, optimizer='sgd', scheduler=None, steps=1000, eval_steps=500, lr=1e-3, prior_lr=1e-2, pretrained_init=None, prior=UniformPrior(), prior_pretrain_steps=0, prior_freq=1, freeze_init=False, stop_prior_grad=False, wandb_run=None, wd=0, step_offset=0, **kwargs):
    n_epochs = steps / len(loaders[0])
    n_epochs = int(n_epochs + 0.5)
    model = model.cuda()
    
    if pretrained_init is not None:
        init_from_pretrained(model, pretrained_init, freeze_init)
    criterion = nn.CrossEntropyLoss
    train_loader, test_loader = loaders

    def infinite_loader(loader):
        while True:
            for batch in loader:
                yield batch

    train_loader = infinite_loader(train_loader)

    # Check if optimizer is a string (need to create) or existing torch optimizer
    if isinstance(optimizer, str):
        print(f'Creating optimizer: {optimizer}')
        if optimizer == 'sgd':
            optim_class = partial(optim.SGD, momentum=0.9)
        elif optimizer == 'adam':
            optim_class = optim.Adam
        else:
            raise ValueError(f'Unknown optimizer {optimizer}')
        
        model_params = [p for p in model.parameters()]
        param_groups = [{'params': model_params, 'lr': lr, 'weight_decay': wd}]
        
        # Add prior parameters if they exist
        if isinstance(prior, nn.Module):
            prior.cuda()
            prior_params = [p for p in prior.parameters()]
            if len(prior_params) > 0:
                param_groups.append({'params': prior_params, 'lr': prior_lr, 'weight_decay': 0})
        
        optimizer = optim_class(param_groups)
    else:
        print("Using existing optimizer")
    
    # Check if scheduler is None (need to create) or existing torch scheduler
    if scheduler is None:
        print("Creating scheduler")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    else:
        print("Using existing scheduler")

    moving_avg = MovingAverage()
    steps_since_eval = 0
    steps_so_far = step_offset

    best_acc = 0
    for step in (pbar := tqdm(range(steps))):
        batch = next(train_loader)
        # for step_count, batch in enumerate(train_loader):
        # prior_freq steps of training the prior, then 1 step of training the model
        freeze_model = (step % prior_freq) > 0
        metrics = train_epoch([batch], model, criterion, prior, optimizer, stop_prior_grad, freeze_model)
        moving_avg.update(metrics)
        
        steps_since_eval += 1
        steps_so_far += 1

        if steps_since_eval >= eval_steps:
            averaged_metrics = moving_avg.average()
            train_acc = averaged_metrics['train_acc']
            test_ce, test_acc, test_ent = evaluate_model(test_loader, model, criterion, max_test_points=3000)
            test_prior_acc = prior.get_test_acc()
            desc = f"Train acc: {train_acc:.1f}, Test acc: {test_acc:.1f}"
            if 'prior_agree' in averaged_metrics:
                desc += f", Prior agree: {averaged_metrics['prior_agree']:.1f}"
            
            epoch = step // len(loaders[0])

            averaged_metrics.update({'test_acc': test_acc, 'test_ce': test_ce, 'test_ent': test_ent, 'epoch': epoch, 'steps': steps_so_far, 'prior_test_acc': test_prior_acc, 'lr': optimizer.param_groups[0]['lr']})
            pbar.set_description(desc)
            
            if wandb_run:
                wandb_run.log(averaged_metrics, step=steps_so_far)
            moving_avg.reset()  # Reset the moving average after logging
            steps_since_eval = 0

            if test_acc > best_acc:
                best_acc = test_acc
                   
        if scheduler is not None:
            scheduler.step()
    
    print('Finished training, evaluating...')
    test_ce, test_acc, test_ent = evaluate_model(test_loader, model, criterion)
    if wandb_run:
        epoch = step // len(loaders[0]) 
        wandb_run.log({'test_acc': test_acc, 'test_ce': test_ce, 'test_ent': test_ent, 'epoch': epoch, 'steps': steps_so_far}, step=steps_so_far)
    print(f'Final test acc: {test_acc:.3f}')

    return test_acc, best_acc


def train(loaders, model, init_model, prior, wandb_run, hypers):
    print(f'Trainable parameters: {sum([p.numel() for p in model.parameters()])/1e6:.2g}M')
    model.cuda()
    optimizer = hypers.pop("optimizer", None)
    scheduler = hypers.pop("scheduler", None)
    last_acc, best_acc = train_model(model, loaders, prior=prior, pretrained_init=init_model, wandb_run=wandb_run, 
                                   optimizer=optimizer, scheduler=scheduler,
                                   **hypers)
    return last_acc, best_acc