import os
from time import time

import click
import torch
from tqdm import tqdm

import data
import models
import utils


def validate_arch(ctx, param, value):
    if models.valid_archs(value):
        return value
    else:
        raise click.BadParameter('Invalid choice: {}. (choose from {})'
                                 .format(value, ', '.join(models.valid_archs())))


def validate_dir(ctx, param, value):
    os.makedirs(value, exist_ok=True)
    return value


@click.command(context_settings={'help_option_names': ['-h', '--help']})
@click.argument('data_dir', type=click.Path(exists=True, file_okay=False), required=1)
@click.option('--arch', 'arch', callback=validate_arch, default='vgg11', help='Net architecture')
@click.option('--save_dir', 'save_dir', callback=validate_dir, default='checkpoints',
              help='Directory to store checkpoints')
@click.option('--learning_rate', 'lr', type=float, default=0.001, help='Initial learning rate')
@click.option('--hidden_units', 'h_units', type=int, default=1024, help='Hyper parameter')
@click.option('--epochs', 'epochs', type=click.IntRange(min=1), default=2, help='Number of training epics')
@click.option('--gpu', 'allow_gpu', flag_value=True, help='Allow gpu (cuda) for training')
def train_model(data_dir, arch, save_dir, lr, h_units, epochs, allow_gpu):
    """Train a deep neural net using images in the data_dir.
    Example Usage:
    python train.py flowers --gpu --epochs 5
    """
    click.echo('Training {} based model.'.format(arch))
    train_dir = os.path.join(data_dir, TRAIN_DIR)
    valid_dir = os.path.join(data_dir, VALID_DIR)

    num_classes = data.count_classes(os.path.join(data_dir, 'train'))
    device = utils.get_device(allow_gpu)
    model = models.build_model(arch, hidden_units=h_units, num_classes=num_classes)
    class_to_idx = data.class_to_idx(train_dir)

    loaders = {'train': data.get_dataloader(train_dir, arch, BATCH_SIZE, True),
               'valid': data.get_dataloader(valid_dir, arch, BATCH_SIZE, False)}
    best_acc = 0.0
    last_checkpoint = ''
    model.to(device)

    # Create the opitmizer, scheduler and loss criterion
    criterion = models.get_criterion()

    optimizer, lr_sched = models.get_optim(model, lr=lr)

    for epoch in range(1, epochs + 1):
        time_e0 = time()
        model, optimizer, train_loss, train_acc = train_epoch(loader=loaders['train'], model=model, device=device,
                                                              desc='Training Epoch {}/{}'.format(epoch, epochs),
                                                              loss_fn=criterion, opt=optimizer)
        valid_loss, valid_acc = validate_epoch(loader=loaders['valid'], model=model, crit=criterion, device=device)
        # Finish the epoch
        lr_sched.step(valid_loss)

        if valid_acc > best_acc:  # Save checkpoint IF it's better than the saved one
            best_acc = valid_acc
            checkpoint_fn = os.path.join(save_dir, 'checkpoint_{}.pth'.format(arch))
            last_checkpoint = checkpoint_fn
            torch.save({'arch'        : arch,
                        'hidden_units': h_units,
                        'num_classes' : num_classes,
                        'state_dict'  : model.state_dict(),
                        'class_to_idx': class_to_idx,
                        # 'optimizer'   : optimizer.state_dict(),
                        }, checkpoint_fn)
        # Report results to the screen for the epoch
        e_time = time() - time_e0
        e_time = (e_time // 3600, e_time % 3600 // 60, e_time % 60)
        click.echo('Epoch{ep:3.0f} ({et[0]:02,.0f}:{et[1]:02.0f}:{et[2]:02.0f}):\t'
                   'Train[Acc:{tacc:6.4f} Loss:{tloss:6.4f}]\t'
                   'Valid[Acc:{vacc:6.4f} Loss:{vloss:6.4f}]'
                   .format(ep=epoch, et=e_time, tloss=train_loss, tacc=train_acc, vloss=valid_loss, vacc=valid_acc),
                   nl=True, err=False)
    # Inform the user of the latest/best checkpoint that was saved
    click.echo('Checkpoint File: "{}"'.format(click.format_filename(last_checkpoint)))


@torch.enable_grad()
def train_epoch(loader, model, loss_fn, opt, device, desc=''):
    """Train one epoch"""
    loss_train = utils.ValueCache()
    correct_train = utils.ValueCache()
    model.train(True)
    # Cycle through training batches
    with tqdm(loader, desc=desc, leave=False, ncols=80) as prog:
        for images, labels in prog:
            images, labels = images.to(device), labels.to(device)

            opt.zero_grad()
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            loss = loss_fn(outputs, labels)
            loss.backward()
            opt.step()

            loss_train.update(loss.data.item(), labels.size(0))
            correct_train.update((preds == labels.data).sum().item(), labels.size(0))
            # Free memory
            del images, labels, outputs, preds
            torch.cuda.empty_cache()
    return model, opt, loss_train.mean, correct_train.mean


@torch.no_grad()
def validate_epoch(loader, model, crit, device, desc='Validating'):
    """Validate performance for one Epoch"""
    valid_loss = utils.ValueCache()
    valid_correct = utils.ValueCache()
    model.train(False)
    model.eval()
    with tqdm(loader, desc=desc, leave=False, ncols=80) as prog:
        for images, labels in prog:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            loss = crit(outputs, labels)

            valid_loss.update(loss.data.item(), labels.size(0))
            valid_correct.update((preds == labels.data).sum().item(), labels.size(0))
            # Clear GPU memory
            del images, labels, outputs, preds
            torch.cuda.empty_cache()
    return valid_loss.mean, valid_correct.mean


if __name__ == '__main__':
    TRAIN_DIR = 'train'
    VALID_DIR = 'valid'
    TEST_DIR = 'test'

    BATCH_SIZE = 48  # This is limited by GPU memory

    train_model()
