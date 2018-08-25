import json
from pprint import pprint

import click
import torch
import torch.nn.functional as F
from PIL import Image

import data
import models
import utils


@click.command(context_settings={'help_option_names': ['-h', '--help']})
@click.argument('image_filename', type=click.Path(exists=True, dir_okay=False), required=1)
@click.argument('checkpoint_fn', type=click.Path(exists=True, dir_okay=False), required=1)
@click.option('--top_k', type=click.IntRange(min=1), default=1, help='Image category predictions to return')
@click.option('--category_names', type=click.Path(exists=True, dir_okay=False),
              help='json file with maps from cat to name')
@click.option('--gpu', 'allow_gpu', flag_value=True)
def predict(image_filename, checkpoint_fn, top_k, category_names, allow_gpu):
    """
    Takes a PIL image filename and stored checkpoint file as arguments then predicts the most likely labels
    """
    # Load checkpoint and create model
    device = utils.get_device(allow_gpu)
    checkpoint = torch.load(checkpoint_fn, map_location='cpu')
    model = models.build_model(checkpoint['arch'], checkpoint['hidden_units'], checkpoint['num_classes'])
    model.load_state_dict(checkpoint['state_dict'])
    setattr(model, 'class_to_idx', checkpoint['class_to_idx'])
    model.to(device)

    classes, probs = do_predict(Image.open(image_filename), model, device, top_k)
    # Converts classes to long names if json file is provided
    if category_names is not None:
        with open(category_names, 'r') as f:
            names = json.load(f)
        classes = [names[c] for c in classes]
    # Display results to the terminal
    pprint(list(zip(classes, probs)))


@torch.no_grad()
def do_predict(pil_image, model, device, top_k=1):
    # def validate_epoch(loader, model, crit, device, desc='Validating'):
    model.train(False)
    model.eval()
    preprocess = data.get_transform(arch=model.arch, training=False)
    image = preprocess(pil_image)
    image.unsqueeze_(0)  # adds a dimension for batch, else it fails
    image = image.to(device)

    output = model(image)
    scores, labels = torch.topk(output.data, top_k)
    probs = F.softmax(scores.data, dim=1)

    probs = probs.to('cpu')[0].numpy()
    labels = labels.to('cpu')[0].numpy().astype(str)
    return labels, probs


if __name__ == '__main__':
    predict()
