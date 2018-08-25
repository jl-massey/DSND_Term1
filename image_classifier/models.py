import re
from collections import OrderedDict

from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import models


def get_clf(in_features, hidden_units, num_classes):
    """
    Create new classifier to replace the one from pre-trained models.
    """
    layers = OrderedDict([
        ('drop0', nn.Dropout()),
        ('fc0', nn.Linear(in_features, hidden_units)),
        ('relu0', nn.ReLU(inplace=True)),
        ('drop1', nn.Dropout()),
        ('fc1', nn.Linear(hidden_units, hidden_units)),
        ('relu1', nn.ReLU(inplace=True)),
        ('fc2', nn.Linear(hidden_units, num_classes))
    ])
    return nn.Sequential(layers)


def get_last(classifier):
    """
    Finds the in_features from the 'cropped' final layer
    from the pre-trained models.
    """
    in_features = getattr(classifier[1], 'in_features', -1)
    if in_features < 1:
        for child in list(classifier[1].children()):
            in_features = getattr(child, 'in_features', -1)
            if in_features > 0:
                break
    return classifier[0], in_features


def valid_archs(arch=None):
    """
    Checks if the architecture requested is valid.
    :param arch: (optional) Arch string to validate
    :return: True or False if arch is provided.
             If None, then a list of valid arches is provided.
    """
    valid = [mname for mname in dir(models)
             if re.match('[a-z]+.*', mname)
             if callable(getattr(models, mname))
             if mname.startswith('squeezenet') is False]
    if arch is None:
        return valid
    else:
        return arch in valid


def build_model(arch, hidden_units, num_classes):
    """
    Build a pre-trained model from torchvision library
    """
    if arch not in valid_archs():
        print('Invalid architecture requested [{}]\n Valid arch:[{}]'
              .format(arch, ', '.join(valid_archs())))
        return None
    else:
        # Use the arch param to call the function in torchvision.models
        get_model = getattr(models, arch)
        model = get_model(pretrained=True)

        # Freeze pre-trained model parameters
        for param in model.parameters():
            param.requires_grad = False

        if arch.startswith('inception'):
            model.aux_logits = False
        # Replace final layer
        (clf_name, in_features) = get_last(list(model.named_children())[-1])
        new_clf = get_clf(in_features, hidden_units, num_classes)
        setattr(model, clf_name, new_clf)
        setattr(model, 'arch', arch)
    return model


def get_optim(model, lr):
    opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr)
    # When .step is called, the validation loss is sent as an argument.  The scheduler reduces based on that.
    sch = lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=3, threshold=0.0001,
                                         threshold_mode='rel')
    return opt, sch


def get_criterion():
    return nn.CrossEntropyLoss()
