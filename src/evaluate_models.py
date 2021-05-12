"""@package evaluate_models

Author: Simon Sedlacek
Email: xsedla1h@stud.fit.vutbr.cz

This script takes all the models present in the data/models directory and 
runs an evaluation of each model. Intended to be used as an automated alternative
to the personal_vad_evaluate.ipynb evaluation jupyter notebook.

Average precision scores are computed for each class, mean average precision is
computed across all classses. Additionally, raw classification accuracy and
confusion matrix is computed.

The results are written to stdout.

$ python src/evaluate_models.py 2>/dev/null

"""

import numpy as np
from copy import copy
import torch

from personal_vad import PersonalVAD, pad_collate
from vad_et import VadETDataset
from vad_set import VadSETDataset
from vad_st import VadSTDataset
from vad_xvector import VadETDatasetX
from vad_ivector import VadETDatasetI
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.metrics import average_precision_score, confusion_matrix, \
        precision_score, accuracy_score
import kaldiio
import os
import sys
import ml_metrics as metrics
from glob import glob
from decimal import Decimal, ROUND_HALF_UP

#NOTE: edit if needed...
batch_size_test = 64
test_dir = 'data/eval_dir/'
data = 'data/test'

def quantize(number, prec):
    """Round to two decimal places"""
    prec = f'1e-{prec}'
    return str(Decimal(number).quantize(Decimal(prec), ROUND_HALF_UP))


def parse_model_name(model):
    """Parse the model name string and determine its specifications.

    Args:
        model (str): Name of the model.
    
    Returns:
        tuple: A tuple containing:

        arch (str): Personal VAD architecture type.
        embed (str): Target speaker embedding type.
        use_fc (bool): Indicates whether the model uses the last hidden layer.
        linear (bool): Indicates whether the last hidden layer activation function
            is linear. If false, the activation is tanh.
        score_type (int): Indicates the scoring method used by the model.
        input_dim (int): Model input layer dimension.
    """

    # determine the embedding type
    if 'ivec' in model:
        embed = 'ivec'
    elif 'xvec' in model:
        embed = 'xvec'
    else:
        embed = 'dvec'

    # determine the architecture and the input layer dimension
    if 'set' in model:
        arch = 'set'
        input_dim = 297

    elif 'st' in model:
        arch = 'st'
        input_dim = 41

    elif 'et' in model:
        arch = 'et'

        if embed == 'dvec':
            input_dim = 296
        elif embed == 'xvec':
            input_dim = 552
        else: # ivec
            input_dim = 440
    else:
        # unknown architecture..
        print(model, "other architecture...")
        return None

    # determine score type
    score_type = 0
    if arch in ['set', 'st']:
        if 'score0' in model: score_type = 0
        elif 'score1' in model: score_type = 1
        elif 'score2' in model: score_type = 2
        else:
            # unknown architecture..
            print(model, "other architecture...")
            return None

    # determine activation
    if embed == 'dvec':
        if 'tanh' in model:
            linear = False
            use_fc = True
        elif 'linear' in model:
            linear = True
            use_fc = True
        elif 'lrelu' in model:
            print(model, "leaky relu...")
            return None
        else:
            # not using the hidden layer..
            linear = True
            use_fc = False
    else:
        if 'tanh' in model:
            linear = False
            use_fc = True
        elif 'linear' in model:
            linear = True
            use_fc = True
        else:
            linear = False
            use_fc = True

    return arch, embed, use_fc, linear, score_type, input_dim

if __name__ == '__main__':
    # move to the evaluation directory
    os.chdir(test_dir)

    # get the model list
    models = glob('models/*pt')

    # evaluate the models one by one...
    for model in models:
        # get the model information
        ret = parse_model_name(model)
        if ret == None: continue
        (arch, embed, use_fc, linear, score_type, input_dim) = ret

        # load the model
        net = PersonalVAD(input_dim=input_dim, hidden_dim=64, num_layers=2,
                out_dim=3, use_fc=use_fc, linear=linear)
        net.load_state_dict(torch.load(model))

        # create the corresponding dataset objects
        if arch == 'et':
            if embed == 'dvec':
                test_data = VadETDataset(data, 'embeddings')
            elif embed == 'xvec':
                test_data = VadETDatasetX(data, 'embeddings_xvec_l2')

            else: # ivec
                if 'l2' in model:
                    test_data = VadETDatasetI(data, 'embeddings_ivec_l2')
                else:
                    test_data = VadETDatasetI(data, 'embeddings_ivec')

        elif arch == 'set':
            test_data = VadSETDataset(data, 'embeddings', score_type)

        elif arch == 'st':
            test_data = VadSTDataset(data, score_type)
            
        else:
            # we should not get here
            continue

        test_loader = DataLoader(dataset=test_data, batch_size=batch_size_test, num_workers=2, shuffle=False, collate_fn=pad_collate)

        # set the device to cuda and move the model to the gpu
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        net = net.to(device)

        # mAP evaluation
        softmax = torch.nn.Softmax(dim=1)
        targets = []
        outputs = []

        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for x_padded, y_padded, x_lens, y_lens in test_loader:
                y_padded = y_padded.to(device)

                # pass the data through the model
                out_padded, _ = net(x_padded.to(device), x_lens, None)

                # value, index
                for j in range(out_padded.size(0)):
                    p = softmax(out_padded[j][:y_lens[j]])
                    
                    outputs.append(p.cpu().numpy())
                    targets.append(y_padded[j][:y_lens[j]].cpu().numpy())
                        
        targets = np.concatenate(targets)
        outputs = np.concatenate(outputs)

        # convert the target array to one hot
        targets_oh = np.eye(3)[targets]

        # and run the AP
        out_AP = average_precision_score(targets_oh, outputs, average=None)
        mAP = average_precision_score(targets_oh, outputs, average='micro')

        print('\n', model)
        print(out_AP) 
        print(f"mAP: {mAP}")

        # compute the confusion matrix
        classes = np.argmax(outputs, axis=1)
        cm = confusion_matrix(classes, targets, normalize='pred')

        print("confusion")
        print(cm)

        # compute the accuracy score
        acc = accuracy_score(classes, targets) * 100
        print(f"accuracy {quantize(acc, 2)}")

        print("\n=======================================\n")
