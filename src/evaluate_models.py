import numpy as np
from copy import copy
import torch

from personal_vad import PersonalVAD, pad_collate
from vad_et import VadETDatasetArk
from vad_set import VadSETDatasetArk
from vad_st import VadSTDatasetArk
from vad_xvector import VadETDatasetArkX
from vad_ivector import VadETDatasetArkI
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, accuracy_score
import kaldiio
import os
import sys
import ml_metrics as metrics
from glob import glob
from decimal import Decimal, ROUND_HALF_UP

def quantize(number, prec):
    prec = f'1e-{prec}'
    return str(Decimal(number).quantize(Decimal(prec), ROUND_HALF_UP))

batch_size_test = 64
test_dir = '/home/pirx/Devel/bp/personalVAD/data/train_dir/'
os.chdir(test_dir)

models = glob('models/final/*pt')
data = 'data/test'


for model in models:
    embed = 'dvec'
    if 'ivec' not in model and 'xvec' not in model:
        continue

    if 'ivec' in model:
        embed = 'ivec'
    elif 'xvec' in model:
        embed = 'xvec'
    else:
        embed = 'dvec'


    # determine the architecture
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
        print(model, "other architecture...")
        continue

    # determine score type
    if arch in ['set', 'st']:
        if 'score0' in model: score_type = 0
        elif 'score1' in model: score_type = 1
        elif 'score2' in model: score_type = 2
        else:
            print(model, "other architecture...")
            continue

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
            continue
        else:
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


    # load the model
    net = PersonalVAD(input_dim=input_dim, hidden_dim=64, num_layers=2, out_dim=3, use_fc=use_fc, linear=linear)
    net.load_state_dict(torch.load(model))

    if arch == 'et':
        if embed == 'dvec':
            test_data = VadETDatasetArk(data, 'embeddings')
        elif embed == 'xvec':
            test_data = VadETDatasetArkX(data, 'embeddings_xvec_l2_non_negative')
        else: # ivec
            if 'l2' in model:
                test_data = VadETDatasetArkI(data, 'l2_normed')
            else:
                test_data = VadETDatasetArkI(data, 'regular')

    elif arch == 'set':
        test_data = VadSETDatasetArk(data, 'embeddings', score_type)

    elif arch == 'st':
        test_data = VadSTDatasetArk(data, score_type)
            
    elif arch == 'xv':
        model = PersonalVAD(input_dim=552, hidden_dim=64, num_layers=2, out_dim=3)
        model.load_state_dict(torch.load('models/vad_et_7ep_xvec.pt'))
        test_data = VadETDatasetArkX(data, 'embeddings_xvec')
        
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

    classes = np.argmax(outputs, axis=1)
    cm = confusion_matrix(classes, targets, normalize='pred')

    print("confusion")
    print(cm)

    #prec = precision_score(classes, targets, average='micro')
    #print(f"micro precision {prec}")

    acc = accuracy_score(classes, targets) * 100
    print(f"accuracy {quantize(acc, 2)}")

    print("\n=======================================\n")
