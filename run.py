#!/usr/bin/env python

import sys
import json

sys.path.insert(0, 'src/data')
sys.path.insert(0, 'src/model')

from etl import get_data
import numpy as np
import pytest
import torch
from PIL import Image

import clip


def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'analysis', 'model'. 
    
    '''

    if 'all' in targets:
        return 

    if 'test' in targets:
        model_name = 'ViT-B/32'
        device = "cpu"
        jit_model, transform = clip.load(model_name, device=device, jit=True)
        py_model, _ = clip.load(model_name, device=device, jit=False)

        image = transform(Image.open("CLIP.png")).unsqueeze(0).to(device)
        text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

        with torch.no_grad():
            logits_per_image, _ = jit_model(image, text)
            jit_probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            logits_per_image, _ = py_model(image, text)
            py_probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        assert np.allclose(jit_probs, py_probs, atol=0.01, rtol=0.1)

    return


if __name__ == '__main__':
    # run via:
    # python main.py data model
    targets = sys.argv[1:]
    main(targets)