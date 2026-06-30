# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 11:37:17 2026

@author: WKS
"""

from src.train import train

def test_accuracy():
    _, acc = train()
    assert acc > 0.9
    
# def test_accuracy():
#     _, acc = train()
#     assert acc > 1.5