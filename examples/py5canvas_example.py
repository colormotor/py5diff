#!/usr/bin/env python3
from py5canvas import *
from slimgui import imgui, implot

from importlib import reload
import py5diff
from py5diff import diff_canvas as dc
reload(dc)
reload(py5diff)

import torch
import matplotlib.pyplot as plt

from py5diff import DiffCanvas
from PIL import Image
import numpy as np

target_img = Image.open('./spock256.jpg')
w, h = target_img.size

    
class MyCanvasOpt(CanvasOptimizer):
    def draw(self, c):
        c.background(1.0)
        
        c.no_stroke()
        c.rect_mode('center')
        n = 20
        for y in np.linspace(0, c.height, n+2)[1:-1]:
            for x in np.linspace(0, c.width, n+2)[1:-1]:
                c.fill(c.var(0.5, 'color'))
                c.square([x, y], c.var(c.width/n-5, 'size'))
                #c.circle([x,y],
                #        c.var(c.width/n, 'size') # Optimize circle radius
                #        )
        c.render(prefiltering=True)
        return c.img

    def postprocess(self, c):
        with torch.no_grad():
            for v in c.get_vars('size'):
                v.data.clamp_(2, 15)
            for v in c.get_vars('color'):
                v.data.clamp_(0, 1)
                
    def setup(self, c):
        self.optimizers = [torch.optim.Adam(c.get_vars('size'), lr=1.0),
                           torch.optim.Adam(c.get_vars('color'), lr=0.1),]
        self.target = self.c._to(np.array(target_img.convert('L'))/255)
        
    def loss(self, img):
        loss = (img.mean(dim=-1) - self.target).pow(2).mean()
        return loss

opt = MyCanvasOpt(w, h)

def parameters():
    return {'foo':False}

def setup():
    create_canvas(w, h)
    color_mode('rgb', 1.0)
    
def gui():
    if imgui.button('Run'):
        opt.run(100)
    pass

def draw():
    background(0)
    opt.step()
    image(opt.get_image())
    
run()
