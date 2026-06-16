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

class CanvasOptimizer:
    def __init__(self, w, h):
        self.c = DiffCanvas(w, h)
        self.optimizers = []
        self.schedulers = []
        self.draw(self.c)
        self.running = False
        self.epoch = 0
        self.num_opt_steps = 0
        
    ######################################
    # Functions for user to override
    def setup(self, c):
        # Setup optimizers etc
        pass

    def release(self):
        pass
    
    def draw(self, c):
        # Do drawing operations on 
        pass
    
    def loss(self, img):
        pass

    def postprocess(self, c):
        # Optional clamp values or other procedures after opt step
        pass
    
    ######################################
    # Built in

    def step(self):
        # Peform an optimization step if optimizing
        if not self.running:
            return

        for opt in self.optimizers:
            opt.zero_grad()

        img = self.draw(self.c)

        if self.epoch >= self.num_opt_steps:
            print("Stopping", self.epoch)
            self.running = False
            return
        
        loss = self.loss(img)
        loss.backward()
        for opt in self.optimizers:
            opt.step()
        for sched in self.schedulers:
            sched.step()
        self.postprocess(self.c)
        self.epoch += 1
        
        
    def run(self, num_steps):
        self.optimizers = []
        self.schedulers = []
        self.epoch = 0
        self.num_opt_steps = num_steps
        self.setup(self.c)
        self.running = True
        
    def get_image(self):
        return self.c.get_image()
    
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
