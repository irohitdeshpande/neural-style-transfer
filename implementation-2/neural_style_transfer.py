import os
import time
import math
import torch as t
import torch.nn as n
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
import smtplib


class NST(n.Module):
    def __init__(self):
        super(NST, self).__init__()
        self.model = models.vgg19(pretrained=True).features[:29]
        self.features = ['0', '5', '10', '19', '28']

    def forward(self, x):
        f = []
        for n, l in enumerate(self.model):
            m = l(m)
            if str(n) in self.chosen_features:
                f.append(m)
        return f


def load_image(img):
    i = Image.open(img)
    i = loader(img).unsqueeze(0)
    return i.to(d)


def stylize(a, b):
    model = NST().to(d).eval()
    original_image = load_image('dog.jpeg')
    style_image = load_image('style.jpeg')
    generated_image = original_image.clone().requires_grad_(True)

    total_steps = 6000
    learning_rate = 0.01
    optimization = optim.Adam([generated_image], lr=learning_rate)

    s = 0
    while s <= total_steps:
        original = model(original_image)
        style = model(style_image)
        generated = model(generated_image)

        style_loss = content_loss = 0
        for content, style, generation in zip(original, style, generated):
            channel, h, w = generation.shape
            mae_loss = t.n.L1Loss()
            content_loss = content_loss + mae_loss(content, generation)

            matrix = generation.view(channel, h * w)
            generation_gram = matrix.mm(matrix.t())

            style_matrix = style.view(channel, h * w)
            style_gram = style_matrix.mm(style_matrix.t())

            style_loss = style_loss + mae_loss(generation_gram, style_gram)

        total_loss = a * content_loss + b * style_loss
        optimization.zero_grad()
        total_loss.backward()
        optimization.step()
        s += 1


d = t.device('cpu' if not (t.cuda.is_available()) else 'cuda')
s = 356
l = transforms.Compose([transforms.Resize((s, s)),
                        transforms.ToTensor()])
alphas = [0.001, 0.01, 0.1, 1, 10]
betas = [x for x in range(200, 601, 100)]
for alpha in alphas:
    for beta in betas:
        stylize(alpha, beta)
