"""Convert a single digit into MNIST-like, centered light ink on dark background."""
import numpy as np
from PIL import Image, ImageOps
import torch


def prepare_image(path, polarity='auto'):
    with Image.open(path) as source:
        source = ImageOps.exif_transpose(source)
        if 'A' in source.getbands() or 'transparency' in source.info:
            rgba = source.convert('RGBA')
            source = Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba)
        gray = ImageOps.grayscale(source)
    pixels = np.array(gray, dtype=np.float32) / 255
    border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    if polarity == 'dark' or (polarity == 'auto' and np.median(border) > .5):
        pixels = 1 - pixels
    border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    pixels = np.maximum(pixels - np.median(border), 0)
    if pixels.max() < .05:
        raise ValueError('The image is blank or has too little foreground contrast.')
    pixels /= pixels.max()
    rows, cols = np.nonzero(pixels > .15)
    crop = Image.fromarray((pixels[rows.min():rows.max()+1,
                                   cols.min():cols.max()+1] * 255).astype('uint8'))
    crop.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new('L', (28, 28))
    canvas.paste(crop, ((28-crop.width)//2, (28-crop.height)//2))
    arr = np.asarray(canvas, dtype=np.float32) / 255
    yy, xx = np.indices(arr.shape)
    dx = int(round(13.5 - float((xx * arr).sum() / arr.sum())))
    dy = int(round(13.5 - float((yy * arr).sum() / arr.sum())))
    shifted = Image.new('L', (28, 28))
    shifted.paste(canvas, (dx, dy))
    arr = np.array(shifted, dtype=np.float32) / 255
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).sub(.5).div(.5)


def explain(model, image):
    """Real forward pass and gradient of the predicted logit wrt input."""
    x = image.detach().clone().requires_grad_(True)
    logits, activations = model(x, trace=True)
    prediction = int(logits.argmax(1).item())
    gradient, = torch.autograd.grad(logits[0, prediction], x)
    contributions = activations['hidden'][0] * model.fc2.weight[prediction]
    return {
        'prediction': prediction,
        'probabilities': logits.softmax(1)[0].detach().cpu().numpy(),
        'activations': {k: v[0].detach().cpu().numpy() for k, v in activations.items()},
        'saliency': gradient[0, 0].detach().abs().cpu().numpy(),
        'contributions': contributions.detach().cpu().numpy(),
        'bias': float(model.fc2.bias[prediction].detach()),
        'logit': float(logits[0, prediction].detach()),
    }
