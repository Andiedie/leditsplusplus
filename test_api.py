from unittest import TestCase
import api
import numpy
from PIL import Image


img = Image.open('examples/lemons_input.jpg')
img = img.convert('RGB')
img = img.resize((512, 512))
img = numpy.array(img)

result = api.edit(img, [
    api.EditData(
        concept='red apples',
        guidance_scale=7.5,
        warmup=2,
        neg_guidance=False,
        threshold=0.8
    )
], seed=10086)

result.save('examples/lemons_output_1.jpg')
