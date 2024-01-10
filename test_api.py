from unittest import TestCase
import api
import numpy
from PIL import Image


class Test(TestCase):
    def test_edit(self):
        img = Image.open('examples/lemons_input.jpg')
        img = img.resize((512, 512))
        img = numpy.array(img)

        result = api.edit(img, [
            api.EditData(
                concept='apple',
                guidance_scale=7,
                warmup=2,
                neg_guidance=False,
                threshold=0.95
            )
        ])

        Image.fromarray(result).save('examples/lemons_output_1.jpg')
