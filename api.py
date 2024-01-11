import io
import base64
import numpy as np
from pydantic import BaseModel, Field
import torch
import random
from diffusers import AutoencoderKL
from pipeline_semantic_stable_diffusion_img2img_solver import SemanticStableDiffusionImg2ImgPipeline_DPMSolver
from scheduling_dpmsolver_multistep_inject import DPMSolverMultistepSchedulerInject
from flask import Flask, request
from diffusers.utils import load_image
from PIL import Image

vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse', torch_dtype=torch.float16)
pipe = SemanticStableDiffusionImg2ImgPipeline_DPMSolver.from_pretrained(
    'stabilityai/stable-diffusion-2-1-base',
    vae=vae, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False).to('cuda')
pipe.scheduler = DPMSolverMultistepSchedulerInject.from_pretrained(
    'stabilityai/stable-diffusion-2-1-base',
    subfolder='scheduler', algorithm_type="sde-dpmsolver++", solver_order=2)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class EditData(BaseModel):
    concept: str
    guidance_scale: float = Field(default=7)
    warmup: int = Field(default=2)
    neg_guidance: bool = Field(default=False)
    threshold: float = Field(default=0.9)


def edit(
        input_image: np.ndarray,
        edit_datas: list[EditData],
        steps: int = 50,
        skip: int = 25,
        tar_prompt: str = '',
        tar_cfg_scale: float = 7.5,
        seed: int = 0,
        src_prompt: str = '',
        src_cfg_scale: float = 3.5,
        progress_callback=None
) -> Image.Image:
    seed_everything(seed)

    total_progress = steps * 4 + steps

    def invert_steps_callback(step, _1, _2):
        if progress_callback:
            progress_callback((step + 1) / total_progress)

    zs_tensor, wts_tensor = pipe.invert(
        image_path=input_image,
        source_prompt=src_prompt,
        source_guidance_scale=src_cfg_scale,
        num_inversion_steps=steps,
        skip=skip,
        eta=1.0,
        step_callback=invert_steps_callback
    )

    while len(edit_datas) < 3:
        edit_datas.append(EditData(
            concept='',
            guidance_scale=7,
            warmup=2,
            neg_guidance=False,
            threshold=0.95
        ))

    editing_args = dict(
        editing_prompt=[one.concept for one in edit_datas],
        reverse_editing_direction=[one.neg_guidance for one in edit_datas],
        edit_warmup_steps=[one.warmup for one in edit_datas],
        edit_guidance_scale=[one.guidance_scale for one in edit_datas],
        edit_threshold=[one.threshold for one in edit_datas],
        edit_momentum_scale=0,
        edit_mom_beta=0,
        eta=1,
        use_cross_attn_mask=False,
        use_intersect_mask=True
    )

    def pipe_steps_callback(step, _1, _2):
        if progress_callback:
            progress_callback((steps + (step + 1) * 4) / total_progress)

    latnets = wts_tensor[-1].expand(1, -1, -1, -1)
    sega_out, attention_store, text_cross_attention_maps = pipe(
        prompt=tar_prompt,
        init_latents=latnets,
        guidance_scale=tar_cfg_scale,
        zs=zs_tensor, attention_store=None,
        text_cross_attention_maps=None,
        callback=pipe_steps_callback,
        callback_steps=1,
        **editing_args
    )

    return sega_out.images[0]


app = Flask(__name__)

global_dict = {
    'progress': 0
}


@app.get('/ping')
def ping():
    return 'pong'


@app.post('/invoke')
def invoke():
    global_dict['progress'] = 0

    data = request.get_json()

    img_data = data['image']
    edit_datas = [
        EditData.model_validate(one)
        for one in data['edits']
    ]
    del data['image']
    del data['edits']

    if img_data.startswith('https://'):
        image = load_image(img_data)
    else:
        img_bytes = base64.b64decode(img_data.encode())
        image = Image.open(BytesIO(img_bytes))
        image = load_image(image)

    image = image.resize((512, 512))

    if len(edit_datas) == 0:
        buffer = io.BytesIO()
        image.save(buffer, format='WEBP')
        return {
            'image': base64.b64encode(buffer.getvalue()).decode()
        }

    image = np.array(image)

    def progress_callback(p):
        global_dict['progress'] = p

    result = edit(
        image, edit_datas,
        progress_callback=progress_callback,
        **data
    )

    buffer = io.BytesIO()
    result.save(buffer, format='WEBP')
    return {
        'image': base64.b64encode(buffer.getvalue()).decode()
    }


@app.get('/progress')
def progress():
    return {
        'progress': global_dict['progress']
    }
