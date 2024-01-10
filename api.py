import numpy as np
from pydantic import BaseModel, Field
import torch
import random
from diffusers import AutoencoderKL
from pipeline_semantic_stable_diffusion_img2img_solver import SemanticStableDiffusionImg2ImgPipeline_DPMSolver
from scheduling_dpmsolver_multistep_inject import DPMSolverMultistepSchedulerInject

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
    guidance_scale: float
    warmup: int
    neg_guidance: bool
    threshold: float


def edit(
        input_image: np.ndarray,
        edit_datas: list[EditData],
        steps: int = 50,
        skip: int = 25,
        tar_prompt: str = '',
        tar_cfg_scale: float = 7.5,
        seed: int = 0,
        src_prompt: str = '',
        src_cfg_scale: float = 3.5
):
    seed_everything(seed)

    zs_tensor, wts_tensor = pipe.invert(
        image_path=input_image,
        source_prompt=src_prompt,
        source_guidance_scale=src_cfg_scale,
        num_inversion_steps=steps,
        skip=skip,
        eta=1.0,
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

    latnets = wts_tensor[-1].expand(1, -1, -1, -1)
    sega_out, attention_store, text_cross_attention_maps = pipe(
        prompt=tar_prompt,
        init_latents=latnets,
        guidance_scale=tar_cfg_scale,
        zs=zs_tensor, attention_store=None,
        text_cross_attention_maps=None,
        **editing_args
    )

    return sega_out.images[0]
