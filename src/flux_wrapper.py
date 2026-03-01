import torch
from torch import nn
import torch.nn.functional as F

from omegaconf import DictConfig

from diffusers import FluxPipeline


MEANs = (0.485, 0.456, 0.406)
STDs  = (0.229, 0.224, 0.225)

class FluxWrapper(nn.Module):
    def __init__(
            self, 
            model_id="black-forest-labs/FLUX.1-dev", 
            decode_steps=32, 
            guidance_scale=3.5, 
            image_size=256, 
            z_clamp=3.,
            optimize_z=True,
            optimize_c=True,
            seed_prompts=None,
            lr_z=7e-2,
            lr_c=2e-2
    ):
        super().__init__()
        self.decode_steps = decode_steps
        self.guidance_scale = guidance_scale
        self.image_size = image_size
        self.z_clamp = z_clamp
        self.optimize_z, self.optimize_c = optimize_z, optimize_c
        self.lr_z, self.lr_c = lr_z, lr_c
        self.seed_prompts = seed_prompts or []

        self._load_model(model_id)

        self.register_buffer("means", torch.tensor(MEANs).view(1, 3, 1, 1))
        self.register_buffer("stds", torch.tensor(STDs).view(1, 3, 1, 1))

    def _load_model(self, model_id):
        pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.vae = pipe.vae.cuda()
        self.transformer = pipe.transformer.cuda()
        self.clip_encoder = pipe.text_encoder
        self.t5_encoder = pipe.text_encoder_2
        self.clip_tokenizer = pipe.tokenizer
        self.t5_tokenizer = pipe.tokenizer_2
        self.scheduler = pipe.scheduler

        for module in [
            self.vae, self.transformer, self.clip_encoder, self.t5_encoder
        ]:
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)

        del pipe
        torch.cuda.empty_cache()
        
    @torch.no_grad()
    def _encode_prompts(self, prompts):
        t5_tokens = self.t5_tokenizer(
            prompts, 
            padding="max_length", 
            max_length=77, 
            truncation=True, 
            return_tensors="pt"
        ).input_ids.cuda()
        t5_embeds = self.t5_encoder.cuda()(t5_tokens).last_hidden_state.clone()
        self.t5_encoder.cpu()


        clip_tokens = self.clip_tokenizer(
            prompts, 
            padding="max_length", 
            max_length=512, 
            truncation=True, 
            return_tensors="pt"
        ).input_ids.cuda()
        
        clip_embeds = self.clip_encoder.cuda()(clip_tokens).pooler_output.clone()
        self.clip_encoder.cpu()

        torch.cuda.empty_cache()
        return t5_embeds, clip_embeds

    def init_latents(
            self,
            batch_size,
            seed=12,
            prompts=None
    ):
        h = w = self.image_size // 8
        torch.manual_seed(seed)
        z = torch.randn(batch_size, 16, h, w).cuda()

        if prompts is None:
            prompts = self.seed_prompts[:batch_size]

        assert len(prompts) == batch_size, "num prompts must match batch_size for diverse images"

        prompt_embeds, pooled_embeds = self._encode_prompts(prompts)


        # cond perturbation
        prompt_embeds = prompt_embeds + 1e-2 * torch.randn_like(prompt_embeds)
        pooled_embeds = pooled_embeds + 1e-2 * torch.randn_like(pooled_embeds)
        
        if self.optimize_z:
            z = z.detach().requires_grad_(True)
        if self.optimize_c:
            prompt_embeds = prompt_embeds.detach().requires_grad_(True)
            pooled_embeds = pooled_embeds.detach().requires_grad_(True)

        return z, prompt_embeds, pooled_embeds
    

    def decode(
            self, z, prompt_embeds, pooled_embeds
    ):
        B = z.shape[0]
        H, W = z.shape[2:]

        N = (H//2)*(W//2)
        # if getattr(getattr(self.scheduler, "config", None), "use_dynamic_shifting", False):
        mu = self._calculate_shift(N)
        self.scheduler.set_timesteps(self.decode_steps, device="cuda:0", mu=mu)
        # else:
        #     self.scheduler.set_timesteps(self.decode_steps, device)
        
        timesteps = self.scheduler.timesteps

        z = self._pack_latents(z, B, 16, H, W)
        sigma0 = self.scheduler.sigmas[0].cuda()
        latents = (z * sigma0).detach().bfloat16()
        
        img_ids = self._prepare_image_ids(H, W)
        txt_ids = torch.zeros(B, prompt_embeds.shape[1], 3).cuda()

        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            for t_ in timesteps:
                t = t_.expand(B)
                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=t / 1e3,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_embeds,
                    txt_ids=txt_ids,
                    img_ids=img_ids.unsqueeze(0).expand(B, -1, -1),
                    guidance=torch.full(
                        (B,), self.guidance_scale
                    ).cuda(),
                    return_dics=False
                )[0]
                latents = self.scheduler.step(noise_pred, t_, latents, return_dict=False)[0]


        # STE: strainght through estimator (for mem capacity)
        latents_ste = z + (latents - z).detach()
        if self.optimize_c:
            c_residual = prompt_embeds.norm(dim=-1).mean() - prompt_embeds.norm(dim=-1).mean().detach() * 0.0
            latents_ste = latents_ste + c_residual

        latents_ste = self._unpack_latents(latents_ste, H, W)
        latents_sc = (latents_ste / self.vae.config.scaling_factor) + self.vae.config.shift_factor



        with torch.no_grad, torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            images = self.vae.decode(latents_sc, return_dict=False)[0]

        images = (images.clamp(-1., 1.) + 1) / 2.
        images = (images - self.means.cuda()) / self.stds.cuda()
        return images, latents_ste

    @staticmethod
    def _compute_shift(
        image_seq_len, 
        base_seq_len=256, 
        max_seq_len=4096, 
        base_shift=.5, 
        max_shift=1.5
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        return image_seq_len * m + b


    @staticmethod
    def _pack_latents(latents, b, c, h, w):
        x = latents.view(b, c, h//2, 2, w//2, 2)
        x = x.permute(0, 2, 4, 1, 3, 5)
        return x.reshape(b, (h//2) * (w//2), c*4)
    
    @staticmethod
    def _prepare_image_ids(h, w):
        h2, w2 = h//2, w//2
        ids = torch.zeros(h2 * w2, 3).cuda()
        row_ids = torch.arange(h2).cuda().repeat_interleave(w2)
        col_ids = torch.arange(w2).cuda().repeat(h2)
        ids[:, 1] = row_ids
        ids[:, 2] = col_ids
        return ids
    
    @staticmethod
    def _unpack_latents(latents, h, w):
        b, _, d = latents.shape
        c = d // 4
        x = latents.view(b, h//2, w//2, c, 2, 2)
        x = x.permute(0, 3, 1, 4, 2, 5)
        return x.reshape(b, c, h, w)
    



def build_generator(config):
    config = config.flux
    return FluxWrapper(
        model_id=config.model_id,
        decode_steps=config.decode_steps,
        guidance_scale=config.guidance_scale,
        image_size=config.synthesis.get("image_size", 256),
        z_xlamp=config.z_clamp,
        optimize_z=config.optimize_z,
        optimize_c=config.optimize_c,
        seed_prompts=list(config.seed_prompts),
        lr_z=config.lr_z,
        lr_c=config.lr_c
    )