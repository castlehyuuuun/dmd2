from DMD2_main.main.edm.edm_network import get_edm_network
import torch.nn.functional as F
import torch.nn as nn
import dnnlib
import pickle
import torch
import copy
import os


import pickle
import torch
import dnnlib
import torch_utils
def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0):
    # from https://github.com/crowsonkb/k-diffusion
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas


class EDMGuidance(nn.Module):
    def __init__(self, args, accelerator):
        super().__init__()
        self.args = args
        self.accelerator = accelerator

        if os.path.exists(args.model_id):
            # 로컬 경로인 경우
            with open(args.model_id, 'rb') as f:
                temp_edm = pickle.load(f)['ema']
        else:
            # URL 경로인 경우 fallback
            with dnnlib.util.open_url(args.model_id) as f:
                temp_edm = pickle.load(f)['ema']

        # initialize the real unet
        self.real_unet = get_edm_network(args)  # real data로 학습된 edm network / 기존 코드

        # self.real_unet.load_state_dict(temp_edm.state_dict(), strict=True) # 기존 코드
        self.real_unet.load_state_dict(temp_edm.state_dict(), strict=False)
        self.real_unet.requires_grad_(False)
        del self.real_unet.model.map_augment
        self.real_unet.model.map_augment = None

        self.fake_unet = copy.deepcopy(self.real_unet)  # / 추가한것
        self.fake_unet.requires_grad_(True)

        # some training hyper-parameters
        self.sigma_data = args.sigma_data
        self.sigma_max = args.sigma_max
        self.sigma_min = args.sigma_min
        self.rho = args.rho

        self.gan_classifier = args.gan_classifier
        self.diffusion_gan = args.diffusion_gan
        self.diffusion_gan_max_timestep = args.diffusion_gan_max_timestep

        if self.gan_classifier:
            self.cls_pred_branch = nn.Sequential(
                nn.Conv2d(kernel_size=4, in_channels=256, out_channels=768, stride=2, padding=1),  # 8x8 -> 4x4
                nn.GroupNorm(num_groups=32, num_channels=768),
                nn.SiLU(),
                nn.Conv2d(kernel_size=4, in_channels=768, out_channels=768, stride=4, padding=0),  # 4x4 -> 1x1
                nn.GroupNorm(num_groups=32, num_channels=768),
                nn.SiLU(),
                nn.Conv2d(kernel_size=1, in_channels=768, out_channels=1, stride=1, padding=0),  # 1x1 -> 1x1
            )
            self.cls_pred_branch.requires_grad_(True)

        # if self.gan_classifier:
        #     self.cls_pred_branch = nn.Sequential(
        #         nn.Conv2d(kernel_size=4, in_channels=256, out_channels=256, stride=2, padding=1),  # 8x8 -> 4x4
        #         nn.GroupNorm(num_groups=32, num_channels=256),
        #         nn.SiLU(),
        #         nn.Conv2d(kernel_size=4, in_channels=256, out_channels=256, stride=4, padding=0),  # 4x4 -> 1x1
        #         nn.GroupNorm(num_groups=32, num_channels=256),
        #         nn.SiLU(),
        #         nn.Conv2d(kernel_size=1, in_channels=256, out_channels=1, stride=1, padding=0),  # 1x1 -> 1x1
        #     )
        #     self.cls_pred_branch.requires_grad_(True)

        self.num_train_timesteps = args.num_train_timesteps
        # small sigma first, large sigma later
        karras_sigmas = torch.flip(
            get_sigmas_karras(self.num_train_timesteps, sigma_max=self.sigma_max, sigma_min=self.sigma_min,
                              rho=self.rho
                              ),
            dims=[0]
        )  # karras sigma schedule을 생성하여 diffusion noise schedule을 정의
        self.register_buffer("karras_sigmas", karras_sigmas)

        self.min_step = int(args.min_step_percent * self.num_train_timesteps)
        self.max_step = int(args.max_step_percent * self.num_train_timesteps)
        del temp_edm

    def forward_discriminator(self, real_images, g_few_clean, g_few_noisy, labels):
        with torch.no_grad():
            timesteps = torch.zeros(real_images.shape[0], dtype=torch.long, device=real_images.device)
            timestep_sigma = self.karras_sigmas[timesteps]

            rep_real = self.fake_unet(real_images, timestep_sigma, labels, return_bottleneck=True)
            rep_clean = self.fake_unet(g_few_clean, timestep_sigma, labels, return_bottleneck=True)
            rep_noisy = self.fake_unet(g_few_noisy, timestep_sigma, labels, return_bottleneck=True)

        real_logits = self.cls_pred_branch(rep_real)
        clean_logits = self.cls_pred_branch(rep_clean)
        noisy_logits = self.cls_pred_branch(rep_noisy)

        loss_real = F.softplus(-real_logits).mean()
        loss_clean = F.softplus(clean_logits).mean()
        loss_noisy = F.softplus(noisy_logits).mean()

        loss_total = loss_real + loss_clean + loss_noisy
        loss_total = 0.05 * loss_total
        return loss_total, {
            "D_real": loss_real.item(),
            "D_clean": loss_clean.item(),
            "D_noisy": loss_noisy.item(),

        }

    def compute_distribution_matching_loss(
            self,
            latents,
            labels
    ):  # generator가 만든 fake image와 real image의 분포 차이를 비교
        # distribution matchin gloss를 계산하여 generator를 개선
        # real unet과 fake unet의 출력을 비교하여 gradient 생성
        original_latents = latents
        batch_size = latents.shape[0]

        with torch.no_grad():
            timesteps = torch.randint(
                self.min_step,
                min(self.max_step + 1, self.num_train_timesteps),
                [batch_size, 1, 1, 1],
                device=latents.device,
                dtype=torch.long
            )  # random timestep을 선택하고, 해당 시점에서 noise 추가

            noise = torch.randn_like(latents)

            timestep_sigma = self.karras_sigmas[timesteps]

            noisy_latents = latents + timestep_sigma.reshape(-1, 1, 1, 1) * noise

            pred_real_image = self.real_unet(noisy_latents, timestep_sigma, labels)

            pred_fake_image = self.fake_unet(
                noisy_latents, timestep_sigma, labels
            )  # real unet과 fake unet의 예측값을 계산

            p_real = (latents - pred_real_image)
            p_fake = (latents - pred_fake_image)

            weight_factor = torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True)
            grad = (p_real - p_fake) / weight_factor

            grad = torch.nan_to_num(grad)

            # this loss gives the grad as gradient through autodiff, following https://github.com/ashawkey/stable-dreamfusion
        loss = 0.5 * F.mse_loss(original_latents, (original_latents - grad).detach(), reduction="mean")
        # 109 ~ 114 / real image와 fake image의 차이(gradient)를 기반으로 loss 계산
        loss_dict = {
            "loss_dm": loss
        }

        dm_log_dict = {
            "dmtrain_noisy_latents": noisy_latents.detach(),
            "dmtrain_pred_real_image": pred_real_image.detach(),
            "dmtrain_pred_fake_image": pred_fake_image.detach(),
            "dmtrain_grad": grad.detach(),
            "dmtrain_gradient_norm": torch.norm(grad).item(),
            "dmtrain_timesteps": timesteps.detach(),
        }
        return loss_dict, dm_log_dict

    def compute_loss_fake(
            self,
            latents,
            labels,
    ):
        # for Guidance, not Generator
        # fake image에 대한 diffusion loss 계산
        # generator가 만든 fake image가 real image처럼 보이도록 유도
        batch_size = latents.shape[0]

        latents = latents.detach()  # no gradient to generator, Generator makes image G(z)

        noise = torch.randn_like(latents)  # ε ~ N(0, I)

        timesteps = torch.randint(
            0,
            self.num_train_timesteps,
            [batch_size, 1, 1, 1],
            device=latents.device,
            dtype=torch.long
        )
        timestep_sigma = self.karras_sigmas[timesteps]
        noisy_latents = latents + timestep_sigma.reshape(-1, 1, 1, 1) * noise  # xₜ = x₀ + σₜ * ε

        fake_x0_pred = self.fake_unet(
            noisy_latents, timestep_sigma, labels
        )
        # noise를 추가한 후 fake unet을 통해 이미지 예측

        snrs = timestep_sigma ** -2  # SNR (Signal-to-Noise Ratio)를 기반으로 timestep별 중요도를 조절하는 weight 계산

        # weight_schedule karras
        weights = snrs + 1.0 / self.sigma_data ** 2  # sigma가 작을수록 더 신뢰할 수 있는 정보이므로 더 높은 weight

        target = latents

        loss_fake = torch.mean(
            weights * (fake_x0_pred - target) ** 2
        )  # 예측된 x₀ (fake_x0_pred)와 원래 이미지 (target=latents) 간의 거리 측정
        # 거기에 SNR 기반 weight를 곱해서 timestep별 영향을 반영

        loss_dict = {
            "loss_fake_mean": loss_fake
        }

        fake_log_dict = {
            "faketrain_latents": latents.detach(),
            "faketrain_noisy_latents": noisy_latents.detach(),
            "faketrain_x0_pred": fake_x0_pred.detach()
        }
        return loss_dict, fake_log_dict

    def compute_cls_logits(self, image, label):
        if self.diffusion_gan:
            timesteps = torch.randint(
                0, self.diffusion_gan_max_timestep, [image.shape[0]], device=image.device, dtype=torch.long
            )
            timestep_sigma = self.karras_sigmas[timesteps]
            image = image + timestep_sigma.reshape(-1, 1, 1, 1) * torch.randn_like(image)
        else:
            timesteps = torch.zeros([image.shape[0]], dtype=torch.long, device=image.device)
            timestep_sigma = self.karras_sigmas[timesteps]

        rep = self.fake_unet(
            image, timestep_sigma, label, return_bottleneck=True
        ).float()

        logits = self.cls_pred_branch(rep).squeeze(dim=[2, 3])
        # gan discriminator(분류기) model을 이용해 fake vs real 분류
        # guidance 모델이 generator가 만든 image가 fake인지 real인지 판별
        # 이미지를 cnn 기반 분류기에 입력하여 fake vs real 점수 출력
        return logits

    def compute_generator_clean_cls_loss(self, fake_image, fake_labels):
        loss_dict = {}
        # generator가 만든 fake image가 real 처럼 보이도록 gan loss 적용
        pred_realism_on_fake_with_grad = self.compute_cls_logits(
            image=fake_image,
            label=fake_labels
        )
        loss_dict["gen_cls_loss"] = F.softplus(-pred_realism_on_fake_with_grad).mean()
        # 206 ~ 210 / gan 분류기가 fake이미지를 'real'로 인식하도록 학습
        return loss_dict

    def compute_guidance_clean_cls_loss(self, real_image, fake_image, real_label, fake_label):
        pred_realism_on_real = self.compute_cls_logits(
            real_image.detach(), real_label,
        )  # guidance 모델이 real vs fake를 구분하도록 학습
        # gan discriminator가 real image와 fake image의 차이를 극대화하도록 학습
        pred_realism_on_fake = self.compute_cls_logits(
            fake_image.detach(), fake_label,
        )
        classification_loss = F.softplus(pred_realism_on_fake) + F.softplus(-pred_realism_on_real)
        # fake image는 낮은 점수를, real image는 높은 점수를 받도록 loss 계산
        log_dict = {
            "pred_realism_on_real": torch.sigmoid(pred_realism_on_real).squeeze(dim=1).detach(),
            "pred_realism_on_fake": torch.sigmoid(pred_realism_on_fake).squeeze(dim=1).detach()
        }

        loss_dict = {
            "guidance_cls_loss": classification_loss.mean()
        }
        return loss_dict, log_dict

    def generator_forward(
            self,
            image,
            labels,
            real_train_dict=None
    ):
        loss_dict = {}
        log_dict = {}
        # generator의 loss를 계산하고 학습 수행
        # distribution matching loss + gan loss 적용
        # image.requires_grad_(True) <- 원래 있던것
        dm_dict, dm_log_dict = self.compute_distribution_matching_loss(image, labels)

        loss_dict.update(dm_dict)
        # distribution matching loss를 계산하여 generator 학습
        log_dict.update(dm_log_dict)
        if real_train_dict is not None:
            pass  # 필요한 로직이 있다면 여기에 추가

        if self.gan_classifier:
            clean_cls_loss_dict = self.compute_generator_clean_cls_loss(image, labels)
            loss_dict.update(clean_cls_loss_dict)
        # gan loss를 적용하여 generator의 출력 개선 <- 내가 쓴 것
        # loss_dm = loss_dict["loss_dm"]
        # gen_cls_loss = loss_dict["gen_cls_loss"]

        # grad_dm = torch.autograd.grad(loss_dm, image, retain_graph=True)[0]
        # grad_cls = torch.autograd.grad(gen_cls_loss, image, retain_graph=True)[0]

        # print(f"dm {grad_dm.abs().mean()} cls {grad_cls.abs().mean()}")

        return loss_dict, log_dict

    def guidance_forward(
            self,
            image,
            labels,
            real_train_dict=None
    ):  # guidance model(real score function)이 fake vs real을 학습하도록 loss 계
        fake_dict, fake_log_dict = self.compute_loss_fake(
            image, labels
        )

        loss_dict = fake_dict
        log_dict = fake_log_dict
        # fake image의 loss 계산하여 guidance model 학습
        if self.gan_classifier:
            clean_cls_loss_dict, clean_cls_log_dict = self.compute_guidance_clean_cls_loss(
                real_image=real_train_dict['real_image'],
                fake_image=image,
                real_label=real_train_dict['real_label'],
                fake_label=labels
            )
            loss_dict.update(clean_cls_loss_dict)
            log_dict.update(clean_cls_log_dict)
        return loss_dict, log_dict

    def forward(
            self,
            generator_turn=False,
            guidance_turn=False,
            generator_data_dict=None,
            guidance_data_dict=None,
            discriminator_turn = False

    ):
        if generator_turn:
            loss_dict, log_dict = self.generator_forward(
                image=generator_data_dict['image'],
                labels=generator_data_dict['label']
            )
        elif guidance_turn:
            loss_dict, log_dict = self.guidance_forward(
                image=guidance_data_dict['image'],
                labels=guidance_data_dict['label'],
                real_train_dict=guidance_data_dict['real_train_dict']
            )
        elif discriminator_turn:
            return self.forward_discriminator(real_images = generator_data_dict['real_image'],
            g_few_clean = generator_data_dict['g_few_clean'],
            g_few_noisy = generator_data_dict['g_few_noisy'],
            labels = generator_data_dict['label'])

        else:
            raise NotImplementedError

            # generator 학습 (generator_turn=True) -> fake image 생성
        # guidance 학습 (guidance-trun=True) -> fake image 평가 및 수정

        return loss_dict, log_dict 