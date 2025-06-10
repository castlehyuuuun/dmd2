# A single unified model that wraps both the generator and discriminator
from DMD2_main.main.edm.edm_guidance import EDMGuidance
from torch import nn
import torch
import copy

class EDMUniModel(nn.Module):
    def __init__(self, args, accelerator):
        super().__init__()

        self.guidance_model = EDMGuidance(args, accelerator)

        self.guidance_min_step = self.guidance_model.min_step
        self.guidance_max_step = self.guidance_model.max_step

        if args.initialie_generator:
            # self.feedforward_model = copy.deepcopy(self.guidance_model.fake_unet) # 기존 코드
            self.feedforward_model = copy.deepcopy(self.guidance_model.fake_unet)
        else:
            raise NotImplementedError("Only support initializing generator from guidance model.")

        self.feedforward_model.requires_grad_(True)

        self.accelerator = accelerator
        self.num_train_timesteps = args.num_train_timesteps

    def forward(self, scaled_noisy_image,
                timestep_sigma, labels,
                real_train_dict=None,
                compute_generator_gradient=False,
                generator_turn=False,
                guidance_turn=False,
                guidance_data_dict=None,
                discriminator_turn=False,
                generator_data_dict=None
                ):
        assert sum([generator_turn, guidance_turn, discriminator_turn]) == 1

        if generator_turn:
            if not compute_generator_gradient:
                with torch.no_grad():
                    generated_image = self.feedforward_model(scaled_noisy_image, timestep_sigma, labels)
            else:
                generated_image = self.feedforward_model(scaled_noisy_image, timestep_sigma, labels)

            if compute_generator_gradient:
                generator_data_dict = {
                    "image": generated_image,
                    "label": labels,
                    "real_train_dict": real_train_dict  # ✅ 반드시 포함해야 함
                }

                self.guidance_model.requires_grad_(False)
                # ✅ 여기서 generator_forward()를 명확히 호출하면 더 직관적
                loss_dict, log_dict = self.guidance_model.generator_forward(
                    image=generated_image,
                    labels=labels,
                    real_train_dict=real_train_dict
                )
                self.guidance_model.requires_grad_(True)
            else:
                loss_dict = {}
                log_dict = {}

            log_dict['generated_image'] = generated_image.detach()
            log_dict['guidance_data_dict'] = {
                "image": generated_image.detach(),
                "label": labels.detach(),
                "real_train_dict": real_train_dict
            }

        elif guidance_turn:
            assert guidance_data_dict is not None
            loss_dict, log_dict = self.guidance_model(
                generator_turn=False,
                guidance_turn=True,
                guidance_data_dict=guidance_data_dict
            )
        elif discriminator_turn:
            assert generator_data_dict is not None
            loss_dict, log_dict = self.guidance_model(
                generator_turn=False,
                guidance_turn=False,
                discriminator_turn=True,
                generator_data_dict=generator_data_dict
            )

        return loss_dict, log_dict




