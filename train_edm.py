import matplotlib
from numpy.ma.core import true_divide

matplotlib.use('Agg')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import dnnlib
import pickle
from DMD2_main.main.utils import prepare_images_for_saving, draw_valued_array, cycle, draw_probability_histogram
from accelerate.utils import ProjectConfiguration
from diffusers.optimization import get_scheduler
# from diffusion.DMD2_main.main.data.lmdb_dataset import LMDBDataset
from DMD2_main.main.edm.edm_unified_model import EDMUniModel
from accelerate.utils import set_seed
from accelerate import Accelerator
import argparse
import shutil
import wandb
import torch
import time
import os
import cv2
from ts import *



class Trainer:
    def __init__(self, args):

        self.args = args

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        accelerator_project_config = ProjectConfiguration(logging_dir=args.output_path)

        accelerator = Accelerator(
            gradient_accumulation_steps=1,  # no accumulation
            mixed_precision="no",
            log_with="wandb",
            project_config=accelerator_project_config,
            kwargs_handlers=None
        )
        set_seed(args.seed + accelerator.process_index)

        print(accelerator.state)

        if accelerator.is_main_process:
            output_path = os.path.join(args.output_path, f"time_{int(time.time())}_seed{args.seed}")
            os.makedirs(output_path, exist_ok=False)
            self.output_path = output_path

            if args.cache_dir != "":
                self.cache_dir = os.path.join(args.cache_dir, f"time_{int(time.time())}_seed{args.seed}")
                os.makedirs(self.cache_dir, exist_ok=False)


        self.model = EDMUniModel(args, accelerator)
        self.dataset_name = args.dataset_name
        self.real_image_path = args.real_image_path

        self.dfake_gen_update_ratio = args.dfake_gen_update_ratio
        self.num_train_timesteps = args.num_train_timesteps

        self.cls_loss_weight = args.cls_loss_weight

        self.gan_classifier = args.gan_classifier
        self.gen_cls_loss_weight = args.gen_cls_loss_weight
        self.no_save = args.no_save
        self.previous_time = None
        self.step = 0
        self.cache_checkpoints = (args.cache_dir != "")
        self.max_checkpoint = args.max_checkpoint

        if args.ckpt_only_path is not None:
            if accelerator.is_main_process:
                print(f"loading checkpoints without optimizer states from {args.ckpt_only_path}")
            # state_dict = torch.load(args.ckpt_only_path, map_location="cpu")
            # print(self.model.load_state_dict(state_dict, strict=False))
            generator_path = os.path.join(args.ckpt_only_path, "pytorch_model.bin")
            guidance_path = os.path.join(args.ckpt_only_path, "pytorch_model_1.bin")

            generator_state_dict = torch.load(generator_path, map_location="cpu")
            guidance_state_dict = torch.load(guidance_path, map_location="cpu")

            print(self.model.feedforward_model.load_state_dict(generator_state_dict, strict=False))
            print(self.model.guidance_model.load_state_dict(guidance_state_dict, strict=False))

            self.step = int(args.ckpt_only_path.replace("/", "").split("_")[-1])

        if args.generator_ckpt_path is not None:
            if accelerator.is_main_process:
                print(f"loading generator checkpoints from {args.generator_ckpt_path}")
            generator_path = os.path.join(args.generator_ckpt_path, "pytorch_model.bin")
            print(self.model.feedforward_model.load_state_dict(torch.load(generator_path, map_location="cpu"),
                                                               strict=True))

        # also load the training dataset images, this will be useful for GAN loss
        # real_dataset = LMDBDataset(args.real_image_path)
        # real_dataset = "/dev/hdd/skh/DMD2-main/experiments/cifar/cifar10/cifar10_train/"

        dataset_path = "C:/Users/user/Desktop/diffusion/DMD2_main/cifar10/"
        real_dataset = ImageDataset(dataset_path)

        real_image_dataloader = torch.utils.data.DataLoader(
            real_dataset, batch_size=args.batch_size, shuffle=True,
            drop_last=True, num_workers=args.num_workers
        )

        real_image_dataloader = accelerator.prepare(real_image_dataloader)
        self.real_image_dataloader = cycle(real_image_dataloader)

        self.optimizer_guidance = torch.optim.AdamW(
            [param for param in self.model.guidance_model.parameters() if param.requires_grad],
            lr=args.guidance_lr,
            betas=(0.9, 0.999),  # pytorch's default
            weight_decay=0.01  # pytorch's default
        )
        self.optimizer_generator = torch.optim.AdamW(
            [param for param in self.model.feedforward_model.parameters() if param.requires_grad],
            lr=args.generator_lr,
            betas=(0.9, 0.999),  # pytorch's default
            weight_decay=0.01  # pytorch's default
        )

        # actually this scheduler is not very useful (it warms up from 0 to max_lr in 500 / num_gpu steps), but we keep it here for consistency
        self.scheduler_guidance = get_scheduler(
            "constant_with_warmup",
            optimizer=self.optimizer_guidance,
            num_warmup_steps=args.warmup_step,
            num_training_steps=args.train_iters
        )

        self.scheduler_generator = get_scheduler(
            "constant_with_warmup",
            optimizer=self.optimizer_generator,
            num_warmup_steps=args.warmup_step,
            num_training_steps=args.train_iters
        )

        # the self.model is not wrapped in ddp, only its two subnetworks are wrapped
        # DistributedDataParallel (DDP)
        # self.model = EDMUniModel -> (self.model.feedforward_model : Generator + self.model.guidance_model : Discriminator)
        # Consequently, self.model 전체를 DDP로 래핑하지 않고,
        # 그 안의 feedforward_model과 guidance_model만 DDP로 래핑
        (
            self.model.feedforward_model, self.model.guidance_model, self.optimizer_guidance,
            self.optimizer_generator, self.scheduler_guidance, self.scheduler_generator
        ) = accelerator.prepare(
            self.model.feedforward_model, self.model.guidance_model, self.optimizer_guidance,
            self.optimizer_generator, self.scheduler_guidance, self.scheduler_generator
        )

        self.accelerator = accelerator
        self.train_iters = args.train_iters
        self.batch_size = args.batch_size
        self.resolution = args.resolution
        self.log_iters = args.log_iters
        self.wandb_iters = args.wandb_iters
        self.conditioning_sigma = args.conditioning_sigma

        self.label_dim = args.label_dim
        self.eye_matrix = torch.eye(self.label_dim, device=accelerator.device)
        self.delete_ckpts = args.delete_ckpts
        self.max_grad_norm = args.max_grad_norm

        if args.checkpoint_path is not None:
            self.load(args.checkpoint_path)

        if self.accelerator.is_main_process:
            run = wandb.init(config=args, dir=self.output_path,
                             **{"mode": "online", "entity": args.wandb_entity, "project": args.wandb_project})
            wandb.run.log_code(".")
            wandb.run.name = args.wandb_name
            print(f"run dir: {run.dir}")
            self.wandb_folder = run.dir
            os.makedirs(self.wandb_folder, exist_ok=True)

    def load(self, checkpoint_path):
        # Please note that, after loading the checkpoints, all random seed, learning rate, etc.. will be reset to align with the checkpoint.
        self.step = int(checkpoint_path.replace("/", "").split("_")[-1])
        print("loading a previous checkpoints including optimizer and random seed")
        print(self.accelerator.load_state(checkpoint_path, strict=False))
        self.accelerator.print(f"Loaded checkpoint from {checkpoint_path}")

    def save(self):
        # training states
        output_path = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
        print(f"start saving checkpoint to {output_path}")

        self.accelerator.save_state(output_path)

        # remove previous checkpoints
        if self.delete_ckpts:
            for folder in os.listdir(self.output_path):
                if folder.startswith("checkpoint_model") and folder != f"checkpoint_model_{self.step:06d}":
                    shutil.rmtree(os.path.join(self.output_path, folder))

        if self.cache_checkpoints:
            # copy checkpoints to cache
            # overwrite the cache
            if os.path.exists(os.path.join(self.cache_dir, f"checkpoint_model_{self.step:06d}")):
                shutil.rmtree(os.path.join(self.cache_dir, f"checkpoint_model_{self.step:06d}"))

            shutil.copytree(
                os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}"),
                os.path.join(self.cache_dir, f"checkpoint_model_{self.step:06d}")
            )

            checkpoints = sorted(
                [folder for folder in os.listdir(self.cache_dir) if folder.startswith("checkpoint_model")]
            )

            if len(checkpoints) > self.max_checkpoint:
                for folder in checkpoints[:-self.max_checkpoint]:
                    shutil.rmtree(os.path.join(self.cache_dir, folder))

        print("done saving")

    def train_one_step(self):
        self.model.train()

        accelerator = self.accelerator

        # Retrieve a batch of real images from the dataloader.
        real_dict = next(self.real_image_dataloader)
        # print(f"🔍 real_dict type: {type(real_dict)}")

        # print(real_dict["class_labels"].shape)  # 실제로 어떤 shape인가?

        # Extract the images from the dictionary and normalize them.
        # scaled from [0,1] to [-1,1].
        real_image = real_dict["images"] * 2.0 - 1.0
        # real_label = self.eye_matrix[real_dict["class_labels"].squeeze(dim=1)]
        real_label = self.eye_matrix[real_dict["class_labels"]]

        # real_label = self.eye_matrix[real_dict["class_labels"].unsqueeze(dim=-1).squeeze(dim=1)]

        real_train_dict = {
            "real_image": real_image,
            "real_label": real_label
        }

        # Generate scaled noise based on the maximum sigma value.
        scaled_noise = torch.randn(
            self.batch_size, 3, self.resolution, self.resolution,
            device=accelerator.device
        ) * self.conditioning_sigma

        # Set timestep sigma to a preset value for all images in the batch.
        timestep_sigma = torch.ones(self.batch_size, device=accelerator.device) * self.conditioning_sigma # use same sigma value

        # For conditional generation, randomly generate labels.
        labels = torch.randint(
            low=0, high=self.label_dim, size=(self.batch_size,),
            device=accelerator.device, dtype=torch.long
        )
        # Convert these labels to one-hot encoding.
        labels = self.eye_matrix[labels] # random class index -> one-hot encoding condition label

        COMPUTE_GENERATOR_GRADIENT = self.step % self.dfake_gen_update_ratio == 0

        # generate images and optionaly compute the generator gradient
        generator_loss_dict, generator_log_dict = self.model(
            scaled_noise, timestep_sigma, labels,
            real_train_dict=real_train_dict,
            compute_generator_gradient=COMPUTE_GENERATOR_GRADIENT,
            generator_turn=True,
            guidance_turn=False
        )

        # first update the generator if the current step is a multiple of dfake_gen_update_ratio
        generator_loss = 0.0

        if COMPUTE_GENERATOR_GRADIENT:
            generator_loss += generator_loss_dict["loss_dm"]

            if self.gan_classifier and "gen_cls_loss" in generator_loss_dict:
                generator_loss += generator_loss_dict["gen_cls_loss"] * self.gen_cls_loss_weight

            if "pre_noise_cycle_loss" in generator_loss_dict:
                generator_loss += generator_loss_dict["pre_noise_cycle_loss"] * 0.05  # 또는 하이퍼파라미터화

            self.accelerator.backward(generator_loss)
            generator_grad_norm = accelerator.clip_grad_norm_(self.model.feedforward_model.parameters(),
                                                              self.max_grad_norm)
            self.optimizer_generator.step()

            # if we also compute gan loss, the classifier also received gradient
            # zero out guidance model's gradient avoids undesired gradient accumulation
            self.optimizer_generator.zero_grad()
            self.optimizer_guidance.zero_grad()

        self.scheduler_generator.step()

        #####
        if self.gan_classifier:
            with torch.no_grad():
                z = torch.randn_like(scaled_noise)
                timesteps = torch.randint(
                    low=self.model.guidance_min_step,
                    high=self.model.guidance_max_step + 1,
                    size=(self.batch_size, 1, 1, 1),
                    device=accelerator.device
                )
                timestep_sigma = self.model.guidance_model.karras_sigmas[timesteps]
                g_few_clean = self.model.guidance_model.real_unet(z, timestep_sigma, labels)
                z_noisy = z + timestep_sigma * torch.randn_like(z)
                g_few_noisy = self.model.guidance_model.real_unet(z_noisy, timestep_sigma, labels)

            d_loss, d_log_dict = self.model(
                scaled_noisy_image=None,
                timestep_sigma=None,
                labels=None,
                discriminator_turn=True,
                generator_data_dict={"real_image": real_image, "g_few_clean": g_few_clean, "g_few_noisy": g_few_noisy, "label": labels}
            )

            self.accelerator.backward(d_loss)
            self.optimizer_guidance.step()
            self.optimizer_generator.zero_grad()

            d_log_dict['discriminator_loss'] = d_loss.item()

        # update the guidance model (dfake and classifier)
        guidance_loss_dict, guidance_log_dict = self.model(
            scaled_noise, timestep_sigma, labels,
            real_train_dict=real_train_dict,
            compute_generator_gradient=False,
            generator_turn=False,
            guidance_turn=True,
            guidance_data_dict=generator_log_dict['guidance_data_dict']
        )

        guidance_loss = 0

        guidance_loss += guidance_loss_dict["loss_fake_mean"]

        if self.gan_classifier:
            guidance_loss += guidance_loss_dict["guidance_cls_loss"] * self.cls_loss_weight

        self.accelerator.backward(guidance_loss)
        guidance_grad_norm = accelerator.clip_grad_norm_(self.model.guidance_model.parameters(), self.max_grad_norm)
        self.optimizer_guidance.step()
        self.optimizer_guidance.zero_grad()
        self.scheduler_guidance.step()
        self.optimizer_generator.zero_grad()

        # combine the two dictionaries
        loss_dict = {**generator_loss_dict, **guidance_loss_dict}
        log_dict = {**generator_log_dict, **guidance_log_dict}

        if self.step % self.wandb_iters == 0:
            log_dict['generated_image'] = accelerator.gather(log_dict['generated_image'])
            log_dict['dmtrain_grad'] = accelerator.gather(log_dict['dmtrain_grad'])
            log_dict['dmtrain_timesteps'] = accelerator.gather(log_dict['dmtrain_timesteps'])
            log_dict['dmtrain_pred_real_image'] = accelerator.gather(log_dict['dmtrain_pred_real_image'])
            log_dict['dmtrain_pred_fake_image'] = accelerator.gather(log_dict['dmtrain_pred_fake_image'])

        if accelerator.is_main_process and self.step % self.wandb_iters == 0:
            # TODO: Need more refactoring here
            with torch.no_grad():
                generated_image = log_dict['generated_image']
                generated_image_brightness = (generated_image * 0.5 + 0.5).clamp(0, 1).mean()
                generated_image_std = (generated_image * 0.5 + 0.5).clamp(0, 1).std()
                # print(f"🔍 generated_image shape: {generated_image.shape}")
                generated_image_grid = prepare_images_for_saving(generated_image, resolution=self.resolution)

                # generated_image_grid = prepare_images_for_saving(generated_image, resolution=self.resolution)

                data_dict = {
                    "generated_image": wandb.Image(generated_image_grid),
                    "generated_image_brightness": generated_image_brightness.item(),
                    "generated_image_std": generated_image_std.item(),
                    "generator_grad_norm": generator_grad_norm.item(),
                    "guidance_grad_norm": guidance_grad_norm.item()
                }

                (
                    dmtrain_noisy_latents, dmtrain_pred_real_image, dmtrain_pred_fake_image,
                    dmtrain_grad, dmtrain_gradient_norm
                ) = (
                    log_dict['dmtrain_noisy_latents'], log_dict['dmtrain_pred_real_image'],
                    log_dict['dmtrain_pred_fake_image'],
                    log_dict['dmtrain_grad'], log_dict['dmtrain_gradient_norm']
                )
                if "pre_noise_cycle_loss" in loss_dict:
                    data_dict["pre_noise_cycle_loss"] = loss_dict["pre_noise_cycle_loss"].item()
                    data_dict["pre_noise_cycle_weight"] = self.args.pre_noise_cycle_weight
                gradient_brightness = dmtrain_grad.mean()
                gradient_std = dmtrain_grad.std(dim=[1, 2, 3]).mean()

                dmtrain_pred_real_image_mean = (dmtrain_pred_real_image * 0.5 + 0.5).clamp(0, 1).mean()
                dmtrain_pred_fake_image_mean = (dmtrain_pred_fake_image * 0.5 + 0.5).clamp(0, 1).mean()

                dmtrain_pred_read_image_std = (dmtrain_pred_real_image * 0.5 + 0.5).clamp(0, 1).std()
                dmtrain_pred_fake_image_std = (dmtrain_pred_fake_image * 0.5 + 0.5).clamp(0, 1).std()

                dmtrain_noisy_latents_grid = prepare_images_for_saving(dmtrain_noisy_latents,
                                                                       resolution=self.resolution)
                dmtrain_pred_real_image_grid = prepare_images_for_saving(dmtrain_pred_real_image,
                                                                         resolution=self.resolution)
                dmtrain_pred_fake_image_grid = prepare_images_for_saving(dmtrain_pred_fake_image,
                                                                         resolution=self.resolution)

                gradient = dmtrain_grad
                gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
                gradient = (gradient - 0.5) / 0.5
                gradient = prepare_images_for_saving(gradient, resolution=self.resolution)

                gradient_scale_grid = draw_valued_array(
                    dmtrain_grad.abs().mean(dim=[1, 2, 3]).cpu().numpy(),
                    output_dir=self.wandb_folder
                ) # dmtrain_grade_shape = B 3 H W
                # ABS().MEAN(DIM=[1,2,3]) : mean gradient per image -> [B] shape
                # visualize heatmap -> 1 image
                #mean gradient by color -> heatmap

                difference_scale_grid = draw_valued_array(
                    (dmtrain_pred_real_image - dmtrain_pred_fake_image).abs().mean(dim=[1, 2, 3]).cpu().numpy(),
                    output_dir=self.wandb_folder
                ) # shape = B 3 H W
                #abs().mean(dim=[1,2,3]) : pred vs real mean pixel difference -> [B] shape
                # visualize heatmap
                # mean difference real vs fake_pred -> number heatmap

                difference = (dmtrain_pred_fake_image - dmtrain_pred_real_image)

                difference_brightness = difference.mean()

                difference = (difference - difference.min()) / (difference.max() - difference.min())
                difference = (difference - 0.5) / 0.5
                difference = prepare_images_for_saving(difference, resolution=self.resolution)

                dmtrain_timesteps_grid = draw_valued_array(
                    log_dict['dmtrain_timesteps'].squeeze().cpu().numpy(),
                    output_dir=self.wandb_folder
                )

                data_dict.update(
                    {
                        "dmtrain_noisy_latents_grid": wandb.Image(dmtrain_noisy_latents_grid),
                        "dmtrain_pred_real_image_grid": wandb.Image(dmtrain_pred_real_image_grid),
                        "dmtrain_pred_fake_image_grid": wandb.Image(dmtrain_pred_fake_image_grid),
                        "loss_dm": loss_dict['loss_dm'].item(),
                        "loss_fake_mean": loss_dict['loss_fake_mean'].item(),
                        "dmtrain_gradient_norm": dmtrain_gradient_norm,
                        "gradient": wandb.Image(gradient),
                        "difference": wandb.Image(difference),
                        "gradient_scale_grid": wandb.Image(gradient_scale_grid),
                        "difference_norm_grid": wandb.Image(difference_scale_grid),
                        "dmtrain_timesteps_grid": wandb.Image(dmtrain_timesteps_grid),
                        "gradient_brightness": gradient_brightness.item(),
                        "difference_brightness": difference_brightness.item(),
                        "gradient_std": gradient_std.item(),
                        "dmtrain_pred_real_image_mean": dmtrain_pred_real_image_mean.item(),
                        "dmtrain_pred_fake_image_mean": dmtrain_pred_fake_image_mean.item(),
                        "dmtrain_pred_read_image_std": dmtrain_pred_read_image_std.item(),
                        "dmtrain_pred_fake_image_std": dmtrain_pred_fake_image_std.item()
                    }
                )

                (
                    faketrain_latents, faketrain_noisy_latents, faketrain_x0_pred
                ) = (
                    log_dict['faketrain_latents'], log_dict['faketrain_noisy_latents'],
                    log_dict['faketrain_x0_pred']
                )

                faketrain_latents_grid = prepare_images_for_saving(faketrain_latents, resolution=self.resolution)
                faketrain_noisy_latents_grid = prepare_images_for_saving(faketrain_noisy_latents,
                                                                         resolution=self.resolution)
                faketrain_x0_pred_grid = prepare_images_for_saving(faketrain_x0_pred, resolution=self.resolution)

                data_dict.update({
                    "faketrain_latents": wandb.Image(faketrain_latents_grid),
                    "faketrain_noisy_latents": wandb.Image(faketrain_noisy_latents_grid),
                    "faketrain_x0_pred": wandb.Image(faketrain_x0_pred_grid)
                })

                if self.gan_classifier:
                    data_dict['guidance_cls_loss'] = loss_dict['guidance_cls_loss'].item()
                    data_dict['gen_cls_loss'] = loss_dict['gen_cls_loss'].item()

                if self.gan_classifier and "discriminator_loss" in d_log_dict:
                    data_dict['discriminator_loss'] = d_log_dict['discriminator_loss']

                    # pred_realism_on_fake = log_dict["pred_realism_on_fake"]
                    # pred_realism_on_real = log_dict["pred_realism_on_real"]

                    # hist_pred_realism_on_fake = draw_probability_histogram(pred_realism_on_fake.cpu().numpy())
                    # hist_pred_realism_on_real = draw_probability_histogram(pred_realism_on_real.cpu().numpy())

                    # data_dict.update(
                    #     {
                    #         "hist_pred_realism_on_fake": wandb.Image(hist_pred_realism_on_fake),
                    #         "hist_pred_realism_on_real": wandb.Image(hist_pred_realism_on_real)
                    #     }
                    # )

                wandb.log(
                    data_dict,
                    step=self.step
                )

        self.accelerator.wait_for_everyone()

    def train(self):
        for index in range(self.step, self.train_iters):
            self.train_one_step()

            if self.accelerator.is_main_process:
                if (not self.no_save) and self.step % self.log_iters == 0:
                    self.save()

                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

            self.accelerator.wait_for_everyone()
            self.step += 1

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_id", type=str, default="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl") # default 추가
    parser.add_argument("--model_id", type=str, default="C:/Users/user/Desktop/diffusion/DMD2_main/edm-cifar10-32x32-cond-vp.pkl") # default 추가
    parser.add_argument("--output_path", type=str, default='C:/Users/user/Desktop/diffusion/DMD2_main/experiments/cifar/output')
    parser.add_argument("--dataset_path", type=str, default="C:/Users/user/Desktop/diffusion/DMD2_main/cifar10/")
    parser.add_argument("--train_iters", type=int, default=505000)
    parser.add_argument("--log_iters", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--initialie_generator", action="store_true", default=True)
    parser.add_argument("--checkpoint_path", type=str, default="C:/Users/user/Desktop/diffusion/DMD2_main/experiments/cifar/output/time_1749446925_seed2/checkpoint_model_010000") #default="C:/Users/user/Desktop/diffusion/DMD2_main/experiments/cifar/output/time_1749446925_seed2/checkpoint_model_010000"
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_iters", type=int, default=100)
    parser.add_argument("--wandb_name", type=str, default="Proposed_DMD2_cifar10") # default 추가
    parser.add_argument("--label_dim", type=int, default=10)
    parser.add_argument("--warmup_step", type=int, default=500, help="warmup step for network")
    parser.add_argument("--min_step_percent", type=float, default=0.02, help="minimum step percent for training")
    parser.add_argument("--max_step_percent", type=float, default=0.98, help="maximum step percent for training")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--conditioning_sigma", type=float, default=80.0)

    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_data", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--dataset_name", type=str, default='cifar10')
    parser.add_argument("--ckpt_only_path", type=str, default=None, help="checkpoint (no optimizer state) only path")
    parser.add_argument("--delete_ckpts", action="store_true")
    parser.add_argument("--max_checkpoint", type=int, default=20000)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--max_grad_norm", type=int, default=10)
    # parser.add_argument("--real_image_path", type=str) # 기존코드
    parser.add_argument("--real_image_path", type=str, default="C:/Users/user/Desktop/skh/diffusion/DMD2_main/experiments/cifar/real_images")
    parser.add_argument("--use_accelerator", action="store_true", help="Use accelerator for distributed training")

    parser.add_argument("--generator_lr", type=float, default=2e-6)
    # parser.add_argument("--guidance_lr", type=float) # 518, 519 각각 default 추가
    parser.add_argument("--guidance_lr", type=float, default=2e-6) # 518, 519 각각 default 추가
    parser.add_argument("--dfake_gen_update_ratio", type=int, default=5)

    parser.add_argument("--cls_loss_weight", type=float, default=1e-2)
    parser.add_argument("--gan_classifier", action="store_true", default=True) # default=True 잇어야함
    parser.add_argument("--gen_cls_loss_weight", type=float, default=3e-3)
    parser.add_argument("--diffusion_gan", action="store_true")
    parser.add_argument("--diffusion_gan_max_timestep", type=int, default=0)

    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--generator_ckpt_path", type=str)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    assert args.wandb_iters % args.dfake_gen_update_ratio == 0, "wandb_iters should be a multiple of dfake_gen_update_ratio"

    return args

if __name__ == "__main__":
    args = parse_args()

    trainer = Trainer(args)

    trainer.train()
