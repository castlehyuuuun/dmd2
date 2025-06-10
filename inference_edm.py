# inference_edm.py
import os
import torch
from tqdm import tqdm
import torchvision.utils as vutils
from argparse import ArgumentParser
from accelerate.utils import set_seed
from main.edm.edm_unified_model import EDMUniModel
from cleanfid import fid
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_only_path", type=str, default="C:/Users/user/Desktop/diffusion/DMD2-main/experiments/cifar/output/time_1745583703_seed2/checkpoint_model_1200000")
    parser.add_argument("--save_dir", type=str, default="C:/Users/user/Desktop/diffusion/DMD2-main/experiments/cifar10/")
    parser.add_argument("--num_images", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--label_dim", type=int, default=10)
    parser.add_argument("--conditioning_sigma", type=float, default=80.0)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--real_dir", type=str, default="C:/Users/user/Desktop/diffusion/DMD2-main/cifar_train/")
    parser.add_argument("--model_id", type=str, default="C:/Users/user/Desktop/diffusion/DMD2-main/edm-cifar10-32x32-cond-vp.pkl")
    parser.add_argument("--dataset_name", type=str, default="cifar10")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--sigma_data", type=float, default=0.5)
    parser.add_argument("--gan_classifier", action="store_true")
    parser.add_argument("--diffusion_gan", action="store_true")
    parser.add_argument("--diffusion_gan_max_timestep", type=int, default=0)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--min_step_percent", type=float, default=0.02, help="minimum step percent for training")
    parser.add_argument("--max_step_percent", type=float, default=0.98, help="maximum step percent for training")
    parser.add_argument("--initialie_generator", action="store_true", default=True)

    return parser.parse_args()

@torch.no_grad()
def generate(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_dir, exist_ok=True)

    # Fake args for model init
    class Dummy:
        pass

    dummy_args = Dummy()
    for key, val in vars(args).items():
        setattr(dummy_args, key, val)

    model = EDMUniModel(dummy_args, accelerator=None)
    model.feedforward_model.load_state_dict(
        torch.load(os.path.join(args.ckpt_only_path, "pytorch_model.bin"), map_location="cpu"),
        strict=False
    )
    model.to(device)
    model.eval()

    eye_matrix = torch.eye(args.label_dim, device=device)
    total = 0
    idx = 0
    pbar = tqdm(total=args.num_images)

    while total < args.num_images:
        bsz = min(args.batch_size, args.num_images - total)
        noise = torch.randn(bsz, 3, args.resolution, args.resolution, device=device) * args.conditioning_sigma
        labels = eye_matrix[torch.randint(0, args.label_dim, (bsz,), device=device)]
        sigma = torch.ones(bsz, device=device) * args.conditioning_sigma

        images = model.feedforward_model(noise, sigma, labels)
        # images = (images.clamp(-1, 1) + 1) / 2
        images = (images * 0.5 + 0.5).clamp(0,1)
        for i in range(bsz):
            save_path = os.path.join(args.save_dir, f"{idx:05d}.png")
            vutils.save_image(images[i], save_path)
            idx += 1
            total += 1
            if total >= args.num_images:
                break
        pbar.update(bsz)

    pbar.close()
    print(f"[DONE] {total} images saved to {args.save_dir}")
    # fid_score = compute_fid(args.real_dir, args.save_dir, batch_size=args.batch_size, device=str(device))
    # print(f"FID: {fid_score:.2f}")
    # save_comparison_from_folders(
    #     pred_dir=args.save_dir,
    #     gt_dir=args.real_dir,
    #     save_dir=os.path.join(args.save_dir, "comparison"),
    #     prefix="compare",
    #     num_to_save=50000,
    #     image_size=(args.resolution, args.resolution)
    # )
    # with open(os.path.join(args.save_dir, "fid_score.txt"), "w") as f:
    #     f.write(f"{fid_score:.4f}\n")
    score = fid.compute_fid(
        "C:/Users/user/Desktop/diffusion/DMD2-main/experiments/cifar10/",          # generated images
        "C:/Users/user/Desktop/diffusion/DMD2-main/cifar_train/",        # real images
        batch_size=100,
        dataset_res=32,
        device='cuda',
        mode='clean',
        num_workers=0
    )

    print(f"FID score: {score:.4f}")
if __name__ == "__main__":
    args = parse_args()
    generate(args)
