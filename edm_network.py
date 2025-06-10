from DMD2_main.third_party.edm.training.networks import EDMPrecond
def get_imagenet_edm_config():
    return dict(
        augment_dim=0,
        # model_channels=192, # 기존 코드
        model_channels=128,
        # channel_mult=[1, 2, 3, 4], # 기존 코드
        channel_mult=[2, 2, 2],
        channel_mult_emb=4,
        # num_blocks=3, # 기존 코드
        num_blocks=4,
        # attn_resolutions=[32,16,8], # 기존 코드
        attn_resolutions=[16],
        dropout=0.0,
        label_dropout=0
    )

def get_edm_network(args):
    if args.dataset_name == "cifar10":
            unet = EDMPrecond(
            img_resolution=args.resolution,
            img_channels=3,
            label_dim=args.label_dim,
            use_fp16=args.use_fp16,
            sigma_min=0,
            sigma_max=float("inf"),
            sigma_data=args.sigma_data,
            # model_type="DhariwalUNet", # 기존 코드
            model_type="SongUNet",
            **get_imagenet_edm_config()
        )
    else:
        raise NotImplementedError

    return unet 