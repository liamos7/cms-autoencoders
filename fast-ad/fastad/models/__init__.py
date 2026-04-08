import torch
from .modules import SimpleEncoder, SimpleDecoder, ConvNet2FC, DeConvNet2,\
 ConfigurableEncoder, ConfigurableDecoder, RGBEncoder, RGBDecoder, CicadaEncoder,\
 CicadaDecoder
from .teachers import AE, VAE, NAE, NAEWithEnergyTraining
from .students import StudentA, StudentB, StudentC, StudentD, StudentE,\
     StudentF, StudentG

# --- Dataset latent dimensions ---
CICADA_LATENT_DIM = 20


def get_mnist_ae():
    encoder = SimpleEncoder(latent_dim=20)
    decoder = SimpleDecoder(latent_dim=20)
    model = AE(encoder, decoder)
    return model

def get_mnist_vae():
    encoder = SimpleEncoder(latent_dim=40)
    decoder = SimpleDecoder(latent_dim=20)
    model = VAE(encoder, decoder)
    return model

def get_mnist_nae():
    encoder = SimpleEncoder(latent_dim=20)
    decoder = SimpleDecoder(latent_dim=20)
    model = NAE(encoder, decoder, spherical=True)
    return model

def get_mnist_nae_with_energy(pretrained_path: str):
    encoder = SimpleEncoder(latent_dim=20)
    decoder = SimpleDecoder(latent_dim=20)
    model = NAEWithEnergyTraining(
        encoder,
        decoder,
        spherical=True,
        x_noise_std=0.001,
        x_steps=10,
        x_step_size=0.5,
        z_noise_std=0.01,
        z_steps=10,
        z_step_size=0.2,
        use_two_stage=True
    )
    model.load_pretrained_nae(pretrained_path)    
    return model


def get_fmnist_ae():
    encoder = SimpleEncoder(latent_dim=20)
    decoder = SimpleDecoder(latent_dim=20)
    model = AE(encoder, decoder)
    return model

def get_fmnist_vae():
    encoder = SimpleEncoder(latent_dim=40)
    decoder = SimpleDecoder(latent_dim=20)
    model = VAE(encoder, decoder)
    return model

def get_fmnist_nae():
    encoder = SimpleEncoder(latent_dim=20)
    decoder = SimpleDecoder(latent_dim=20)
    model = NAE(encoder, decoder, spherical=True)
    return model

def get_fmnist_nae_with_energy(pretrained_path: str):
    encoder = SimpleEncoder(latent_dim=20)
    decoder = SimpleDecoder(latent_dim=20)
    model = NAEWithEnergyTraining(
        encoder,
        decoder,
        spherical=True,
        x_noise_std=0.001,
        x_steps=10,
        x_step_size=0.5,
        z_noise_std=0.01,
        z_steps=10,
        z_step_size=0.2,
        use_two_stage=True
    )
    model.load_pretrained_nae(pretrained_path)
    return model

def get_cifar10_ae():
    pass

def get_cifar10_vae():
    pass

def get_cifar10_nae():
    pass

def get_cifar10_nae_with_energy(pretrained_path: str):
    pass

def get_cicada_ae(latent_dim: int = CICADA_LATENT_DIM):
    encoder = CicadaEncoder(latent_dim=latent_dim)
    decoder = CicadaDecoder(latent_dim=latent_dim)
    model = AE(encoder, decoder)
    return model


def get_cicada_nae_with_energy(pretrained_path: str, latent_dim: int = CICADA_LATENT_DIM):
    encoder = CicadaEncoder(latent_dim=latent_dim)
    decoder = CicadaDecoder(latent_dim=latent_dim)
    model = NAEWithEnergyTraining(
        encoder,
        decoder,
        spherical=True,
        gamma=5e-1,
        z_steps=25,
        z_step_size=0.005,
        z_noise_std=0.005,
        x_steps=10,
        x_step_size=0.001,
        x_noise_std=0.0005,
        x_use_annealing=True,
        buffer_size=10000,
        buffer_prob=0.95,
        buffer_reinit_prob=0.10,
        latent_dim=latent_dim,
    )
    model.load_pretrained_nae(pretrained_path)
    return model


def get_teacher_model(model_name: str, dataset_name: str, pretrained_path: str = None, latent_dim: int = None):

    print(f"get_teacher_model(model_name={model_name}, dataset_name={dataset_name}, pretrained_path={pretrained_path})")
    if dataset_name == "MNIST":
        if model_name == "AE":
            model = get_mnist_ae()
        elif model_name == "NAE":
            model = get_mnist_nae()
        elif model_name == "NAEWithEnergyTraining":
            model = get_mnist_nae_with_energy(pretrained_path)
        elif model_name == "VAE":
            model = get_mnist_vae()
        else:
            raise ValueError(f"Model {model_name} not recognized for MNIST (use AE, VAE, NAE, or NAEWithEnergyTraining)")
        
    elif dataset_name == "FMNIST":
        if model_name == "AE":
            model = get_fmnist_ae()
        elif model_name == "NAE":
            model = get_fmnist_nae()
        elif model_name == "NAEWithEnergyTraining":
            model = get_fmnist_nae_with_energy(pretrained_path)
        elif model_name == "VAE":
            model = get_fmnist_vae()
        else:
            raise ValueError(f"Model {model_name} not recognized for FMNIST (use AE, VAE, NAE, or NAEWithEnergyTraining)")
        
    elif dataset_name == "CIFAR10":
        if model_name == "AE":
            model = get_cifar10_ae()
        elif model_name == "NAE":
            model = get_cifar10_nae()
        elif model_name == "NAEWithEnergyTraining":
            model = get_cifar10_nae_with_energy(pretrained_path)
        elif model_name == "VAE":
            model = get_cifar10_vae()
        else:
            raise ValueError(f"Model {model_name} not recognized for CIFAR10 (use AE, VAE, NAE, or NAEWithEnergyTraining)")

    elif dataset_name == "CICADA":
        ld = latent_dim if latent_dim is not None else CICADA_LATENT_DIM
        if model_name == "AE":
            model = get_cicada_ae(latent_dim=ld)
        elif model_name == "NAEWithEnergyTraining":
            model = get_cicada_nae_with_energy(pretrained_path, latent_dim=ld)
        else:
            raise ValueError(f"Model {model_name} not recognized for CICADA (use AE or NAEWithEnergyTraining)")

    else:
        raise ValueError(f"Dataset {dataset_name} not recognized (use MNIST, FMNIST, or CIFAR10)")
    
    if pretrained_path:
        model_dict = torch.load(pretrained_path)
        model.load_state_dict(model_dict['model_state'], strict=False)

    return model

    # if dataset_name in {"MNIST", "FMNIST"}:
    #     encoder_cls, decoder_cls = SimpleEncoder, SimpleDecoder
    # elif dataset_name == "CIFAR10":
    #     encoder_cls, decoder_cls = RGBEncoder, RGBDecoder
    # else:
    #     raise ValueError(f"Dataset {dataset_name} not recognized (use MNIST, FMNIST, or CIFAR10)")
    # if model_name == "AE":
    #     ae_cls = AE
    #     latent_dims = (20, 20)
    #     enc_activation = "sigmoid"
    # elif model_name == "VAE":
    #     ae_cls = VAE
    #     latent_dims = (40, 20)
    #     enc_activation = "sigmoid"
    # elif model_name == "NAE":
    #     ae_cls = NAE
    #     latent_dims = (20, 20)
    #     enc_activation = "spherical"
    # elif model_name == "NAEWithEnergyTraining":
    #     ae_cls = NAEWithEnergyTraining
    #     latent_dims = (20, 20)
    #     enc_activation = "spherical"
    # else:
    #     raise ValueError(f"Model {model_name} not recognized (use AE, VAE, or NAE)")
    
    # encoder = encoder_cls(latent_dim=latent_dims[0])
    # decoder = decoder_cls(latent_dim=latent_dims[1])
    
    # # Create the model
    # if model_name in ["NAE", "NAEWithEnergyTraining"]:
    #     model = ae_cls(encoder, decoder, spherical=True)
    # else:
    #     model = ae_cls(encoder, decoder)
    
    # # Handle pretrained loading
    # if pretrained_path:
    #     assert model_name == "NAEWithEnergyTraining", "Pretrained path is only supported for NAEWithEnergyTraining"
    #     model.load_pretrained_nae(pretrained_path)

    # model.to(device)
    
    # return model


# def get_student_model(model_name: str, dataset_name: str):
#     cls_dict = {
#         "StudentA": StudentA,
#         "StudentB": StudentB,
#         "StudentC": StudentC,
#         "StudentD": StudentD,
#         "StudentE": StudentE,
#         "StudentF": StudentF,
#         "StudentG": StudentG,
#     }
#     if model_name not in cls_dict:
#         raise ValueError(f"Model {model_name} not recognized (use {', '.join(cls_dict.keys())})")
#     model_cls = cls_dict[model_name]
#     return model_cls()


def get_student_model_dict():
    return {
        "StudentA": StudentA(),
        "StudentB": StudentB(),
        "StudentC": StudentC(),
        "StudentD": StudentD(),
        "StudentE": StudentE(),
        "StudentF": StudentF(),
        "StudentG": StudentG(),
    }