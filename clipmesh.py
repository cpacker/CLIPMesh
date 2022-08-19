import os
import sys
import yaml
import json
import torch
import argparse

sys.path.append("./nvdiffmodeling")

from loops.single import single_loop
from loops.util import random_string, set_seed

def dir_path(string):
    if os.path.exists(string):
        return string
    else:
        raise FileNotFoundError

def str2bool(v):
    """https://stackoverflow.com/a/43357954"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='CLIPMesh | Text to 3D meshes, please provide a config file with --path OR pass in parameters (default values for params are in configs/base.yaml)')
parser.add_argument('--path', type=dir_path, default='configs/defaults.yaml')

parser.add_argument('--optim', nargs='*', help="What to optimize from [body, expression, texture, normal, specular, pose, displacement]")
parser.add_argument('--options', nargs='*', help="What views to optimize from [face, full, back] or [main]")

parser.add_argument('--epochs', type=int, help="How many epochs to run")
parser.add_argument('--blur_epochs', type=int, help="If blurring the textures, how long to do it for")
parser.add_argument('--notex_epochs', type=int, help="How many epochs of 50/50 tex/notex renders")
parser.add_argument('--notex_batch_size', type=int, help="How much of the batch should be notex during a notex epoch (must be <= batch_size)")
parser.add_argument('--gpu', type=int, help="Which GPU")
parser.add_argument('--log_int', type=int, help="How often you want to log debug (also needs --debug_log=True")
parser.add_argument('--batch_size', type=int, help="How many images per epoch")
parser.add_argument('--render_res', type=int, help="Render iamge size")
parser.add_argument('--texture_res', type=int, help="Texture resolution ex: 256x256")

parser.add_argument('--shape_lr', type=float, help="LR for expression, body and pose")
parser.add_argument('--TV_weight', type=float, help="Texture denoising weight")
parser.add_argument('--texture_lr', type=float, help="Texture weight")
parser.add_argument('--displacement_lr', type=float, help="Displacement map weight")

parser.add_argument('--text', type=str, help="Text prompt (no split text prompts per-region)")
parser.add_argument('--negative_text', nargs='*', type=str, help="Negative text prompts, useful for pushing CLIP away from faces and text")
parser.add_argument('--CLIP', type=str, help="CLIP Model to use")
parser.add_argument('--render', type=str, help="One from [diffuse, pbr, normal, tangent]")
parser.add_argument('--uv_path', type=str, help="Path to base obj containing UV cords and faces info")
parser.add_argument('--output_path', type=str, help="Where to output result")
parser.add_argument('--uv_mask_path', type=str, help="Path to UV mask (default to None)")

parser.add_argument('--rand_bkg', type=str2bool, help="Augment background during training")
parser.add_argument('--blur', type=str2bool, help="Blur the materials")
parser.add_argument('--blur_kernel', nargs='*', type=int, help="Blur gaussian kernel size")
parser.add_argument('--blur_sigma', nargs='*', type=int, help="Blur gaussian kernel sigma")
parser.add_argument('--debug_log', type=str2bool, help="Log video and training renders")
parser.add_argument('--plot', type=str2bool, default=False, help="Plot or Save logging images")

parser.add_argument('--subdivision', type=int, help="How many times to subdivide the base mesh (each level quadruples the triangle count)")
parser.add_argument('--laplacian', type=str2bool, help="Apply Laplacian regularization to mesh vertices")
parser.add_argument('--laplacian_factor', type=float, help="Laplacian regularization weight (default to a schedule)")
parser.add_argument('--laplacian_on_fine_mesh', type=str2bool, help="Whether to apply regulariation to base mesh or subdivided mesh")

if __name__ == "__main__":

    # Load default values first
    with open('configs/defaults.yaml', "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Next load provided config file (if provided)
    parsed_args = parser.parse_args()
    if parsed_args.path != 'configs/defaults.yaml':
        with open(parsed_args.path, "r") as stream:
            try:
                provided_cfg = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        cfg.update(provided_cfg)

    # Next overwrite any YAML settings with provided kwarg flags
    # NOTE: This relies on none of the argparse flags having defaults
    parsed_args = vars(parsed_args)
    parsed_args = {k: v for k, v in parsed_args.items() if v is not None}
    cfg.update(parsed_args)
    print("parsed_args",parsed_args)
    
    device = torch.device('cuda:' + str(cfg['gpu']))
    torch.cuda.set_device(device)

    # FIXME should use python uuid instead
    attempts = 0
    while attempts == 0 or os.path.isdir(save_path):
        if attempts > 10:
            raise RuntimeError("Failed to create unique directory")
        cfg['ID'] = cfg['text'] + "-" + random_string(5)
        save_path = os.path.join(cfg["output_path"], cfg["ID"])
        attempts += 1

    if cfg['notex_epochs'] is None:
        # Default is 50% of the epochs with mixed no-texture renders
        cfg['notex_epochs'] = cfg['epochs'] // 2

    if cfg['blur'] and cfg['blur_epochs'] is None:
        # Default if blur_epochs = True is all epochs
        cfg['blur_epochs'] = cfg['epochs']    

    # Make sure kernel and sigma are tuples
    if isinstance(cfg['blur_kernel'], int):
        cfg['blur_kernel'] = [cfg['blur_kernel'],cfg['blur_kernel']]
    if len(cfg['blur_kernel']) == 1:
        cfg['blur_kernel'] = cfg['blur_kernel']*2
    if isinstance(cfg['blur_sigma'], int):
        cfg['blur_sigma'] = [cfg['blur_sigma'],cfg['blur_sigma']]
    if len(cfg['blur_sigma']) == 1:
        cfg['blur_sigma'] = cfg['blur_sigma']*2

    #if cfg['negative_text'] is None:
    #    cfg['negative_text'] = []

    set_seed(cfg['seed'])

    print(json.dumps(cfg, sort_keys=True, indent=4))
    if cfg['loop'] == "single":
        single_loop(cfg)
