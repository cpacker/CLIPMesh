import os
import yaml
import glm
import tqdm
import smplx
import torch
import kornia
import torchvision

import numpy as np
import nvdiffrast.torch as dr

from resize_right import resize
from resize_right.interp_methods import lanczos3

from loops.util import CLIP, Video, batch_rodrigues, cosine_avg, get_random_bg, persp_proj, unit_size

from nvdiffmodeling.src import obj
from nvdiffmodeling.src import mesh
from nvdiffmodeling.src import render
from nvdiffmodeling.src import texture

from nvdiffmodeling.src import regularizer
from nvdiffmodeling.train import load_mesh 
from nvdiffmodeling.src.mesh import Mesh 


def single_loop(config):

    if config["plot"]:
        from IPython.display import clear_output
        import matplotlib.pyplot as plt

    glctx = dr.RasterizeGLContext()

    config["path"] = os.path.join(config["output_path"], config["ID"])
    os.makedirs(config["path"])
    with open(os.path.join(config["path"], 'config.yaml'), 'w') as fh:
        yaml.dump(config, fh, allow_unicode=True)

    device = torch.device('cuda')

    if config["debug_log"]:
        video = Video(os.path.join(config["path"]))
    clip_model = CLIP(device, model=config["CLIP"])

    ##############################################
    ##############################################

    # TODO move these to kwargs
    # Values based on: https://github.com/NVlabs/nvdiffmodeling/blob/9abcce13e92efd31976e6781073af53571ece7fd/train.py#L420
    # and https://github.com/NVlabs/nvdiffmodeling/blob/9abcce13e92efd31976e6781073af53571ece7fd/configs/spot.json
    config["random_textures"] = True   # we have to use random textures since there's no reference
    config["custom_mip"] = False  # CLIPMesh-SMPLX uses auto mipmaps
    config["skip_train"] = []   # this is handled by config['optim'], TODO remove
    config["min_roughness"] = 0.25  # CLIPMesh-SMPLX uses a default of 0.0 (if texture is being optimized)  TODO expose option
    config["relative_laplacian"] = False  # Not used TODO should we use it?
    config["layers"] = 1  # CLIPMesh-SMPLX has this implicitly set to 1
    config["displacement"] = 0.15  # hardcoded in nvdiffmodeling

    # default in nvdiffmodelling is 1024, default from CLIPMesh-SMPLX is 256
    if not isinstance(config["texture_res"], list):
        assert isinstance(config["texture_res"], int), type(config["texture_res"])
        config["texture_res"] = [config["texture_res"], config["texture_res"]]

    # Base mesh
    # TODO add option to try multiple base mesh candidates (sphere + horizontal + vertical prism)
    base_mesh = load_mesh(config["uv_path"])
    print("Base mesh has %d triangles and %d vertices." % (base_mesh.t_pos_idx.shape[0], base_mesh.v_pos.shape[0]))
    print("Avg edge length: %f" % regularizer.avg_edge_length(base_mesh))
    normalized_base_mesh = mesh.unit_size(base_mesh)

    # ==============================================================================================
    #  Initialize weights / variables for trainable mesh
    # ==============================================================================================
    # From: https://github.com/NVlabs/nvdiffmodeling/blob/main/train.py#L105

    v_pos_opt = normalized_base_mesh.v_pos.clone().detach().requires_grad_(True)

    # Normal map
    normal_map_opt = texture.create_trainable(np.array([0, 0, 1]), config["texture_res"], auto_mipmaps=True)
    # Texture map
    kd_map_opt     = texture.create_trainable(np.random.uniform(size=config["texture_res"] + [3], low=0.0 if "texture" in config["optim"] else 0.85, high=1.0), config["texture_res"], auto_mipmaps=True)
    # Specular map
    ks_map_opt     = texture.create_trainable(np.array([0, 0, 0]), config["texture_res"], auto_mipmaps=True)

    # Only create displacement map if we subdivide the base mesh
    if config["subdivision"] > 0:
        ds_map_opt = torch.tensor(np.zeros(config["texture_res"] + [1], dtype=np.float32), dtype=torch.float32, device=device, requires_grad=True)
    else:
        ds_map_opt = None

    print("texture_res is", config["texture_res"])

    # ==============================================================================================
    #  Setup material for optimized mesh
    # ==============================================================================================

    opt_material = {
        'bsdf'   : config['render'],
        'kd'     : kd_map_opt,
        'ks'     : ks_map_opt,
        'normal' : normal_map_opt
    }

    # ==============================================================================================
    #  Setup base mesh operation graph, precomputes topology etc.
    # ==============================================================================================

    # TODO need to figure out what to do with the UV mask
    if "uv_mask_path" in config and config["uv_mask_path"] is not None:
        uv_mask = resize(
            torchvision.io.read_image(config["uv_mask_path"]),
            out_shape=config["texture_res"]
        ).permute(1, 2, 0).repeat(1, 1, 3).to(device)
        print("UV mask size is", uv_mask.shape)
    else:
        uv_mask = None

    # Create optimized mesh with trainable positions 
    opt_base_mesh = Mesh(v_pos_opt, normalized_base_mesh.t_pos_idx, material=opt_material, base=normalized_base_mesh)

    # Compute smooth vertex normals
    opt_base_mesh = mesh.auto_normals(opt_base_mesh)

    # Set up tangent space
    opt_base_mesh = mesh.compute_tangents(opt_base_mesh)

    # Subdivide if we're doing displacement mapping
    if config["subdivision"] > 0:
        # Subdivide & displace optimized mesh
        displ_map_var = ds_map_opt
        subdiv_opt_mesh = mesh.subdivide(opt_base_mesh, steps=config["subdivision"])
        opt_detail_mesh = mesh.displace(subdiv_opt_mesh, displ_map_var, config["displacement"], keep_connectivity=True)
    else:
        opt_detail_mesh = opt_base_mesh

    # Laplace regularizer
    if config["laplacian"]:
        #lap_loss_fn = regularizer.laplace_regularizer_const(opt_detail_mesh)
        lap_loss_fn = regularizer.laplace_regularizer_const(opt_base_mesh)

    # ==============================================================================================
    #  Setup torch optimizer
    # ==============================================================================================

    #optimizer  = torch.optim.Adam(trainable_list, lr=FLAGS.learning_rate)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.0, 10**(-x*0.0002)))

    # NOTE: ignoring "trainable_list", instead using "train_params" and "shape_params"
    train_params     = []
    shape_params     = []
    #if not 'position' in config["skip_train"]:  # vertices go in shape
    if 'shape' in config["optim"]:
        #trainable_list += [v_pos_opt]        
        print("Adding `shape` to the optimizer")
        shape_params += [v_pos_opt]        
    #if not 'normal' in config["skip_train"]:  # normal goes in "train"
    if 'normal' in config["optim"]:
        #trainable_list += normal_map_opt.getMips()
        print("Adding `normal` to the optimizer")
        train_params += normal_map_opt.getMips()
    #if not 'kd' in config["skip_train"]:  # texture goes in "train"
    if 'texture' in config["optim"]:
        #trainable_list += kd_map_opt.getMips()
        print("Adding `texture` to the optimizer")
        train_params += kd_map_opt.getMips()
    #if not 'ks' in config["skip_train"]:  # specular goes in "train"
    if 'specular' in config["optim"]:    
        #trainable_list += ks_map_opt.getMips()
        print("Adding `specular` to the optimizer")
        train_params += ks_map_opt.getMips()
    #if not 'displacement' in config["skip_train"] and displacement_map_var is not None:  # TODO I think displacement should go in shape?
    #if 'displacement' in config["optim"] and displacement_map_var is not None:
        #shape_params += [displacement_map_var]
        #pass

    optimizers = []
    if len(train_params) > 0:
        optimizers.append(torch.optim.Adam(train_params, lr=config["texture_lr"]))
    if len(shape_params) > 0:
        optimizers.append(torch.optim.Adam(shape_params, lr=config["shape_lr"]))

    # Displacement gets its own optimizer with a separate learning rate
    # FIXME modify to allow displacement map tuning w/o subdivision?
    if "displacement" in config["optim"] and ds_map_opt is not None:
        print("Adding `displacement` to the optimizer")
        optimizers.append(torch.optim.Adam([ds_map_opt], lr=config["displacement_lr"]))

    # ==============================================================================================
    #  Mesh optimization loop 
    # ==============================================================================================

    t_loop = tqdm.tqdm(range(config["epochs"]))
    for i in t_loop:

        if i == 0:
            # Create random textures for the "no-texture" renders
            # FIXME Why are these "trainable"?
            if isinstance(config["texture_res"], list):
                assert len(config["texture_res"]) == 2, config["texture_res"]
                # modified
                no_texture = texture.create_trainable(
                        np.random.uniform(size=config["texture_res"] + [3], low=0.85, high=1.0),
                        config["texture_res"], auto_mipmaps=True)
            else:
                # original
                no_texture = texture.create_trainable(
                        np.random.uniform(size=[config["texture_res"], config["texture_res"]] + [3], low=0.85, high=1.0),
                        [config["texture_res"], config["texture_res"]], auto_mipmaps=True)

        # Option to render with blurred versions of the texture/normal/displacement maps
        if config["blur"] is True and i <= config["blur_epochs"]:

            # low pass filter for textures
            texture_blur = texture.Texture2D(
                kornia.filters.gaussian_blur2d(
                    kd_map_opt.data.permute(0, 3, 1, 2),
                    kernel_size=config["blur_kernel"],
                    sigma=config["blur_sigma"],
                ).permute(0, 2, 3, 1).contiguous()
            )

            normal_blur = texture.Texture2D(
                kornia.filters.gaussian_blur2d(
                    normal_map_opt.data.permute(0, 3, 1, 2),
                    kernel_size=config["blur_kernel"],
                    sigma=config["blur_sigma"],
                ).permute(0, 2, 3, 1).contiguous()
            )

            if ds_map_opt is not None:
                displ_blur = kornia.filters.gaussian_blur2d(
                ds_map_opt.unsqueeze(0).permute(0, 3, 1, 2),
                kernel_size=config["blur_kernel"],
                sigma=config["blur_sigma"],
                ).permute(0, 2, 3, 1).contiguous().squeeze(0)

            # Seems like we can just change the base materials dictionary and everything will propogate through the renderer
            # (Mesh classes aren't doing any deep copies)
            opt_material['kd'] = texture_blur
            opt_material['normal'] = normal_blur
            # TODO make sure that this change is propogating through the subdivision op
            if ds_map_opt is not None:
                displ_map_var = displ_blur

        elif config["blur"] is True and i > config["blur_epochs"]:
            # After blurring, reset to the non-blurred mats
            # TODO implement a blurring schedule
            opt_material['kd'] = kd_map_opt
            opt_material['normal'] = normal_map_opt
            if ds_map_opt is not None:
                displ_map_var = ds_map_opt
            
            # Don't swap again
            config["blur"] = False

        mvp      = np.zeros((config["batch_size"], 4,4),  dtype=np.float32)
        campos   = np.zeros((config["batch_size"], 3), dtype=np.float32)
        lightpos = np.zeros((config["batch_size"], 3), dtype=np.float32)
        bkgs     = torch.zeros((config["batch_size"], config["render_res"], config["render_res"], 3)).to(device)
        prompts  = []

        # (Default batch_size is 60)
        for b in range(config["batch_size"]):
            op_ = 'main'  # no full/face/back, just main

            # Randomize the camera parameters
            # Choose a random:
            #   - elevation ("sample from a Beta distribution with a=1.0 and B=5.0 within a range 0 to 100")
            #   - azimuth ("uniformly sample azimuth from 0 to 360")
            #   - distance ("distance of the camera from the object is set to 5.0 in our examples")
            #   - and FOV ("randomly selecting a camera field of view between 30 to 60")
            assert config["cameras"][op_]["elev"][0] == 0.0
            elev = np.radians( np.random.beta( 1.0, 5.0 ) * config["cameras"][op_]["elev"][1] )
            #elev = np.radians( np.random.uniform( config["cameras"][op_]["elev"][0], config["cameras"][op_]["elev"][1] ) )
            azim = np.radians( np.random.uniform( config["cameras"][op_]["azim"][0], config["cameras"][op_]["azim"][1] ) )
            dist = np.random.uniform( config["cameras"][op_]["dist"][0], config["cameras"][op_]["dist"][1] ) 
            fov = np.random.uniform( config["cameras"][op_]["fov"][0], config["cameras"][op_]["fov"][1] ) 
            
            # Also, choose a random:
            #   - object offset (no further details in the paper, use CLIP-SMPLX default which is (0.0,0.15,0.0) for "full")
            #   - random background
            offsets = config["cameras"][op_]["offset"]
            bkgs[b] = get_random_bg(device, config["render_res"], config["render_res"]).squeeze(0) \
                      if config["rand_bkg"] \
                      else torch.ones((config["render_res"], config["render_res"], 3), device=device)

            # NOTE: all the prompts (per camera/image) are the same here, but 
            # if we want we can change them so each camera has a different prompt
            prompts.append(config['text'])

            proj_mtx = persp_proj(fov)

            # Generate random view
            cam_z = dist * np.cos(elev) * np.sin(azim)
            cam_y = dist * np.sin(elev)
            cam_x = dist * np.cos(elev) * np.cos(azim)
            
            # Random offset
            limit = config["cameras"][op_]["dist"][0] / 4.0
            rand_x = np.random.uniform( -limit, limit )
            rand_y = np.random.uniform( -limit, limit )

            modl = glm.translate(glm.mat4(), glm.vec3(rand_x, rand_y, 0))
                
            view  = glm.lookAt(
                glm.vec3(cam_x, cam_y, cam_z),
                glm.vec3(0 + offsets[0], 0 + offsets[1], 0 + offsets[2]),
                glm.vec3(0, -1, 0),
            )

            r_mv = view * modl
            r_mv = np.array(r_mv.to_list()).T

            mvp[b]      = np.matmul(proj_mtx, r_mv).astype(np.float32)
            campos[b]   = np.linalg.inv(r_mv)[:3, 3]
            lightpos[b] = campos[b]

        params = {
            'mvp': mvp,
            'lightpos': lightpos,
            'campos': campos,
            'resolution': [config["render_res"], config["render_res"]]
        }

        # We should mimic the render call from nvdiffmodelling, see:
        # https://github.com/NVlabs/nvdiffmodeling/blob/main/train.py#L344-L346
        """
        # Subdivide & displace optimized mesh
        subdiv_opt_mesh = mesh.subdivide(opt_base_mesh, steps=FLAGS.subdivision)
        opt_detail_mesh = mesh.displace(subdiv_opt_mesh, displacement_map_var, FLAGS.displacement, keep_connectivity=True)
        ...
        _opt_detail = mesh.center_by_reference(
            opt_detail_mesh.eval(params),  # <- NOTE: this is the subdivided mesh, not the base mesh
            ref_mesh_aabb, mesh_scale
        )
        ...
        color_opt = render.render_mesh(
            glctx,
            _opt_detail,  # <- NOTE: this is the subdivided mesh, not the base mesh
            mvp,
            campos,
            lightpos,
            FLAGS.light_power,
            iter_res, 
            spp=iter_spp,
            num_layers=FLAGS.layers,
            msaa=True,
            background=randomBgColor, 
            min_roughness=FLAGS.min_roughness
        )
        """

        ready_mesh = opt_detail_mesh

        if i > config["notex_epochs"]:
            # Render all the images with textures
            log_image = render.render_mesh(
                glctx,
                ready_mesh.eval(params),  # <- NOTE: "ready_mesh" here should be equivalent to "opt_detail_mesh"
                mvp,
                campos,
                lightpos,
                2.0,
                config["render_res"],
                background=bkgs
            )

        else:
            # During the first half of training, render out half the images
            # normally, and half the images without texture (for debugging)
            if config["notex_batch_size"] is None:
                with_tex = config["batch_size"] // 2
            else:
                assert config["notex_batch_size"] <= config["batch_size"]
                with_tex = config["batch_size"] - config["notex_batch_size"]

            if with_tex > 0:

                with_tex_params = {
                    'mvp': mvp[:with_tex],
                    'lightpos': lightpos[:with_tex],
                    'campos': campos[:with_tex],
                    'resolution': [config["render_res"], config["render_res"]]
                }

                with_tex_train_render = render.render_mesh(
                    glctx,
                    ready_mesh.eval(with_tex_params),
                    mvp[:with_tex],
                    campos[:with_tex],
                    lightpos[:with_tex],
                    2.0,
                    config["render_res"],
                    num_layers=2,
                    background=bkgs[:with_tex],
                )
            
            """
            The key difference is:

            material={
                    'kd': no_texture,
            """
            #opt_detail_mesh_no_tex = opt_detail_mesh.clone()
            
            # TODO what's the cleanest way to render out a version of the same mesh
            # with a different texture? is it to create a new mesh with a new materials dict,
            # run auto_normals etc, displacement etc, then render that?
            # or is it to swap the texture file on-the-fly like this?
            #opt_detail_mesh_no_tex = opt_detail_mesh.eval(no_tex_params).clone()
            #regular_tex = opt_detail_mesh_no_tex.material['kd']
            # Swap new texture (random noise) in
            #opt_detail_mesh_no_tex.material['kd'] = no_texture

            no_tex_params = {
                'mvp': mvp[with_tex:],
                'lightpos': lightpos[with_tex:],
                'campos': campos[with_tex:],
                'resolution': [config["render_res"], config["render_res"]],
            }

            reg_tex = opt_material['kd']
            opt_material['kd'] = no_texture

            no_tex_train_render = render.render_mesh(
                glctx,
                ready_mesh.eval(no_tex_params),
                #opt_detail_mesh_no_tex,
                mvp[with_tex:],
                campos[with_tex:],
                lightpos[with_tex:],
                2.0,
                config["render_res"],
                num_layers=2,
                background=bkgs[with_tex:],
            )

            # Swap normal texture back in (if we don't we get perma textureless)
            #opt_detail_mesh_no_tex.material['kd'] = regular_tex
            opt_material['kd'] = reg_tex

            if with_tex > 0:
                log_image = torch.cat([
                    with_tex_train_render,
                    no_tex_train_render
                ])
            else:
                log_image = no_tex_train_render

        # Write out copies of the images every N epochs
        if i % config["log_int"] == 0 and config["debug_log"]:

            log = torchvision.utils.make_grid(log_image[0].unsqueeze(0).permute(0, 3, 1, 2))
            if config["plot"]:
                clear_output()
                plt.imshow( log.permute(1, 2, 0).detach().cpu().numpy() )
                plt.show()
            else:
                torchvision.utils.save_image(log_image.permute(0, 3, 1, 2), os.path.join(config["path"], 'img_%d.png' % i))
            video.ready_image( log.permute(1, 2, 0) )

        log_image = resize(
            log_image.permute(0, 3, 1, 2),
            out_shape=(224, 224) if config["CLIP"] != "ViT-L/14@336px" else (336, 336), # resize to clip
            interp_method=lanczos3
        )

        #################################################
        ######   Loss function and optimization  ########
        #################################################

        image_embeds = clip_model.image_embeds( log_image )
        texts_embeds = clip_model.text_embeds( clip_model.text_tokens(prompts_list=prompts) )

        clip_loss = cosine_avg(image_embeds, texts_embeds)
        loss_t = clip_loss
        log_str = "CLIP Loss = %.3f" % clip_loss.item()

        # e.g., create additional text embedings for things we don't want to see, eg faces and text
        # NOTE: same negative prompts for every image
        # TODO do this once outside the batch loop, no need to do this every time
        if config["negative_text"] is not None:
            neg_clip_losses = []
            for neg_prompt in config["negative_text"]:
                # Create batch dimension (apply same neg prompt for all images)
                neg_prompts = [neg_prompt for _ in range(len(prompts))]
                neg_texts_embeds = clip_model.text_embeds( clip_model.text_tokens(prompts_list=neg_prompts) )
                neg_clip_loss = -cosine_avg(image_embeds, neg_texts_embeds)
                neg_clip_losses.append(neg_clip_loss)

            neg_clip_loss = torch.stack(neg_clip_losses, dim=0).sum(dim=0)
            loss_t += config["negative_text_weight"] * neg_clip_loss
            log_str += " | CLIP Neg Loss = %.3f" % neg_clip_loss.item()

        if "texture" in config["optim"]:
            if uv_mask is not None:
                t_l = kornia.losses.total_variation( (ready_mesh.eval().material['kd'].data[0] * uv_mask).permute(2, 0, 1))
            else:
                t_l = kornia.losses.total_variation( (ready_mesh.eval().material['kd'].data[0]).permute(2, 0, 1))
            loss_t += config["TV_weight"] * t_l
            log_str += " | TVL Tex = %.3f" % t_l.item()

        if "normal" in config["optim"]:
            if uv_mask is not None:
                t_n = kornia.losses.total_variation( (ready_mesh.eval().material['normal'].data[0] * uv_mask).permute(2, 0, 1))
            else:
                t_n = kornia.losses.total_variation( (ready_mesh.eval().material['normal'].data[0]).permute(2, 0, 1))
            loss_t += config["TV_weight"] * t_n
            log_str += " | TVL Nrm = %.3f" % t_n.item()

        if "specular" in config["optim"]:
            if uv_mask is not None:
                t_s = kornia.losses.total_variation( (ready_mesh.eval().material['ks'].data[0] * uv_mask).permute(2, 0, 1))
            else:
                t_s = kornia.losses.total_variation( (ready_mesh.eval().material['ks'].data[0]).permute(2, 0, 1))
            loss_t += config["TV_weight"] * t_s
            log_str = " | TVL Spc = %.3f" % t_s.item()

        if config["laplacian"]:
            # https://github.com/NVlabs/nvdiffmodeling/blob/9abcce13e92efd31976e6781073af53571ece7fd/train.py#L323
            # from: https://github.com/NVlabs/nvdiffmodeling/blob/9abcce13e92efd31976e6781073af53571ece7fd/train.py#L354-L355
            lap_loss = lap_loss_fn.eval(params)

            # default in nvdiffmodeling is a schedule, shown below
            # https://github.com/NVlabs/nvdiffmodeling/blob/9abcce13e92efd31976e6781073af53571ece7fd/train.py#L366
            #
            # From the CLIPMesh paper:
            #   l_t = (l_t-1 - l_min) * 10^(-k * t) + l_min
            #   with k = 10^-6
            #   and l_min = 2% of initial weight
            #
            # ^The two are the exact same, just that in CLIPMesh we assume a provided
            #  initial weight between 10 and 50
            if i == 0:
                if config["laplacian_factor"] is not None:
                    lap_fac = config["laplacian_factor"]
                else:
                    #ratio = 0.1 / lap_loss.item() # Hack that assumes RMSE ~= 0.1
                    #lap_fac = ratio * 0.25
                    raise NotImplementedError("For CLIPMesh need to specify lap_fac, paper recommends 10-50")
                min_lap_fac = lap_fac * 0.02
            else:
                lap_fac = (lap_fac - min_lap_fac) * 10**(-i*0.000001) + min_lap_fac
            # Add laplacian loss to the image loss
            loss_t += lap_fac * lap_loss
            log_str += " | Lap loss (fac) = %.3f (%.3f)" % (lap_loss.item(), lap_fac)

        else:
            raise NotImplementedError("Should be using Lap loss")

        for optimizer in optimizers:
            optimizer.zero_grad()
        loss_t.backward()
        for optimizer in optimizers:
            optimizer.step()

        kd_map_opt.clamp_(min=0, max=1)
        normal_map_opt.clamp_(min=-1, max=1)
        ks_map_opt.clamp_rgb_(minR=0, maxR=1, minG=0.5, maxG=1.0, minB=0.0, maxB=1.0)

        if ds_map_opt is not None:
            torch.clamp(ds_map_opt, min=0, max=1)

        # Update tqdm
        t_loop.set_description(log_str)
        
    # Save the mesh after training is over
    obj.write_obj(
        os.path.join(config["path"]),
        ready_mesh.eval()
    )
