# CLIPMesh

This is an unofficial implementation of CLIPMesh (https://arxiv.org/abs/2203.13333), a method for text2mesh using CLIP.

The results generated using this repo are currently hit-or-miss: for the example prompts shown in the paper and on the website, some are significantly worse than the results shown, but some are comparable. Final mesh quality is not great but it seems to be comparable to Figure 3(i) and (j) in the paper. If you have any suggestions or find any particularly good hyperparameters feel free to open an issue or PR and I'll gladly merge them in.

## Install

```sh
# Clone recursively
git clone --recurse-submodules git@github.com:cpacker/CLIPMesh.git
cd CLIPMesh

# Setup pytorch and cudatoolkit using conda
conda create -n clipmesh-py37 python=3.7
conda activate clipmesh-py37
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=10.2 -c pytorch

# Install the rest of the deps
pip install -r requirements.txt
```

## Usage

Example with a prompt from the paper that works well:

```sh
python clipmesh.py --text "a matte painting of a bonsai tree; trending on artstation"
```
![image](https://user-images.githubusercontent.com/5475622/185275526-56526e27-8fa9-44c9-9c87-dcb3d5d940aa.png)

Example with a prompt from the paper that is significantly worse:

```sh
python clipmesh.py --text "a cowboy hat"
```
![image](https://user-images.githubusercontent.com/5475622/185278786-7b08b149-4d53-4917-b211-4a478467adc4.png)

Note that config file `configs/arxiv.yaml` intends to mimic the algorithm described in the paper as closely as possibly (there currently is no official public implementation). 
The default config (loaded w/o a `--path` arg) has a few differences, e.g., the use of a TV weight on the texture maps (used in the CLIPMesh-SMPLX repo), and negative prompts for "face" and "text" (without them, I found CLIP had a tendency to "stamp" faces and text into textures as well as geometry).
`configs/improved.yaml` has more experimental changes with the goal of matching the performance in the paper (i.e., creating reasonable looking objects for all the the prompts shown in the paper and website).

### Example prompts

<details>
  <summary>Prompts to try from the paper</summary>

    (Figure 2) "a christmas tree with a star on top"
    (Figure 3a) "a 🛸"
    (Figure 3b) "thors hammer"
    (Figure 3c) "a red and blue fire hydrant with flowers round it."
    (Figure 3d) "a cowboy hat"
    (Figure 3e) "a red chair"
    (Figure 3g) "a matte painting of a bonsai tree; trending on artstation"
</details>

<details>
  <summary>Prompts to try from the website</summary>

    an armchair in the shape of an avocado
    a lamp shade
    a wooden table
    a 🥞
    a colorful crotchet candle
    a pyramid of giza
    a professional high quality emoji of a lovestruck cup of boba.
    matte painting of a bonsai tree; trending on artstation
    a red and blue fire hydrant with flowers around it.
    a cowboy hat
    a redbull can
    a UFO
    a milkshake
    salvador dali
    a table with oranges on it
</details>

## License and acknolwedgements

This codebase was built using the [CLIPMesh-SMPLX repo](https://github.com/NasirKhalid24/CLIPMesh-SMPLX), and similarly makes heavy use of [nvdiffmodeling](https://github.com/NVlabs/nvdiffmodeling), which uses the NVIDIA Source Code License.
