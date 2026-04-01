# art_ab_initio

Code for the paper ["Inventing art styles with no artistic training data"](https://arxiv.org/pdf/2305.12015.pdf).


<p align="center">
  <img src="btower.jpg" alt="Example from the paper combining a subject image, an inspiration image, and the resulting painting." width="720">
</p>

## What this repo is about

This repository is a research codebase for making paintings **without training on human-made artworks**. The core idea is that the model learns to paint its artistic representation of a subject (say, a boring photograph) by controlling the artistic tool (the digital paintbrush) to create an image that it can decode back into the original subject. The artistic style comes from trying to encode the information of the subject under the constraints of the painting medium. 

The paper proposes two ways to create new visual styles while avoiding artistic training data:

1. **Medium+perception-driven procedure**
   - Train an artist model on natural images only.
   - The model does not directly output pixels; it outputs structured painterly actions that are rendered by a differentiable brush engine.
   - A learned decoder tries to reconstruct the original photograph from the painting, which encourages the painting to remain a recognizable but abstract representation of the subject.

2. **Inspiration-driven procedure**
   - Use a second **natural image** as inspiration.
   - Apply style transfer from the inspiration image onto the subject image to produce an intermediate image the paper calls an **imagination** image.
   - Feed that imagination image into a baseline painting procedure to obtain the final artwork.

In the paper, this setup is meant to give an objective guarantee that the model is not reproducing human art styles from training data.

## How the repository maps to the paper

- `art/`: differentiable painting components, including brush geometry, color compositing, neural modules, and training logic.
- `train.py`: trains the **medium+perception-driven** painter on photographs.
- `render_patchwise.py`: applies a trained painter patch-by-patch to render higher-resolution paintings.
- `render_hardcoded.py`: runs a simpler hard-coded baseline used for the **inspiration-driven** pipeline.
- `imagine.sh`: generates imagination images from subject + inspiration pairs using the optional `style_transfer` submodule.
- `pics/`: example subject, inspiration, imagination, and rendered outputs.

If you only want to understand the repository at a high level, the most important split is:

- `train.py` + `art/`: the learned painter from the paper's **medium+perception-driven** method.
- `imagine.sh` + `render_hardcoded.py`: the paper's **inspiration-driven** method.

## Visual examples

### 1) Medium+perception-driven painting

The subject is a photograph, and the output is a painting produced by the learned brush-based artist.

<table>
  <tr>
    <td align="center"><strong>Subject photo</strong></td>
    <td align="center"><strong>Rendered painting</strong></td>
  </tr>
  <tr>
    <td align="center"><img src="pics/content/IMG_3208.JPG" alt="Subject photograph." width="320"></td>
    <td align="center"><img src="pics/rendered_patchwise/painting_IMG_3208.jpg" alt="Painting rendered with the trained model." width="320"></td>
  </tr>
</table>

Another example from the repository assets:

<table>
  <tr>
    <td align="center"><strong>Subject photo</strong></td>
    <td align="center"><strong>Painting</strong></td>
  </tr>
  <tr>
    <td align="center"><img src="pics/content/dandelionpic.jpg" alt="Dandelion subject photograph." width="260"></td>
    <td align="center"><img src="dandelionpainting.jpg" alt="Dandelion painting result." width="260"></td>
  </tr>
</table>

### 2) Inspiration-driven pipeline

For this procedure, the paper uses style transfer only as a way to combine the **content of a subject photo** with the **appearance cues of another natural photo**. The resulting imagination image is then painted by a baseline renderer.

<table>
  <tr>
    <td align="center"><strong>Subject</strong></td>
    <td align="center"><strong>Inspiration</strong></td>
    <td align="center"><strong>Imagination</strong></td>
    <td align="center"><strong>Final painting</strong></td>
  </tr>
  <tr>
    <td align="center"><img src="pics/content/IMG_4988.JPG" alt="Subject image." width="210"></td>
    <td align="center"><img src="pics/inspiration/grill1.jpg" alt="Inspiration image." width="210"></td>
    <td align="center"><img src="pics/imagination/c_IMG_4988_i_grill1.jpg" alt="Imagination image produced by style transfer." width="210"></td>
    <td align="center"><img src="pics/rendered_2/painting_c_IMG_4988_i_grill1.jpg" alt="Painting produced from the imagination image." width="210"></td>
  </tr>
</table>

## What the style-transfer submodule is used for

You do **not** need the `style_transfer` submodule to understand the main idea of this repository, and you do not need it to view the included example results.

Its role in the paper is narrow:

- It is used only in the **inspiration-driven** procedure.
- It creates the intermediate **imagination** image from a subject photo and an inspiration photo.
- The painting code in this repository then turns that imagination image into a final artwork.

So the style-transfer step is **not** the core painter. It is just a way to steer the final painting with another natural image.

## If you are reading the code

You do not need to run everything to understand the project. A useful reading order is:

1. Read the paper's description of the two procedures.
2. Look at `art/artist.py` and `art/trainer.py` for the learned brush-based painter.
3. Look at `train.py` to see how the painter and reconstruction model are optimized together.
4. Look at `render_patchwise.py` to see how a trained model is applied to a full image.
5. Look at `render_hardcoded.py` and `imagine.sh` for the inspiration pipeline.

## Notes

- This repository is best read as **paper code** rather than a polished package.
- Reproducing the exact environment today may take extra work because the repo reflects a specific research snapshot.
- The training objective follows the paper's idea of learning a painting procedure under medium constraints, with reconstruction used to preserve recognizable structure.
- `render_patchwise.py` is the script that most directly reproduces the learned painter described in the paper.
- `render_hardcoded.py` is helpful for understanding the inspiration pipeline in isolation.

## Reference

Nilin Abrahamsen and Jiahao Yao, *Inventing art styles with no artistic training data*.

Paper: https://arxiv.org/pdf/2305.12015.pdf
