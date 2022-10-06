import gradio as gr
import torch
from clip2latent import models
from PIL import Image

device = "cuda"
model_choices = {
    "faces": {
        "checkpoint": "https://huggingface.co/lambdalabs/clip2latent/resolve/main/ffhq-sg2-510.ckpt",
        "config": "https://huggingface.co/lambdalabs/clip2latent/resolve/main/ffhq-sg2-510.yaml",
        },
    "landscape": {
        "checkpoint": "https://huggingface.co/lambdalabs/clip2latent/resolve/main/lhq-sg3-410.ckpt",
        "config": "https://huggingface.co/lambdalabs/clip2latent/resolve/main/lhq-sg3-410.yaml",
    }
}

model_cache = {}
for k, v in model_choices.items():
    checkpoint = v["checkpoint"]
    cfg_file = v["config"]
    # Moving to the cpu seems to break the model, so just put all on the gpu
    model_cache[k] = models.Clip2StyleGAN(cfg_file, device, checkpoint)

@torch.no_grad()
def infer(prompt, model_select, n_samples, scale):
    model = model_cache[model_select]
    images, _ = model(prompt, n_samples_per_txt=n_samples, cond_scale=scale, skips=250, clip_sort=True)
    images = images.cpu()
    make_im = lambda x: (255*x.clamp(-1, 1)/2 + 127.5).to(torch.uint8).permute(1,2,0).numpy()
    images = [Image.fromarray(make_im(x)) for x in images]
    return images


css = """
        a {
            color: inherit;
            text-decoration: underline;
        }
        .gradio-container {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .gr-button {
            color: white;
            border-color: #9d66e5;
            background: #9d66e5;
        }
        input[type='range'] {
            accent-color: #9d66e5;
        }
        .dark input[type='range'] {
            accent-color: #dfdfdf;
        }
        .container {
            max-width: 730px;
            margin: auto;
            padding-top: 1.5rem;
        }
        #gallery {
            min-height: 22rem;
            margin-bottom: 15px;
            margin-left: auto;
            margin-right: auto;
            border-bottom-right-radius: .5rem !important;
            border-bottom-left-radius: .5rem !important;
        }
        #gallery>div>.h-full {
            min-height: 20rem;
        }
        .details:hover {
            text-decoration: underline;
        }
        .gr-button {
            white-space: nowrap;
        }
        .gr-button:focus {
            border-color: rgb(147 197 253 / var(--tw-border-opacity));
            outline: none;
            box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
            --tw-border-opacity: 1;
            --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
            --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
            --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
            --tw-ring-opacity: .5;
        }
        #advanced-options {
            margin-bottom: 20px;
        }
        .footer {
            margin-bottom: 45px;
            margin-top: 35px;
            text-align: center;
            border-bottom: 1px solid #e5e5e5;
        }
        .footer>p {
            font-size: .8rem;
            display: inline-block;
            padding: 0 10px;
            transform: translateY(10px);
            background: white;
        }
        .dark .logo{ filter: invert(1); }
        .dark .footer {
            border-color: #303030;
        }
        .dark .footer>p {
            background: #0b0f19;
        }
        .acknowledgments h4{
            margin: 1.25em 0 .25em 0;
            font-weight: bold;
            font-size: 115%;
        }
"""

examples = [
    [
        'a photograph of a happy person wearing sunglasses by the sea',
        'faces',
        2,
        2,
    ],
    [
        'a photograph of Captain Jean Luc Picard',
        'faces',
        2,
        2,
    ],
    [
        'a mountain in the middle of the sea',
        'landscape',
        2,
        2,
    ],
    [
        'The sun setting over the sea',
        'landscape',
        2,
        2,
    ],
]

def main():
    block = gr.Blocks(css=css)

    with block:
        gr.HTML(
            """
                <div style="text-align: center; max-width: 650px; margin: 0 auto;">
                <div>
                    <img class="logo" src="https://lambdalabs.com/static/images/lambda-logo.svg" alt="Lambda Logo"
                        style="margin: auto; max-width: 7rem;">
                    <h1 style="font-weight: 900; font-size: 3rem;">
                    clip2latent
                    </h1>
                </div>
                <p style="font-size: 94%">
                    Official demo for <em>clip2latent: Text driven sampling of a pre-trained StyleGAN using denoising diffusion and CLIP</em>, accepted to BMVC 2022
                </p>
                <p style="margin-bottom: 10px; font-size: 94%">
                    Get the <a href="https://github.com/justinpinkney/clip2latent">code on GitHub</a>, see the <a href="#">paper on Arxiv</a>.
                </p>
                </div>
            """
        )
        with gr.Group():
            with gr.Box():
                with gr.Row().style(mobile_collapse=False, equal_height=True):
                    text = gr.Textbox(
                        label="Enter your prompt",
                        show_label=False,
                        max_lines=1,
                        placeholder="Enter your prompt",
                    ).style(
                        border=(True, False, True, True),
                        rounded=(True, False, False, True),
                        container=False,
                    )
                    btn = gr.Button("Generate image").style(
                        margin=False,
                        rounded=(False, True, True, False),
                    )

            gallery = gr.Gallery(
                label="Generated images", show_label=False, elem_id="gallery"
            ).style(grid=[2], height="auto")


            with gr.Row(elem_id="advanced-options"):
                model_select = gr.Dropdown(label="Model", choices=["faces", "landscape"], value="faces",)
                samples = gr.Slider(label="Images", minimum=1, maximum=4, value=2, step=1)
                scale = gr.Slider(
                    label="Guidance Scale", minimum=0, maximum=10, value=2, step=0.5
                )


            ex = gr.Examples(examples=examples, fn=infer, inputs=[text, model_select, samples, scale], outputs=gallery, cache_examples=False)
            ex.dataset.headers = [""]

            text.submit(infer, inputs=[text, model_select, samples, scale], outputs=gallery)
            btn.click(infer, inputs=[text, model_select, samples, scale], outputs=gallery)
            gr.HTML(
                """
                    <div class="footer">
                        <p> Gradio Demo by Lambda Labs
                        </p>
                    </div>
                    <div class="acknowledgments">
                        <img src="https://raw.githubusercontent.com/justinpinkney/clip2latent/main/images/headline-large.jpeg"></img>
                        <br>
                        <h2 style="font-size:1.5em">clip2latent: Text driven sampling of a pre-trained StyleGAN using denoising diffusion and CLIP</h2>
                        <p>Justin N. M. Pinkney and Chuan Li @ <a href="https://lambdalabs.com/">Lambda Inc.</a>
                        <br>
                        <br>
                        <em>Abstract:</em>
                        We introduce a new method to efficiently create text-to-image models from a pre-trained CLIP and StyleGAN.
                        It enables text driven sampling with an existing generative model without any external data or fine-tuning.
                        This is achieved by training a diffusion model conditioned on CLIP embeddings to sample latent vectors of a pre-trained StyleGAN, which we call <em>clip2latent</em>.
                        We leverage the alignment between CLIP’s image and text embeddings to avoid the need for any text labelled data for training the conditional diffusion model.
                        We demonstrate that clip2latent allows us to generate high-resolution (1024x1024 pixels) images based on text prompts with fast sampling, high image quality, and low training compute and data requirements.
                        We also show that the use of the well studied StyleGAN architecture, without further fine-tuning, allows us to directly apply existing methods to control and modify the generated images adding a further layer of control to our text-to-image pipeline.
                        </p>
                        <br>
                        <p>Trained using <a href="https://lambdalabs.com/service/gpu-cloud">Lambda GPU Cloud</a></p>
                </div>
            """
            )

    block.queue()
    block.launch()


if __name__ == "__main__":
    main()