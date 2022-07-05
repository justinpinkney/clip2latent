from scripts.generate_dataset import main

import numpy as np
import pytest

@pytest.mark.parametrize("save_im", [True, False])
def test_generate(tmp_path, save_im):
    n_samples = 100
    per_folder = 50
    out_dir = tmp_path/"out"
    main(out_dir=out_dir, n_samples=n_samples, samples_per_folder=per_folder, save_im=save_im)

    created_folders = list(out_dir.glob("*"))
    assert len(created_folders) == n_samples // per_folder
    for d in created_folders:
        ims = list(d.glob("*.jpg"))
        npy = list(d.glob("*.npy"))

        if save_im:
            assert len(ims) == per_folder
        assert len(npy) == 2*per_folder # latent and embedding

        # Defaults for sg2 and clip vitb/32
        embed = np.load(list(d.glob("*.img_feat.npy"))[0])
        assert embed.shape == (512,)

        latent = np.load(list(d.glob("*.latent.npy"))[0])
        assert latent.shape == (512,)

