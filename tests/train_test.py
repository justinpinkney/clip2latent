from hydra import initialize, compose
from scripts import train, generate_dataset

def test_end_to_end(tmp_path):
    """minimal training test"""

    n_samples = per_folder = 100
    gen_dir = tmp_path/"out"
    generate_dataset.main(
        out_dir=gen_dir,
        n_samples=n_samples,
        samples_per_folder=per_folder,
        )
    
    tar_dir = tmp_path/"webdataset"
    generate_dataset.make_webdataset(gen_dir, tar_dir)
    with initialize(config_path="../config"):
        cfg = compose(
            config_name="config",
            overrides=[
                "logging=null",
                f"data.path={tar_dir}/00000.tar",
                "model=small",
                "model.network.num_timesteps=10",
                "train.max_it=10",
                "train.val_it=9",
                ],
            )
    train.main(cfg)

