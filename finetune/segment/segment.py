"""
Command line interface to run the neural network model!

From the project root directory, do:

    python segment.py fit --config configs/segment_chesapeake.yaml

References:
- https://lightning.ai/docs/pytorch/2.1.0/cli/lightning_cli.html
- https://pytorch-lightning.medium.com/introducing-lightningcli-v2-supercharge-your-training-c070d43c7dd6
"""

from lightning.pytorch.cli import LightningCLI

from finetune.segment.pskelp_datamodule import PSKelpDataModule  # noqa: F401
from finetune.segment.pskelp_model import PSKelpSegmentor  # noqa: F401


# %%
def cli_main():
    """
    Command-line interface to run Segmentation Model with PSKelpDataModule.
    """
    cli = LightningCLI(
        PSKelpSegmentor,
        PSKelpDataModule,
        save_config_kwargs={"overwrite": True},
    )
    return cli


# %%
if __name__ == "__main__":
    cli_main()

    print("Done!")
