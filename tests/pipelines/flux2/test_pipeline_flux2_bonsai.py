import tempfile
import unittest
from pathlib import Path
from unittest import mock

from diffusers import Flux2BonsaiPipeline, Flux2KleinPipeline
from diffusers.pipelines.flux2.bonsai_loading import _ensure_hqq_checkpoint_dir


class Flux2BonsaiLoaderTests(unittest.TestCase):
    def test_pipeline_subclasses_flux2_klein(self):
        self.assertTrue(issubclass(Flux2BonsaiPipeline, Flux2KleinPipeline))

    def test_ensure_hqq_checkpoint_dir_aliases_pytorch_bin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            text_encoder_dir = Path(tmpdir) / "text_encoder"
            text_encoder_dir.mkdir()
            (text_encoder_dir / "config.json").write_text('{"architectures": ["Qwen3ForCausalLM"]}', encoding="utf-8")
            (text_encoder_dir / "pytorch_model.bin").write_bytes(b"bonsai")

            load_dir, temp_dir = _ensure_hqq_checkpoint_dir(text_encoder_dir)

            try:
                self.assertIsNotNone(temp_dir)
                self.assertTrue((load_dir / "config.json").exists())
                self.assertTrue((load_dir / "qmodel.pt").exists())
                self.assertEqual((load_dir / "qmodel.pt").read_bytes(), b"bonsai")
            finally:
                if temp_dir is not None:
                    import shutil

                    shutil.rmtree(temp_dir, ignore_errors=True)

    @mock.patch.object(Flux2BonsaiPipeline, "to", autospec=True, side_effect=lambda self, device: self)
    @mock.patch("diffusers.pipelines.flux2.pipeline_flux2_bonsai.Qwen2TokenizerFast.from_pretrained")
    @mock.patch("diffusers.pipelines.flux2.pipeline_flux2_bonsai.load_bonsai_transformer")
    @mock.patch("diffusers.pipelines.flux2.pipeline_flux2_bonsai.load_bonsai_text_encoder")
    @mock.patch("diffusers.pipelines.flux2.pipeline_flux2_bonsai.load_bonsai_vae")
    @mock.patch("diffusers.pipelines.flux2.pipeline_flux2_bonsai.load_bonsai_scheduler")
    @mock.patch("diffusers.pipelines.flux2.pipeline_flux2_bonsai.resolve_bonsai_snapshot_path")
    def test_from_pretrained_uses_manual_component_loaders(
        self,
        resolve_snapshot_path_mock,
        load_scheduler_mock,
        load_vae_mock,
        load_text_encoder_mock,
        load_transformer_mock,
        tokenizer_from_pretrained_mock,
        to_mock,
    ):
        snapshot_path = Path("/tmp/bonsai-model")
        resolve_snapshot_path_mock.return_value = snapshot_path

        scheduler = mock.Mock()
        vae = mock.Mock()
        vae.config.block_out_channels = (1, 2, 3)
        text_encoder = mock.Mock()
        transformer = mock.Mock()
        tokenizer = mock.Mock()
        load_scheduler_mock.return_value = scheduler
        load_vae_mock.return_value = vae
        load_text_encoder_mock.return_value = text_encoder
        load_transformer_mock.return_value = transformer
        tokenizer_from_pretrained_mock.return_value = tokenizer

        pipe = Flux2BonsaiPipeline.from_pretrained("gabe-engineers/bonsai", device="cuda")

        self.assertIsInstance(pipe, Flux2BonsaiPipeline)
        resolve_snapshot_path_mock.assert_called_once()
        load_scheduler_mock.assert_called_once_with(snapshot_path)
        load_vae_mock.assert_called_once()
        load_text_encoder_mock.assert_called_once()
        load_transformer_mock.assert_called_once()
        tokenizer_from_pretrained_mock.assert_called_once_with(str(snapshot_path), subfolder="tokenizer")
        to_mock.assert_called_once_with(pipe, "cuda")
