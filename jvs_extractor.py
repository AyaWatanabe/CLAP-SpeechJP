from pathlib import Path
import typer
from typing_extensions import Annotated

from src.laion_clap import CLAP_Module
from pydub import AudioSegment
import numpy as np
import torch
import json

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def get_wav_emb(wav_path: str, model: CLAP_Module) -> torch.Tensor:

    audio_data = AudioSegment.from_wav(wav_path)
    audio_data = audio_data.set_frame_rate(16000)
    audio_data = np.array(audio_data.get_array_of_samples()).reshape(1, -1)
    audio_embed = model.get_audio_embedding_from_data(x = audio_data, use_tensor=False).flatten()

    return audio_embed

def main(
    model_path: Annotated[Path, typer.Option("--model_path", "-m")],
    jvs_dir: Annotated[Path, typer.Option("--jvs_dir", "-j")],
    epoch: Annotated[str, typer.Option("--epoch", "-e")] = "latest",
):

    model = CLAP_Module(amodel="HuBERT")
    model.load_ckpt(model_path / "checkpoints" / f"epoch_{epoch}.pt")
    output_dir = model_path / "jvs_embedding"
    output_dir.mkdir(exist_ok=True)

    gender_dict = {}
    with open(jvs_dir / "gender_f0range.txt", "r") as f:
        lines = f.readlines()
    for l in lines[1:]:
        attributes = l.split()
        gender_dict[attributes[0]] = attributes[1]
    with open(output_dir / "gender_dict.json", "w") as f:
        json.dump(gender_dict, f)

    for speaker_id in gender_dict.keys():
        (output_dir / speaker_id).mkdir(exist_ok=True)
        for wav in (jvs_dir / speaker_id / "parallel100" / "wav24kHz16bit").iterdir():
            emb = emb = get_wav_emb(str(wav), model)
            save_path = ((output_dir / speaker_id) / (f"{speaker_id}_{wav.name}")).with_suffix(".pt")
            torch.save(emb, save_path)

    for gender in ["男", "女"]:
        emb = model.get_text_embedding(f"{gender}性が喋っている。")
        save_path = output_dir / f"{gender}.pt"
        torch.save(emb, save_path)
        
    # with tarfile.open(model_path / "jvs_embedding.tar.gz", "w:gz") as tar:
    #     tar.add(output_dir)

if __name__ == "__main__":
    typer.run(main)