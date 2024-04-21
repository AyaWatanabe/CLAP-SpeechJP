from pathlib import Path
import pandas as pd
import typer
from typing_extensions import Annotated
from typing import Tuple, Optional
from rich.progress import track

from src.laion_clap import CLAP_Module
import pandas as pd
from pydub import AudioSegment
import numpy as np
import torch
import pickle
import random

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def get_embs(args: dict) -> Tuple[dict, torch.Tensor]:
    wav_dir = args["wav_dir"]
    model = CLAP_Module(amodel="HuBERT")
    test = pd.read_csv(args["test_path"], index_col=0)
    embs = {"wav_id": [], "audio_emb": []}
    for i in range(5):
        embs[f"prompt_{i+1}"] = []
        embs[f"text_emb_{i+1}"] = []

    model.load_ckpt(args["ckpt_path"])

    for wav_id, v in track(test.iterrows(), description="extract embeddings......"):
        audio_data = AudioSegment.from_wav(wav_dir / f"{wav_id}.wav")
        audio_data = audio_data.set_frame_rate(16000)
        audio_data = np.array(audio_data.get_array_of_samples()).reshape(1, -1)
        audio_embed = model.get_audio_embedding_from_data(x = audio_data, use_tensor=False).flatten()
        embs["wav_id"].append(wav_id)
        embs["audio_emb"].append(audio_embed)
        for i in range(5):
            text_data = v[f"characteristics_prompt_{i+1}"]
            text_embed = model.get_text_embedding(text_data).flatten()
            embs[f"text_emb_{i+1}"].append(text_embed)
            embs[f"prompt_{i+1}"].append(text_data)
    
    logit_scale_a = model.model.logit_scale_a.exp().to(device)
    return embs, logit_scale_a

def ranking(embs: dict, logit_scale_a: torch.Tensor) -> pd.DataFrame:
    df = [None]*5
    for i in range(5):
        audio_emb = torch.from_numpy(np.stack(embs["audio_emb"])).to(device)
        text_emb = torch.from_numpy(np.stack(embs[f"text_emb_{i+1}"])).to(device)
        logit = (logit_scale_a * text_emb @ audio_emb.t()).t().detach().cpu()
        rank = torch.argsort(logit, descending=True)

        df[i] = rank_to_map(embs["wav_id"], embs[f"prompt_{i+1}"], rank)
    df = pd.concat(df)
    return df

def rank_to_map(wav_ids: list, characteristics_prompts: list, rank: torch.Tensor) -> pd.DataFrame:
    df = {"txt": []}
    df["GT_wav"] = []
    for i in [1, 5, 10]:
        df[f"R@{i}"] = []
    df["mAP@10"] = []
    
    for i, txt_id in track(enumerate(characteristics_prompts), description="check ranking......"):
        df["txt"].append(txt_id)
        df["GT_wav"].append(wav_ids[i])
        preds = rank[i][i]
        for j in [1, 5, 10]:
            r = 1 if preds < j else 0.0
            df[f"R@{j}"].append(r)
        ap = (1 / (preds + 1)).item() if preds < 10 else 0.0
        df[f"mAP@10"].append(ap)
    
    df = pd.DataFrame.from_dict(df)
    return df
            

def main(
    ckpt_path: Annotated[str, typer.Option("--ckpt_path", "-c")],
    test_path: Annotated[str, typer.Option("--test_path", "-t")],
    wav_dir: Annotated[str, typer.Option("--wav_dir", "-w")],
    save_rank: Annotated[str, typer.Option("--save_rank", "-s")],
    save_embs: Annotated[Optional[str], typer.Option("--save_embs", "-e")] = None,
):
    args = {
        "ckpt_path": Path(ckpt_path),
        "test_path": Path(test_path),
        "wav_dir": Path(wav_dir),
        "save_embs": Path(save_embs) if save_embs!=None else None,
        "save_rank": Path(save_rank)
    }

    embs, logit_scale_a = get_embs(args)
    if args["save_embs"] is not None:
        with open(args["save_embs"], "wb") as f:
            pickle.dump(embs, f)
        print(f"save embeddings.pkl at {args['save_embs']}")
    rank = ranking(embs, logit_scale_a)
    rank.to_csv(args["save_rank"])
    print(f"save ranking.csv at {args['save_rank']}")


if __name__ == "__main__":
    typer.run(main)