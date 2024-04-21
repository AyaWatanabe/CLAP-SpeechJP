import torch
from transformers import HubertModel

class Hubert_Mean(HubertModel):
    def forward(self, x, mixup_lambda = None):
        output = x
        # for k, v in output.items():
        #     output[k] = v.to(device, non_blocking=True)
        output = super().forward(output["waveform_input_values"], attention_mask=output["waveform_attention_mask"])
        # output = torch.mean(output.last_hidden_state, dim=1)
        output = output.last_hidden_state
        # mask = (torch.arange(output.shape[-2])[None, :].to(device) < (x["wavelength"]//321).squeeze(0)[:, None]).unsqueeze(2).to(device)
        # output = mask * output
        output = torch.mean(output, dim=1)
        return {"embedding": output}

def create_hubert_model(audio_cfg, enable_fusion=False, fusion_type='None'):
    model = Hubert_Mean.from_pretrained("rinna/japanese-hubert-base")
    return model