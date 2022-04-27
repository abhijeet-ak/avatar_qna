# __author__ Abhijeet Shrivastava
# __ email__ abhijeet@akaiketech.com

import torch
import config
from nemo.collections.tts.models.base import TextToWaveform


class SpeechSynthesisModel:
    def __init__(self):
        if config.ss_model == "fastpitch_hifigan":
            pretrained_model = "tts_en_e2e_fastpitchhifigan"
        elif config.ss_model == "fastspeech2_hifigan":
            pretrained_model = "tts_en_e2e_fastspeech2hifigan"
        else:
            raise NotImplementedError

        self._model = TextToWaveform.from_pretrained(pretrained_model)
        self._model.eval().to(config.ss_device)

    def predict(self, text):
        with torch.no_grad():
            parsed = self._model.parse(text)
            audio = self._model.convert_text_to_waveform(tokens=parsed)[0]
            audio = audio.to('cpu').numpy()
        return audio


if __name__ == '__main__':
    model = SpeechSynthesisModel()
    res = model.predict('Which NFL team represented the AFC at Super Bowl 50?')
    print(res)

    from IPython.display import Audio

    a = Audio(res.detach().cpu().numpy(), rate=22050)
    with open('res.wav', 'wb') as fi:
        fi.write(a.data)
    print(a)
