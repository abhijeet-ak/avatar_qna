# __author__ Abhijeet Shrivastava
# __ email__ abhijeet@akaiketech.com


import os
from nemo.collections.asr.models.asr_model import ASRModel
from nemo.utils import logging, model_utils
import config


class AutomaticSpeechRecogModel:
    def __init__(self, ):
        if config.model_path is not None:
            # restore model from .nemo file path
            model_cfg = ASRModel.restore_from(restore_path=config.model_path, return_config=True)
            classpath = model_cfg.target  # original class path
            imported_class = model_utils.import_class_by_path(classpath)  # type: ASRModel
            logging.info(f"Restoring model : {imported_class.__name__}")
            self._asr_model = imported_class.restore_from(restore_path=config.model_path,
                                                          map_location=config.device)  # type: ASRModel
            self.model_name = os.path.splitext(os.path.basename(config.model_path))[0]
        else:
            # restore model by name
            self._asr_model = ASRModel.from_pretrained(model_name=config.pretrained_name,
                                                       map_location=config.device)  # type: ASRModel
            self.model_name = config.pretrained_name

    def predict(self, filenames):

        transcriptions = self._asr_model.transcribe(paths2audio_files=filenames,
                                                    batch_size=config.transcribe_batch_size)
        return {fname: trasn for fname, trasn in zip(filenames, transcriptions)}


if __name__ == '__main__':
    model = AutomaticSpeechRecogModel()
    print(model.predict(['~/Downloads/q1.wav']))
