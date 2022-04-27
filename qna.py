# __author__ Abhijeet Shrivastava
# __ email__ abhijeet@akaiketech.com

# todo rewrite the QnA file as a better replacement of qa_model.py QAModel

import json
import torch
from nemo.collections.nlp.models import QAModel
from nemo.collections.nlp.parts import tensor2list
from nemo.utils import logging
from omegaconf import DictConfig, OmegaConf

import config
from qna_dataloader import SquadDataset


class QnAModel:
    def __init__(self):
        self._model = QAModel.from_pretrained(model_name=config.qna_model_name)  # type:QAModel
        # Switch model to evaluation mode
        self._model.eval()
        self._model.to(config.device)
        logging.set_verbosity(logging.WARNING)

    def predict(self, json_question_file):
        all_preds, all_nbests = self._model.inference(file=json_question_file,
                                                      batch_size=config.qna_batch_size,
                                                      num_samples=config.qna_num_samples,
                                                      output_nbest_file=config.qna_nbest_log,
                                                      output_prediction_file=config.qna_pred_log)
        return all_preds, all_nbests

    def predict_json(self, data: dict):
        """
                Get prediction for unlabeled inference data

                Args:
                    data: inference data

                Returns:
                    model predictions, model nbest list
                """
        # store predictions for all queries in a single list
        all_predictions = []
        all_nbest = []
        try:
            dataloader_cfg = {
                "batch_size": config.qna_batch_size,
                "file": data,
                "shuffle": False,
                "num_samples": config.qna_num_samples,
                'num_workers': 2,
                'pin_memory': False,
                'drop_last': False,
            }
            dataloader_cfg = OmegaConf.create(dataloader_cfg)
            infer_datalayer = self._setup_dataloader_from_config(cfg=dataloader_cfg, mode='infer')

            all_logits = []
            all_unique_ids = []
            for i, batch in enumerate(infer_datalayer):
                input_ids, token_type_ids, attention_mask, unique_ids = batch
                logits = self._model.forward(
                    input_ids=input_ids.to(config.device),
                    token_type_ids=token_type_ids.to(config.device),
                    attention_mask=attention_mask.to(config.device),
                )
                all_logits.append(logits)
                all_unique_ids.append(unique_ids)
            logits = torch.cat(all_logits)
            unique_ids = tensor2list(torch.cat(all_unique_ids))
            s, e = logits.split(dim=-1, split_size=1)
            start_logits = tensor2list(s.squeeze(-1))
            end_logits = tensor2list(e.squeeze(-1))
            (all_predictions, all_nbest, scores_diff) = infer_datalayer.dataset.get_predictions(
                unique_ids=unique_ids,
                start_logits=start_logits,
                end_logits=end_logits,
                n_best_size=self._model._cfg.dataset.n_best_size,
                max_answer_length=self._model._cfg.dataset.max_answer_length,
                version_2_with_negative=self._model._cfg.dataset.version_2_with_negative,
                null_score_diff_threshold=self._model._cfg.dataset.null_score_diff_threshold,
                do_lower_case=self._model._cfg.dataset.do_lower_case,
            )

            test_data = data
            id_to_question_mapping = {}
            for title in test_data:
                for par in title["paragraphs"]:
                    for question in par["qas"]:
                        id_to_question_mapping[question["id"]] = question["question"]

            for question_id in all_predictions:
                all_predictions[question_id] = (id_to_question_mapping[question_id], all_predictions[question_id])

            if config.qna_nbest_log is not None:
                with open(config.qna_nbest_log, "w") as writer:
                    writer.write(json.dumps(all_nbest, indent=4) + "\n")
            if config.qna_pred_log is not None:
                with open(config.qna_pred_log, "w") as writer:
                    writer.write(json.dumps(all_predictions, indent=4) + "\n")

        finally:
            # set mode back to its original value
            # logging.set_verbosity(logging_level)
            pass

        return all_predictions, all_nbest
        pass

    def _setup_dataloader_from_config(self, cfg: DictConfig, mode: str):
        dataset = SquadDataset(
            tokenizer=self._model.tokenizer,
            data=cfg.file,
            doc_stride=self._model._cfg.dataset.doc_stride,
            max_query_length=self._model._cfg.dataset.max_query_length,
            max_seq_length=self._model._cfg.dataset.max_seq_length,
            version_2_with_negative=self._model._cfg.dataset.version_2_with_negative,
            num_samples=cfg.num_samples,
            mode=mode,
            use_cache=self._model._cfg.dataset.use_cache,
        )

        dl = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            collate_fn=dataset.collate_fn,
            drop_last=cfg.drop_last,
            shuffle=cfg.shuffle,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
        return dl


if __name__ == '__main__':
    model = QnAModel()
    print(model.predict_json(json.load(open('/data/projects/digital_assistance/demo/super_bowl.json', 'rb'))["data"]))
    # print(model.predict('C:\\Users\\Akaike\\Downloads\\dev-v1.1.json'))
