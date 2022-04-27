# asr configs
model_path = None
pretrained_name = 'QuartzNet15x5Base-En'
transcribe_batch_size = 1
device = 'cuda:0'

# qna configs
qna_model_name = "qa_squadv1.1_bertbase"
qna_nbest_log = None  # 'qna_nbest.log'
qna_pred_log = None  # 'qna_pred.log'
qna_batch_size = 5
qna_num_samples = 1

# Speech Synthesis configs
ss_model = 'fastpitch_hifigan'
ss_device = 'cuda:0'
