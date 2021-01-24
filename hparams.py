import json
from utils import *

class HParams:
    def __init__(self):
        self.USE_CPU = False
        self.USE_CUDA = not self.USE_CPU and bool(torch.cuda.is_available())

        self.MODEL_TYPE = "s2s"
        self.N_EPOCHS = 48
        self.LEARNING_RATE = 0.0003
        self.BATCH_SIZE = 128
        self.HIDDEN_SIZE = 100
        self.DROPOUT = 0.3
        self.RNN_LAYERS = 2
        self.DECODER_LEARNING_RATIO = 5.0
        self.MAX_GRAD_NORM = 1
        self.MAX_POST_LEN = 100
        self.MAX_QUES_LEN = 20
        self.MAX_KWD = 5000
        self.ATTN_TYPE = 'dot'
        self.DECODER_CONDITION_TYPE = 'replace'  # or 'none', 'concat'

        # kwd predictor
        self.FREEZE_KWD_MODEL = False
        self.KWD_MODEL_LAYERS = 1
        self.PATIENCE = 4
        self.NEG_KWD_PER = 30
        self.MIN_NEG_KWD = self.NEG_KWD_PER
        self.KWD_PREDICTOR_TYPE = 'cnn'             # or gru
        self.NO_NEG_SAMPLE = False

        # kwd bridge
        self.BRIDGE_NORM_TYPE = 'dropout'           # or layer_norm, batch_norm, sigmoid, none
        self.HARD_KWD_BRIDGE = False

        # decode
        self.BEAM_SIZE = 6
        self.BLOCK_NGRAM_REPEAT = 2
        self.AVOID_REPEAT_WINDOW = 2  # avoid 1-gram repeat with the previous 2
        self.DECODE_USE_KWD_LABEL = False
        self.CLUSTER_KWD = False
        self.KWD_CLUSTERS = 2
        self.SHOW_TOP_KWD = 20
        self.THRESHOLD = -1.0
        self.SAMPLE_KWD = 4
        self.SAMPLE_TOP_K = 6
        self.SAMPLE_TOP_P = 0.9
        self.SAMPLE_DECODE_WORD = False   # currently hard-code top-20, top-0.9 sampling, BEAM_SIZE seqs
        self.USER_FILTER = False
        self.SAVE_EPOCH_INTERVAL = 4
        self.SEED = 77
        seed_everywhere(self.SEED)


        # diverse beam search
        self.DIVERSE_BEAM = False
        self.DIVERSE_GROUP = 3
        self.DIVERSE_LAMBDA = 0.4

        # not useful tricks
        # end2end kwd model training
        self.KWD_LOSS_RATIO = 1.0
        self.UPDATE_WD_EMB = False
        self.SCHEDULED_SAMPLE = False
        self.MIN_TF_RATIO = 0.2
        self.BALANCE_KWD_CLASS = False

        # kwd bridge
        self.WITH_MEMORY = False
        self.MEMORY_HOPS = 2
        self.NO_ENCODER_BRIDGE = False
        self.NO_DECODER_BRIDGE = False


    def get_exp_name(self, kwd_model_prefix=None):
        # distinguish kwd_only model and pipeline model with this, pipeline model won't inherit kwd_only model's name
        exp_name = self.MODEL_TYPE
        # exp_name += "_h{}".format(self.HIDDEN_SIZE)
        exp_name += "_D{}".format(self.DROPOUT)
        # exp_name += "_lr{}".format(self.LEARNING_RATE)
        if kwd_model_prefix is None:
            exp_name += "_{}".format(self.KWD_PREDICTOR_TYPE)
            if not self.NO_NEG_SAMPLE:
                exp_name += "_neg{}".format(self.NEG_KWD_PER)
            else:
                exp_name += "_noneg"
        else:
            exp_name += kwd_model_prefix
        # exp_name += "_kl{}".format(self.KWD_MODEL_LAYERS)
        if self.MODEL_TYPE == "s2s":
            # exp_name += "_dr{}".format(self.DECODER_LEARNING_RATIO)
            exp_name += "_{}".format(self.BRIDGE_NORM_TYPE)
            exp_name += "_{}".format(self.DECODER_CONDITION_TYPE)
            if self.UPDATE_WD_EMB:
                exp_name += "_upwd"
            if self.SCHEDULED_SAMPLE:
                exp_name += "_schsp{}".format(self.MIN_TF_RATIO)
            if self.WITH_MEMORY:
                exp_name += "_{}hop".format(self.MEMORY_HOPS)
            if self.FREEZE_KWD_MODEL:
                exp_name += "_fr"
            else:
                exp_name += "_kr{}".format(self.KWD_LOSS_RATIO)
            if self.NO_ENCODER_BRIDGE:
                exp_name += "_neb"
            if self.NO_DECODER_BRIDGE:
                exp_name += "_ndb"
            if self.HARD_KWD_BRIDGE:
                exp_name += "_hard"
        return exp_name

    def get_decode_name(self, exp_name):
        # exp_name = self.get_exp_name()+"."
        exp_name += "."
        if self.SAMPLE_DECODE_WORD:
            exp_name += "bsp"    # biased sampling
        else:
            exp_name += "bm{}".format(self.BEAM_SIZE)
        if self.DIVERSE_BEAM:
            exp_name += "_div{}".format(self.DIVERSE_GROUP)
        # exp_name += "_br{}".format(self.BLOCK_NGRAM_REPEAT)
        # exp_name += "_bw{}".format(self.AVOID_REPEAT_WINDOW)
        if self.USER_FILTER:
            exp_name += "_filt"
        if self.DECODE_USE_KWD_LABEL:
            if self.CLUSTER_KWD:
                exp_name += "_cluster{}".format(self.KWD_CLUSTERS)
                if self.THRESHOLD > 0:
                    exp_name += "_thr{}".format(self.THRESHOLD)
                else:
                    exp_name += "_sp{}_k{}_p{}".format(self.SAMPLE_KWD, self.SAMPLE_TOP_K, self.SAMPLE_TOP_P)
            else:
                exp_name += "_label"
        elif self.THRESHOLD > 0:
            exp_name += "_thr{}".format(self.THRESHOLD)
        else:
            exp_name += "_sp{}_k{}_p{}".format(self.SAMPLE_KWD, self.SAMPLE_TOP_K, self.SAMPLE_TOP_P)
        exp_name += "_s{}".format(self.SEED)
        return exp_name

    def register_arguments(self, parser):
        for k, v in self.__dict__.items():
            if k == "USE_CUDA":
                continue
            if type(v) == bool:
                parser.add_argument("--{}".format(k.lower()), action='store_true')
            else:
                parser.add_argument("--{}".format(k.lower()), default=v, type=type(v))

    def update(self, args):
        args_dict = dict(args._get_kwargs())
        for k, v in self.__dict__.items():
            if k == "USE_CUDA":
                continue
            if k.lower() in args_dict:
                self.__setattr__(k, args_dict[k.lower()])
        self.MIN_NEG_KWD = self.NEG_KWD_PER
        self.USE_CUDA = not self.USE_CPU and bool(torch.cuda.is_available())
        seed_everywhere(self.SEED)

    def save(self, save_dir):
        with open(save_dir, "w", encoding="utf-8") as f:
            json.dump(hparams.__dict__, f, indent=4, ensure_ascii=False)

    def load(self, load_dir):
        with open(load_dir, encoding="utf-8") as f:
            data = json.load(f)
            for k, v in self.__dict__.items():
                if k in {"USE_CUDA", "USE_CPU", "SEED", "DIVERSE_BEAM", "DIVERSE_GROUP", "DIVERSE_LAMBDA",
                         "DECODE_USE_KWD_LABEL", "THRESHOLD", "BEAM_SIZE", "BLOCK_NGRAM_REPEAT", "AVOID_REPEAT_WINDOW",
                         "SHOW_TOP_KWD", "SAMPLE_KWD", "SAMPLE_TOP_K", "SAMPLE_TOP_P", "SAVE_EPOCH_INTERVAL",
                         "SAMPLE_DECODE_WORD", "BATCH_SIZE", "CLUSTER_KWD", "N_EPOCHS", "KWD_CLUSTERS",
                         "USER_FILTER"}:
                    continue
                if k in data:
                    self.__setattr__(k, data[k])
        self.MIN_NEG_KWD = self.NEG_KWD_PER
        self.USE_CUDA = not self.USE_CPU and bool(torch.cuda.is_available())
        seed_everywhere(self.SEED)

hparams = HParams()