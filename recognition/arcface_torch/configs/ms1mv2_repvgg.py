from easydict import EasyDict as edict

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "repvgg"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.momentum = 0.9
config.weight_decay = 1e-4
config.batch_size = 128
config.lr = 0.1
config.verbose = 2000
config.dali = False

config.rec = "/content/faces_emore"
config.num_classes = 85742
config.num_image = 5822653
config.num_epoch = 40
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
