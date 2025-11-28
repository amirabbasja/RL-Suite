from models import *
from huggingface_hub import HfApi, login
import os, json, shutil
from utils import *
import numpy as np
import gymnasium as gym
import torch
from DQN import DQN

parser = modelParamParser()
args, unknown = parser.parse_known_args()

args.env_options = json.loads(args.env_options)
args.network_options = json.loads(args.network_options)
args.algorithm_options = json.loads(args.algorithm_options)
args.stop_condition = json.loads(args.stop_condition)
try:
    args.extra_info = json.loads(args.extra_info)
except:
    args.extra_info = args.extra_info if type(args.extra_info) == str else args.extra_info

uploadInfo = None
if args.upload_to_cloud:
    huggingface_read = os.getenv("huggingface_read")
    huggingface_write = os.getenv("huggingface_write")
    repoID = os.getenv("repo_ID")
    api = HfApi()
    login(token=huggingface_write)
    uploadInfo = {"platform":"huggingface","api":api,"repoID":repoID,"dirName":"","private":False,"replace":True}

if args.finished:
    print("Training is already finished. Exiting...")
    exit()

continueLastRun = args.continue_run
_, runSavePath = get_next_run_number_and_create_folder(continueLastRun, args)

if "--forcedconfig" in unknown:
    _config = json.loads(unknown[unknown.index("--forcedconfig") + 1])
    with open(os.path.join(runSavePath, "conf.json"), 'w') as f:
        json.dump(_config, f, indent=4)
else:
    shutil.copyfile(os.path.join(os.path.dirname(__file__), "DQN_cinf.json"), os.path.join(runSavePath, "conf.json"))

env = gym.make(args.env)
state, info = env.reset()
stateSize = env.observation_space.shape
nActions = env.action_space.n
actionSpace = np.arange(nActions).tolist()

if args.network == "ann":
    qNetwork_model = qNetwork_ANN([stateSize[0], *args.network_options["hidden_layers"], nActions])
    targetQNetwork_model = qNetwork_ANN([stateSize[0], *args.network_options["hidden_layers"], nActions])
elif args.network == "snn":
    qNetwork_model = qNetwork_SNN([stateSize[0], *args.network_options["hidden_layers"], nActions], beta=args.network_options.get("snn_beta",0.95), tSteps=args.network_options.get("snn_tSteps",25), DEBUG=args.debug)
    targetQNetwork_model = qNetwork_SNN([stateSize[0], *args.network_options["hidden_layers"], nActions], beta=args.network_options.get("snn_beta",0.95), tSteps=args.network_options.get("snn_tSteps",25), DEBUG=args.debug)
else:
    raise ValueError(f"Unknown network: {args.network}")

targetQNetwork_model.load_state_dict(qNetwork_model.state_dict())

optimizer_main = torch.optim.Adam(qNetwork_model.parameters(), lr=args.algorithm_options["learning_rate"])
optimizer_target = torch.optim.Adam(targetQNetwork_model.parameters(), lr=args.algorithm_options["learning_rate"])

_networks = {
    "qNetwork_model": qNetwork_model,
    "targetNetwork_model": targetQNetwork_model,
    "optimizer_main": optimizer_main,
    "optimizer_target": optimizer_target
}

args = vars(args)
args["action_space"] = actionSpace
args["env"] = env
args["stateSize"] = stateSize

if args["algorithm"] != "DQN":
    raise ValueError(f"Algorithm should be DQN, not {args['algorithm']}")

args["uploadInfo"] = uploadInfo
args["run_save_path"] = runSavePath

agent = DQN(os.getenv("session_name"), args, _networks)
agent.train()