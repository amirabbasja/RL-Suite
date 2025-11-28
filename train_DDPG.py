from models import *
from huggingface_hub import HfApi, login
from utils import *
import numpy as np
import gymnasium as gym
import torch, os, json, shutil
from DDPG import DDPG

parser = modelParamParser()
args, unknown = parser.parse_known_args()

args.env_options = json.loads(args.env_options)
args.network_actor_options = json.loads(args.network_actor_options)
args.network_critic_options = json.loads(args.network_critic_options)
args.algorithm_options = json.loads(args.algorithm_options)
args.stop_condition = json.loads(args.stop_condition)
try:
    args.extra_info = json.loads(args.extra_info)
except:
    args.extra_info = args.extra_info if type(args.extra_info) == str else args.extra_info

uploadInfo = None
if args.upload_to_cloud:
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
    shutil.copyfile(os.path.join(os.path.dirname(__file__), "DDPG_cinf.json"), os.path.join(runSavePath, "conf.json"))

env = gym.make(args.env)
state, info = env.reset()
stateSize = env.observation_space.shape
assert hasattr(env.action_space, "shape"), "DDPG requires a continuous (Box) action space"
action_dim = int(np.prod(env.action_space.shape))
act_high = torch.tensor(env.action_space.high, dtype=torch.float32)
act_low = torch.tensor(env.action_space.low, dtype=torch.float32)

if args.network_actor == "ann":
    actorNetwork = qNetwork_ANN([stateSize[0], *args.network_actor_options["hidden_layers"], action_dim])
elif args.network_actor == "snn":
    actorNetwork = qNetwork_SNN([stateSize[0], *args.network_actor_options["hidden_layers"], action_dim], beta=args.network_actor_options.get("snn_beta",0.95), tSteps=args.network_actor_options.get("snn_tSteps",25), DEBUG=args.debug)
else:
    raise ValueError(f"Unknown network: {args.network_actor}")

# Critic takes [state, action] concatenated
class CriticANN(torch.nn.Module):
    def __init__(self, in_dim, hidden_layers):
        super().__init__()
        layers = []
        dims = [in_dim, *hidden_layers, 1]
        for i in range(len(dims)-1):
            layers.append(torch.nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2: layers.append(torch.nn.ReLU())
        self.model = torch.nn.Sequential(*layers)
    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.model(x), None

if args.network_critic == "ann":
    criticNetwork = CriticANN(stateSize[0] + action_dim, args.network_critic_options["hidden_layers"])
elif args.network_critic == "snn":
    raise ValueError("SNN critic not supported for DDPG in this setup")
else:
    raise ValueError(f"Unknown network: {args.network_critic}")

optimActor = torch.optim.Adam(actorNetwork.parameters(), lr=args.algorithm_options["learning_rate"])
optimCritic = torch.optim.Adam(criticNetwork.parameters(), lr=args.algorithm_options["learning_rate"])

_networks = {
    "actorNetwork": actorNetwork,
    "criticNetwork": criticNetwork,
    "optimActor": optimActor,
    "optimCritic": optimCritic
}

args = vars(args)
args["env"] = env
args["stateSize"] = stateSize
args["action_bounds"] = {"low": act_low.tolist(), "high": act_high.tolist()}

if args["algorithm"] != "DDPG":
    raise ValueError(f"Algorithm should be DDPG, not {args['algorithm']}")

args["uploadInfo"] = uploadInfo
args["run_save_path"] = runSavePath

agent = DDPG(os.getenv("session_name"), args, _networks)
agent.train()