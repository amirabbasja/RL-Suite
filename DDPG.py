import torch, torch.nn as nn
import numpy as np, random, time, os, json, pandas as pd
from collections import namedtuple, deque
from utils import plotEpisodeReward
class ReplayBufferContinuous:
    def __init__(self, size, dtype, device):
        self.exp = deque([], maxlen=size); self.size = size; self.len = 0
        self.dtype = dtype; self.device = device
    def addNew(self, e): self.exp.appendleft(e); self.len = len(self.exp)
    def sample(self, n):
        mb = random.sample(self.exp, n)
        s = torch.from_numpy(np.array([e.state for e in mb])).to(self.device, dtype=self.dtype)
        a = torch.from_numpy(np.array([e.action for e in mb])).to(self.device, dtype=self.dtype)
        r = torch.from_numpy(np.array([e.reward for e in mb])).to(self.device, dtype=self.dtype)
        ns = torch.from_numpy(np.array([e.nextState for e in mb])).to(self.device, dtype=self.dtype)
        d = torch.from_numpy(np.array([e.done for e in mb]).astype(np.uint8)).to(self.device, dtype=torch.int)
        return s, a, r, ns, d

class DDPG:
    def __init__(self, sessionName, args, networks):
        assert args["algorithm"] == "DDPG"
        _opts = args["algorithm_options"]
        for k in ["learning_rate","gamma","tau","batch","memorySize","nEpisodes","maxNumTimeSteps","noise_std","noise_decay"]:
            assert k in _opts, f"Missing {k} in algorithm_options"
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.dtype = torch.float
        self.sessionName = sessionName
        self.projectName = args["name"]
        self.runSavePath = args["run_save_path"]
        self.env = args["env"]
        self.stateSize = args["stateSize"][0]
        self.bounds = args["action_bounds"]
        self.low = torch.tensor(self.bounds["low"], dtype=self.dtype, device=self.device)
        self.high = torch.tensor(self.bounds["high"], dtype=self.dtype, device=self.device)
        self.stop_condition = args["stop_condition"]
        self.debugMode = args["debug"]; self.localBackup = args["local_backup"]
        self.maxRunTime = args["max_run_time"]
        self.learningRate = _opts["learning_rate"]; self.gamma = _opts["gamma"]; self.tau = _opts["tau"]
        self.batch = _opts["batch"]; self.memorySize = _opts["memorySize"]
        self.nEpisodes = _opts["nEpisodes"]; self.maxNumTimeSteps = _opts["maxNumTimeSteps"]
        self.noise_std = _opts["noise_std"]; self.noise_decay = _opts["noise_decay"]
        self.mem = ReplayBufferContinuous(self.memorySize, self.dtype, self.device)
        self.Actor = networks["actorNetwork"].to(self.device, dtype=self.dtype)
        self.Critic = networks["criticNetwork"].to(self.device, dtype=self.dtype)
        self.OptimActor = networks["optimActor"]; self.OptimCritic = networks["optimCritic"]
        self.TargetActor = type(self.Actor)()
        self.TargetActor.load_state_dict(self.Actor.state_dict()); self.TargetActor = self.TargetActor.to(self.device, dtype=self.dtype)
        self.TargetCritic = type(self.Critic)(self.Critic.model[0].in_features - 0, [])
        self.TargetCritic.load_state_dict(self.Critic.state_dict()); self.TargetCritic = self.TargetCritic.to(self.device, dtype=self.dtype)
        self.agentExp = namedtuple("exp", ["state","action","reward","nextState","done"])
        self.avgWindow = 100
        self.saveFileName = f"{self.projectName}_DDPG.pth"
        self.lstHistory = []; self.avgReward = -float("inf")

    def _action(self, s, noise_sigma):
        a, _ = self.Actor(s)
        a = torch.tanh(a)
        scale = (self.high - self.low) / 2.0
        mean = (self.high + self.low) / 2.0
        a = mean + scale * a
        a = a + torch.randn_like(a) * noise_sigma
        return torch.clamp(a, self.low, self.high)

    def _soft_update(self, target, source, tau):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tp.data * (1 - tau) + sp.data * tau)

    def train(self):
        startTs = time.time(); finishTs = startTs + self.maxRunTime
        episodePointHist = []; latestCheckpoint = 0
        for episode in range(self.nEpisodes):
            s, info = self.env.reset()
            points = 0.0; t0 = time.time(); noise = self.noise_std
            for t in range(self.maxNumTimeSteps):
                s_t = torch.tensor(s, device=self.device, dtype=self.dtype).unsqueeze(0)
                a = self._action(s_t, noise).detach().cpu().numpy()[0]
                ns, r, terminated, truncated, info = self.env.step(a)
                self.mem.addNew(self.agentExp(s, a, r, ns, True if terminated or truncated else False))
                s = ns; points += r
                if self.batch < self.mem.len:
                    state, action, reward, nextState, done = self.mem.sample(self.batch)
                    with torch.no_grad():
                        na = self._action(nextState, 0.0)
                        q_next, _ = self.TargetCritic(nextState, na)
                        y = reward + self.gamma * q_next.squeeze() * (1 - done)
                    q, _ = self.Critic(state, action)
                    loss_c = nn.functional.mse_loss(q.squeeze(), y)
                    self.OptimCritic.zero_grad(); loss_c.backward(); self.OptimCritic.step()
                    qa = self._action(state, 0.0)
                    actor_loss = -self.Critic(state, qa)[0].mean()
                    self.OptimActor.zero_grad(); actor_loss.backward(); self.OptimActor.step()
                    self._soft_update(self.TargetActor, self.Actor, self.tau)
                    self._soft_update(self.TargetCritic, self.Critic, self.tau)
                noise *= self.noise_decay
                if terminated or truncated or finishTs < time.time(): break
            self.lstHistory.append({"episode":episode,"points":points,"timesteps":t,"duration":time.time()-t0,"finalEpisodeReward":points,"state":"terminated" if terminated else "truncated" if truncated else "none","nActionInEpisode":None})
            episodePointHist.append(points)
            self.avgReward = float(np.mean(episodePointHist[-self.avgWindow:])) if len(episodePointHist) >= 1 else -float("inf")
            if self.localBackup and ((episode + 1) % 100 == 0 or episode == 2):
                torch.save({
                    "episode": episode,
                    "actor_network_state_dict": self.Actor.state_dict(),
                    "critic_network_state_dict": self.Critic.state_dict(),
                    "optimizer_actor_state_dict": self.OptimActor.state_dict(),
                    "optimizer_critic_state_dict": self.OptimCritic.state_dict(),
                    "elapsedTime": int(time.time() - startTs),
                    "train_history": self.lstHistory
                }, os.path.join(self.runSavePath, self.saveFileName))
                latestCheckpoint = episode
            if (episode + 1) % 100 == 0 or episode == 2:
                histDf = pd.DataFrame(self.lstHistory)
                plotEpisodeReward(histDf, os.path.join(self.runSavePath, f"episode_rewards.png"))
            if (self.stop_condition.get("maxAvgPoint") is not None and self.stop_condition["maxAvgPoint"] <= self.avgReward) or (self.stop_condition.get("maxEpisodes") is not None and self.stop_condition["maxEpisodes"] <= episode):
                with open(os.path.join(self.runSavePath, f"training_details.json"), 'w') as f:
                    json.dump({"episode": episode, "avg_reward": self.avgReward}, f, indent=2)
                with open(os.path.join(self.runSavePath, f"conf.json"), 'r+') as f:
                    _conf = json.load(f); _conf["finished"] = True; f.seek(0); f.truncate(); json.dump(_conf, f, indent=4)
                with open('conf.json', 'r+') as f:
                    _conf = json.load(f); _conf["finished"] = True; f.seek(0); f.truncate(); json.dump(_conf, f, indent=4)
                return True