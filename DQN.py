import gymnasium as gym
import numpy as np
import torch, torch.nn as nn
import random, time, os, json
import pandas as pd
from collections import namedtuple
from utils import ReplayMemory, plotEpisodeReward, plotTrainingProcess
from models import *

class DQN():
    def __init__(self, sessionName, args, networks):
        assert args["algorithm"] == "DQN", f"Algorithm should be DQN, not {args['algorithm']}"
        assert "stop_condition" in args
        assert "env" in args and "action_space" in args and "stateSize" in args
        assert "network" in args and "network_options" in args
        assert "algorithm_options" in args
        _opts = args["algorithm_options"]
        for k in ["learning_rate","decay","batch","gamma","memorySize","startEbsilon","endEbsilon","numUpdateTS","nEpisodes","maxNumTimeSteps"]:
            assert k in _opts, f"Missing {k} in algorithm_options"

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.dtype = torch.float

        self.sessionName = sessionName
        self.projectName = args["name"]
        self.extraInfo = args.get("extra_info","")
        self.continueLastRun = args["continue_run"]
        self.debugMode = args["debug"]
        self.stopLearningPercent = args["stop_learning_at_win_percent"]
        self.uploadToCloud = args["upload_to_cloud"]
        self.localBackup = args["local_backup"]
        self.NUM_ENVS = args.get("agents",1)
        self.maxRunTime = args["max_run_time"]
        self.runSavePath = args["run_save_path"]
        self.actionSpace = args["action_space"]
        self.nActions = len(self.actionSpace)
        self.env = args["env"]
        self.stateSize = args["stateSize"]
        self.stop_condition = args["stop_condition"]

        self.learningRate = _opts["learning_rate"]
        self.eDecay = _opts["decay"]
        self.gamma = _opts["gamma"]
        self.miniBatchSize = _opts["batch"]
        self.memorySize = _opts["memorySize"]
        self.agentExp = namedtuple("exp", ["state","action","reward","nextState","done"])
        self.nEpisodes = _opts["nEpisodes"]
        self.maxNumTimeSteps = _opts["maxNumTimeSteps"]
        self.ebsilon = _opts["startEbsilon"]
        self.endEbsilon = _opts["endEbsilon"]
        self.numUpdateTS = _opts["numUpdateTS"]
        self.avgWindow = 100

        self.hiddenNodes = args["network_options"]["hidden_layers"]
        self.saveFileName = f"{self.projectName}_{'_'.join([str(l) for l in self.hiddenNodes])}_{self.learningRate}_{self.eDecay}_{self.miniBatchSize}_{self.gamma}_{self.NUM_ENVS}_{self.extraInfo}.pth"

        self.mem = ReplayMemory(self.memorySize, self.dtype, self.device)

        self.qNetwork_model = networks['qNetwork_model'].to(self.device, dtype=self.dtype)
        self.targetQNetwork_model = networks['targetNetwork_model'].to(self.device, dtype=self.dtype)
        self.optimizer_main = networks['optimizer_main']
        self.optimizer_target = networks['optimizer_target']

        self.startEpisode = 0
        self.startEbsilon = None
        self.lstHistory = None
        self.avgReward = -float("inf")

        if self.continueLastRun and os.path.isfile(os.path.join(self.runSavePath, self.saveFileName)):
            try:
                __data = torch.load(os.path.join(self.runSavePath, self.saveFileName), weights_only=False)
                self.qNetwork_model.load_state_dict(__data['qNetwork_state_dict'])
                self.optimizer_main.load_state_dict(__data['qNetwork_optimizer_state_dict'])
                self.targetQNetwork_model.load_state_dict(__data['targetQNetwork_state_dict'])
                self.startEpisode = __data["episode"]
                self.startEbsilon = __data["hyperparameters"]["ebsilon"]
                self.lstHistory = __data["train_history"]
                self.eDecay = __data["hyperparameters"]["eDecay"]
                self.NUM_ENVS = __data["hyperparameters"]["NUM_ENVS"]
                self.mem.loadExperiences(
                    __data["experiences"]["state"],
                    __data["experiences"]["action"],
                    __data["experiences"]["reward"],
                    __data["experiences"]["nextState"],
                    __data["experiences"]["done"]
                )
                self.ebsilon = self.startEbsilon if self.startEbsilon is not None else self.ebsilon
                print("Continuing from episode:", self.startEpisode)
            except Exception as e:
                print("Could not continue from the last run. Starting a new session. Error:", e)

        print(f"Device is: {self.device}")

    def getAction(self, qVal, e, actionSpace, device):
        rndMask = torch.rand(qVal.shape[0], device=device) < e
        actions = torch.zeros(qVal.shape[0], dtype=torch.long, device=device)
        actions[rndMask] = torch.randint(0, len(actionSpace), (int(rndMask.sum().item()),), device=device)
        actions[~rndMask] = torch.argmax(qVal[~rndMask], dim=1)
        return actions

    def updateNetworks(self, timeStep, replayMem, miniBatchSize, C):
        return True if ((timeStep+1) % C == 0 and miniBatchSize < replayMem.len) else False

    def decayEbsilon(self, currE, rate, minE):
        return max(currE * rate, minE)

    def computeLoss(self, experiences, gamma, qNetwork, target_qNetwork):
        state, action, reward, nextState, done = experiences
        target_qNetwork.eval(); qNetwork.eval()
        _targetQValues, _ = target_qNetwork(nextState)
        Qhat = torch.amax(_targetQValues, dim=1)
        yTarget = reward + gamma * Qhat * (1 - done)
        qValues, _ = qNetwork(state)
        qValues = qValues[torch.arange(state.shape[0], dtype=torch.long), action]
        loss = nn.functional.mse_loss(qValues, yTarget)
        return loss

    def fitQNetworks(self, experience, gamma, qNetwork, target_qNetwork):
        __qNetworkModel = qNetwork[0]; __qNetworkOptim = qNetwork[1]
        __targetQNetworkModel = target_qNetwork[0]
        loss = self.computeLoss(experience, gamma, __qNetworkModel, __targetQNetworkModel)
        __qNetworkModel.train(); __targetQNetworkModel.train()
        __qNetworkOptim.zero_grad(); loss.backward(); __qNetworkOptim.step()
        for targetParams, primaryParams in zip(__targetQNetworkModel.parameters(), __qNetworkModel.parameters()):
            targetParams.data.copy_(targetParams.data * (1 - .001) + primaryParams.data * .001)

    def _printProgress(self, delay, lastPrintTime, trainingStartTime, **kwargs):
        if delay < (time.time() - lastPrintTime):
            os.system('cls' if os.name == 'nt' else 'clear')
            lastPrintTime = time.time()
            print(f'ElapsedTime: {int(time.time() - trainingStartTime):<5}s | Episode: {kwargs["episode"]:<5} | Timestep: {kwargs["t"]:<5} | The average of the {self.avgWindow:<5} episodes is: {kwargs["epPointAvg"]:<5.2f}')
            print(f'Latest chekpoint: {kwargs["latestCheckpoint"]} | Speed {kwargs["t"]/(time.time()-kwargs["episodeStartTime"]+1e-9):.1f} tps | ebsilon: {self.ebsilon:.3f}')
            print(f'Remaining time of this run: {kwargs["finishTime"] - time.time():.1f}s')
            print(f"Memory details: {self.mem.len}")
        return lastPrintTime

    def _stopTraining_maxEpisodes(self, episode):
        return True if self.stop_condition["maxEpisodes"] <= episode else False

    def _stopTraining_maxAvgPoint(self, epPointAvg):
        return True if self.stop_condition["maxAvgPoint"] <= epPointAvg else False

    def train(self):
        __trainingStartTime = time.time()
        __finishTime = __trainingStartTime + self.maxRunTime
        __device = self.device; __dtype = self.dtype

        print("Starting training...")
        episodePointHist = []
        self.lstHistory = [] if self.lstHistory is None else self.lstHistory
        self.avgReward = -float("inf") if len(self.lstHistory) == 0 else pd.DataFrame(self.lstHistory).iloc[-self.avgWindow:]["points"].mean()
        latestCheckpoint = 0; _lastPrintTime = 0; self._last100WinPercentage = 0

        for episode in range(self.startEpisode, self.nEpisodes):
            initialSeed = random.randint(1, 1_000_000_000)
            self.state, self.info = self.env.reset(seed=initialSeed)
            points = 0; tempTime = time.time()

            for t in range(self.maxNumTimeSteps):
                qValueForActions, trainInfo = self.qNetwork_model(torch.tensor(self.state, device=__device, dtype=__dtype).unsqueeze(0))
                action = self.getAction(qValueForActions, self.ebsilon, self.actionSpace, self.device).cpu().numpy()[0]
                observation, reward, terminated, truncated, info = self.env.step(action)
                self.mem.addNew(self.agentExp(self.state, action, reward, observation, True if terminated or truncated else False))

                if not self.stopLearningPercent < self._last100WinPercentage:
                    update = self.updateNetworks(t, self.mem, self.miniBatchSize, self.numUpdateTS)
                    if update:
                        experience = self.mem.sample(self.miniBatchSize)
                        self.fitQNetworks(experience, self.gamma, [self.qNetwork_model, self.optimizer_main], [self.targetQNetwork_model, None])

                points += reward
                self.state = observation.copy()

                _lastPrintTime = self._printProgress(1, _lastPrintTime, __trainingStartTime,
                                                     episode=episode, epPointAvg=self.avgReward, finishTime=__finishTime,
                                                     latestCheckpoint=latestCheckpoint, t=t, episodeStartTime=tempTime)
                if terminated or truncated or __finishTime < time.time(): break

            self._last100WinPercentage = np.sum([1 if exp["finalEpisodeReward"] > 75 else 0 for exp in self.lstHistory[-100:]]) / 100
            self.lstHistory.append({
                "episode": episode, "seed": initialSeed, "points": points, "timesteps": t,
                "duration": time.time() - tempTime, "finalEpisodeReward": reward,
                "state": "terminated" if terminated else "truncated" if truncated else "none",
                "nActionInEpisode": None, "totalGradientNorms": None, "layerWiseNorms": None
            })

            episodePointHist.append(points)
            self.avgReward = np.mean(episodePointHist[-self.avgWindow:])
            self.ebsilon = self.decayEbsilon(self.ebsilon, self.eDecay, self.endEbsilon)

            if self.localBackup and ((episode + 1) % 100 == 0 or episode == 2):
                _exp = self.mem.exportExperience()
                backUpData = {
                    "episode": episode,
                    'qNetwork_state_dict': self.qNetwork_model.state_dict(),
                    'qNetwork_optimizer_state_dict': self.optimizer_main.state_dict(),
                    'targetQNetwork_state_dict': self.targetQNetwork_model.state_dict(),
                    'targetQNetwork_optimizer_state_dict': self.optimizer_target.state_dict(),
                    'hyperparameters': {"ebsilon": self.ebsilon, "eDecay": self.eDecay, "NUM_ENVS": self.NUM_ENVS},
                    "elapsedTime": int(time.time() - __trainingStartTime),
                    "train_history": self.lstHistory,
                    "experiences": _exp
                }
                latestCheckpoint = episode
                torch.save(backUpData, os.path.join(self.runSavePath, self.saveFileName))

            if (episode + 1) % 100 == 0 or episode == 2:
                histDf = pd.DataFrame(self.lstHistory)
                plotEpisodeReward(histDf, os.path.join(self.runSavePath, f"episode_rewards.png"))
                plotTrainingProcess(histDf, os.path.join(self.runSavePath, f"training_process.png"))

            if self._stopTraining_maxAvgPoint(self.avgReward) or self._stopTraining_maxEpisodes(episode):
                with open(os.path.join(self.runSavePath, f"training_details.json"), 'w') as f:
                    stopReason = "maxAvgPoint" if self._stopTraining_maxAvgPoint(self.avgReward) else "maxEpisodes"
                    json.dump({"stopReason": stopReason, "episode": episode, "avg_reward": self.avgReward}, f, indent=2)
                with open(os.path.join(self.runSavePath, f"conf.json"), 'r+') as f:
                    _conf = json.load(f); _conf["finished"] = True; f.seek(0); f.truncate(); json.dump(_conf, f, indent=4)
                with open('conf.json', 'r+') as f:
                    _conf = json.load(f); _conf["finished"] = True; f.seek(0); f.truncate(); json.dump(_conf, f, indent=4)
                return True