function deepClone(obj) {
    return JSON.parse(JSON.stringify(obj));
}

function flattenTuningOptionsToSpec(tuningOptions) {
    const paths = [];
    const values = [];

    const walk = (prefix, node) => {
        if (node && typeof node === "object" && !Array.isArray(node)) {
            for (const [k, v] of Object.entries(node)) {
                walk(prefix.concat(k), v);
            }
        } else if (Array.isArray(node)) {
            paths.push(prefix.join("."));
            values.push(node);
        } else {
            throw new TypeError("tuning_options leaf values must be arrays.");
        }
    };

    walk([], tuningOptions || {});
    return { paths, values };
}

function generateIndexCombos(spec) {
    const lengths = spec.values.map((arr) => arr.length);
    if (lengths.length === 0) return [[]];

    const combos = [];
    const cur = new Array(lengths.length).fill(0);

    const rec = (i) => {
        if (i === lengths.length) {
            combos.push(cur.slice());
            return;
        }
        for (let idx = 0; idx < lengths[i]; idx++) {
            cur[i] = idx;
            rec(i + 1);
        }
    };

    rec(0);
    return combos;
}

function combosToOverrides(spec, indexCombo) {
    const overrides = {};
    for (let i = 0; i < spec.paths.length; i++) {
        const path = spec.paths[i];
        const valIdx = indexCombo[i];
        overrides[path] = spec.values[i][valIdx];
    }
    return overrides;
}

function setByDotPath(obj, dotPath, value) {
    const keys = dotPath.split(".");
    let cur = obj;
    for (let i = 0; i < keys.length - 1; i++) {
        const k = keys[i];
        if (cur[k] == null || typeof cur[k] !== "object") {
            cur[k] = {};
        }
        cur = cur[k];
    }
    cur[keys[keys.length - 1]] = value;
}

function applyOverrides(baseConfig, overrides) {
    const cfg = deepClone(baseConfig);
    for (const [dotPath, value] of Object.entries(overrides)) {
        setByDotPath(cfg, dotPath, value);
    }
    return cfg;
}


const baseConfig = {
    name: "SNN_DDQN",
    algorithm: "PPO",
    algorithm_options: {
        learning_rate: "_",
        maxNumTimeSteps: 1000,
        gamma: 0.99,
        clip: "_",
        nUpdatesPerIteration: 10,
        timeStepsPerBatch: 4096,
        entropyCoef: "_",
        advantage_method: "gae",
        gae_lambda: 0.95,
        maxEpisodesCount: 50000
    },
    env: "LunarLander-v3",
    env_options: {},
    network_actor: "snn",
    network_actor_options: {
        hidden_layers: [64, 64],
        snn_tSteps: 25,
        snn_beta: 0.95
    },
    network_critic: "ann",
    network_critic_options: {
        hidden_layers: [64, 64]
    },
    continue_run: true,
    agents: 1,
    extra_info: "",
    max_run_time: 12600,
    stop_learning_at_win_percent: 80,
    train_max_time: 12600,
    upload_to_cloud: false,
    local_backup: true,
    debug: true,
    tuning: true,
    tuning_options: {
        algorithm_options: {
            learning_rate: [0.01, 0.005, 0.001],
            clip: [0.1, 0.2, 0.3],
            entropyCoef: [0.01, 0.03, 0.08]
        },
        network_actor_options: {
            hidden_layers: [[64, 64], [128, 128, 64]],
            snn_beta: [.95, .99]
        }
    },
    availibleStudios: ["studio_1", "studio_2", "studio_3", "studio_4", "studio_5"]
};

const spec = flattenTuningOptionsToSpec(baseConfig.tuning_options);
const combos = generateIndexCombos(spec);
    console.log(spec)
    console.log(combos)

// Compact object: paths + values + index combos
const compact = { paths: spec.paths, values: spec.values, combos };

console.log(`Paths: ${JSON.stringify(compact.paths)}`);
console.log(`Total combinations: ${compact.combos.length}`);

// Show first 5 overrides and reconstructed config snippets
for (let i = 0; i < compact.combos.length; i++) {
    const comboIdxs = compact.combos[i];
    const overrides = combosToOverrides(compact, comboIdxs);
    const variant = applyOverrides(baseConfig, overrides);
    const ao = variant.algorithm_options;
    console.log(comboIdxs)
    console.log(overrides)
    // console.log(
    //     `${i}: idxs=${JSON.stringify(comboIdxs)}, ` +
    //     `lr=${ao.learning_rate}, clip=${ao.clip}, entropyCoef=${ao.entropyCoef}`
    // );
}

// module.exports = { generateGridSearchConfigs };