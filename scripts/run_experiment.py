#!/usr/bin/env python3
import os
import uuid
import yaml
import json
import argparse
import shutil
import hashlib
from datetime import datetime, timezone
from dotenv import load_dotenv

def get_git_commit():
    try:
        import subprocess
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
        return commit
    except Exception:
        return "unknown"

def hash_file(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as afile:
        hasher.update(afile.read())
    return hasher.hexdigest()

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the LLM affective reactivity experiment.")
    parser.add_argument("--config", type=str, default="configs/experiment_main.yaml", help="Path to config file.")
    parser.add_argument("--mode", type=str, choices=["dry-run", "api"], default="dry-run", help="Execution mode.")
    args = parser.parse_args()

    with open(args.config) as f:
        exp_config = yaml.safe_load(f)
    with open("configs/prompts.yaml") as f:
        prompts_config = yaml.safe_load(f)
    with open("configs/models.yaml") as f:
        models_config = yaml.safe_load(f)
    
    print(f"Running experiment in {args.mode} mode...")
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{args.mode.replace('-', '')}"
    print(f"Run ID: {run_id}")
    
    phase = exp_config["experiment"].get("phase", "preliminary")
    out_dir = os.path.join("results/raw", phase, run_id)
    os.makedirs(out_dir, exist_ok=True)
    
    # Save metadata
    metadata = {
        "run_id": run_id,
        "mode": args.mode,
        "config_file": args.config,
        "project_name": exp_config["experiment"].get("name"),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    shutil.copy2(args.config, os.path.join(out_dir, "config_snapshot.yaml"))
    
    responses_path = os.path.join(out_dir, "responses.jsonl")
    
    # Pre-compute prompt hashes
    prompt_hashes = {}
    for condition in exp_config["conditions"]:
        p_info = prompts_config["prompts"][condition]
        prompt_hashes[condition] = {
            "prompt_id": p_info["id"],
            "hash": hash_file(p_info["file"]) if os.path.exists(p_info["file"]) else "missing"
        }
    
    # Pre-compute system prompt hash
    sys_info = prompts_config["prompts"].get("system")
    if sys_info:
        system_prompt_id = sys_info["id"]
        system_prompt_hash = hash_file(sys_info["file"]) if os.path.exists(sys_info["file"]) else "missing"
    else:
        system_prompt_id = None
        system_prompt_hash = None

    
    if args.mode == "dry-run":
        # Generate dummy dry-run results
        import pandas as pd
        stimuli_path = "data/processed/stimuli.csv"
        if os.path.exists(stimuli_path):
            df = pd.read_csv(stimuli_path)
            stimulus_ids = df["stimulus_id"].tolist()
        else:
            stimulus_ids = ["emobank_0001", "emobank_0002"]
            
        models = [m["id"] for m in models_config["models"] if m.get("enabled", False)]
        if not models:
            models = ["dummy-model"]
            
        repetitions = exp_config["experiment"].get("repetitions", 1)
        
        with open(responses_path, "a") as f:
            for m_id in models:
                for rep in range(1, repetitions + 1):
                    # 1. Baseline (per model x rep)
                    baseline_id = f"baseline_{m_id}_rep{rep:02d}"
                    baseline_res = {
                        "run_id": run_id,
                        "request_id": str(uuid.uuid4()),
                        "stimulus_id": None,
                        "baseline_id": baseline_id,
                        "source_dataset": "EmoBank",
                        "annotation_perspective": exp_config["experiment"].get("annotation_perspective", "reader"),
                        "model_provider": "dummy_provider",
                        "model_id": m_id,
                        "condition": "baseline",
                        "repetition": rep,
                        "temperature": exp_config["inference"].get("temperature", 0.0),
                        "top_p": exp_config["inference"].get("top_p", 1.0),
                        "seed": exp_config["experiment"].get("random_seed"),
                        "system_prompt_id": system_prompt_id,
                        "system_prompt_hash": system_prompt_hash,
                        "prompt_id": prompt_hashes["baseline"]["prompt_id"],
                        "prompt_hash": prompt_hashes["baseline"]["hash"],
                        "parsed_valence": 5.0, # Neutral baseline on 1-9 scale
                        "parsed_arousal": 5.0,
                        "raw_response_text": '{"valence": 5, "arousal": 5}',
                        "parse_status": "success",
                        "failure_reason": None,
                        "request_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "response_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "latency_ms": 100,
                        "code_commit": get_git_commit(),
                        "config_hash": hash_file(args.config)
                    }
                    f.write(json.dumps(baseline_res) + "\n")
                    
                    # 2. Recognition, Affective Reception, & Empathic Response
                    for s_id in stimulus_ids:
                        for cond in ["recognition", "affective_reception", "empathic_response"]:
                            cond_res = dict(baseline_res)
                            cond_res["request_id"] = str(uuid.uuid4())
                            cond_res["stimulus_id"] = s_id
                            cond_res["condition"] = cond
                            cond_res["prompt_id"] = prompt_hashes[cond]["prompt_id"]
                            cond_res["prompt_hash"] = prompt_hashes[cond]["hash"]
                            
                            if cond == "empathic_response":
                                cond_res["parsed_valence"] = None
                                cond_res["parsed_arousal"] = None
                                cond_res["raw_response_text"] = "I can imagine that was very difficult for you."
                            else:
                                # Mock some variation for 1-9 scale
                                v_val = 7.0 if cond == "affective_reception" else 6.0
                                a_val = 4.0 if cond == "affective_reception" else 5.0
                                cond_res["parsed_valence"] = v_val
                                cond_res["parsed_arousal"] = a_val
                                cond_res["raw_response_text"] = f'{{"valence": {v_val}, "arousal": {a_val}}}'
                            
                            f.write(json.dumps(cond_res) + "\n")
                            
        print(f"Dry-run finished. Dummy results saved to {out_dir}/")
    else:
        print("API mode is not fully implemented yet. Use --mode dry-run.")

if __name__ == "__main__":
    main()
