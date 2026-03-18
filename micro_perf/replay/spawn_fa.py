import argparse
import pathlib
import jsonlines
import shutil
import json



PWD_PATH = pathlib.Path.cwd()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_path", type=pathlib.Path, required=True)
    parser.add_argument("--replay_path", type=pathlib.Path, default=str(PWD_PATH / "temp"))

    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--cache_dtype", type=str, default="int8")
    parser.add_argument("--compute_dtype", type=str, default="bfloat16")
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--step", type=int, default=0)

    parser.add_argument("--q_head_num", type=int, default=80)
    parser.add_argument("--kv_head_num", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)

    args = parser.parse_args()


    dump_dir = pathlib.Path(args.dump_path)
    if not dump_dir.is_dir():
        raise ValueError(f"dump_path {args.dump_path} is not a directory")
    
    temp_dir = pathlib.Path(args.replay_path)
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)


    for dump_file in dump_dir.glob(f"fa_rank_*.jsonl"):
        rank_idx = dump_file.stem.split("_")[-1]

        target_dict = {
            "flash_attention": []
        }

        with jsonlines.open(dump_file, "r") as reader:
            lines = list(reader)
            if args.step > 0:
                lines = lines[args.step:args.step+1:]
            
            for line in lines:
                template_dict = {
                    "arg_type": "llm_batch", 
                    "dtype": args.dtype,
                    "cache_dtype": args.cache_dtype,
                    "compute_dtype": args.compute_dtype,
                    "block_size": args.block_size,
                    "q_head_num": args.q_head_num,
                    "kv_head_num": args.kv_head_num,
                    "head_dim": args.head_dim,
                }
                template_dict["mode"] = "prefill" if line["is_context"] else "decode"
                template_dict["q_lens"] = line["q_lens"]
                template_dict["cache_lens"] = line["kv_lens"]
                template_dict["block_table"] = line["kv_idx"]
                target_dict["flash_attention"].append(template_dict)

        with open(temp_dir / f"rank_{rank_idx}.json", "w") as f:
            json.dump(target_dict, f, indent=4)
                
    