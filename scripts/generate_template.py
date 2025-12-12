import argparse
from pathlib import Path

import yaml

def replace_save(src_path: Path, target_path: Path, name: str):
    with open(src_path, "r", encoding="utf-8") as f:
        template_str = f.read()
    template_str = template_str.replace("template", name)
    if not target_path.parent.is_dir():
        target_path.parent.mkdir(parents=True)
    with open(target_path, "w") as f:
        f.write(template_str)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", "-n", required=True, type=str, help="スクリプト名")
    # parser.add_argument(
    #     "--dir", "-d", default="script", type=str, help="スクリプトを配置するディレクトリ"
    # )
    # parser.add_argument(
    #     "--docker", default="cpu", type=str, help="cpu用かgpu用か設定する [cpu/gpu]"
    # )
    # parser.add_argument(
    #     "--no_docker", action="store_true", help="Dockefileを作成しない場合はこのフラグを設定する"
    # )
    args = parser.parse_args()

    # スクリプトの生成
    template_script = Path(".") / "template.py"
    target_script = Path("../scripts")/f"{args.name}/{args.name}.py" # Path(args.dir) / (args.name + ".py")
    replace_save(template_script, target_script, args.name)

    # requirementsの作成
    # target_requirements = Path("requirements") / (args.name + ".txt")
    # target_requirements.touch()

    # configの作成
    template_config = Path(".") / "conf/template.yaml" # Path("script") / "conf" / "template" / "template_default.yaml"
    target_config = Path(".")  / f"conf/{args.name}/{args.name}_default.yaml"  # Path("script") / "conf" / args.name / (args.name + "_default.yaml")
    replace_save(template_config, target_config, args.name)
    root_config = Path(".")  / f"conf/config.yaml" # Path("script") / "conf" / "config.yaml"
    with open(root_config) as f:
        config = yaml.safe_load(f)
    config["defaults"].append({args.name: target_config.stem})
    with open(root_config, "w") as f:
        yaml.safe_dump(config, f)

    # Dockerfileのコピー
    # if not args.no_docker:
    #     if args.docker == "cpu":
    #         template_docker = Path("docker") / "templates" / "Dockerfile"
    #     else:
    #         template_docker = Path("docker") / "templates" / "Dockerfile.gpu"
    #     target_docker = Path("docker") / args.name / template_docker.name
    #     replace_save(template_docker, target_docker, args.name)


if __name__ == "__main__":
    main()
