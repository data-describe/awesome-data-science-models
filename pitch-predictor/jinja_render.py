from jinja2 import Environment
import os
import glob
import fire


def run(GCP_PROJECT):
    os.environ["GCP_PROJECT"] = GCP_PROJECT
    env = Environment()
    for path, folders, files in os.walk("."):
        for file in files:
            if os.path.splitext(file)[1] in [".py", ""]:
                filepath = os.path.join(path, file)
                print(filepath)
                with open(filepath, "r") as f:
                    template = env.from_string(f.read())
                    post_template = template.render(**os.environ)
                with open(filepath, "w") as f:
                    f.write(post_template)

if __name__ == "__main__":
    fire.Fire(run)