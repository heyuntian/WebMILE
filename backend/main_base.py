from defs import MILEAPIControl
from utils import updateCtrl, parse_args
import numpy as np
import os


def createDocker(ctrl, language='python'):
    """
    Create a docker file
    """
    os.system(f'cp {ctrl.root}/{language[:2]}_dockerfile {ctrl.path}/Dockerfile')
#     with open(f'{ctrl.path}/Dockerfline', 'w') as f:
#         f.write(f"FROM python:3.7-slim\n\
# WORKDIR /usr/src/app\n\n\
# RUN apt-get update && apt-get install  -y --no-install-recommends gfortran libopenblas-dev liblapack-dev && rm -rf /var/lib/apt/lists/*\n\
# RUN apt update && apt install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*\n\n\
# COPY {ctrl.path}/requirements.txt .\n\
# RUN pip install --upgrade pip setuptools && \\\n\
#     pip install --no-cache-dir -r requirements.txt\n")


def base_embed(ctrl):
    createDocker(ctrl, language=ctrl.language)
    image_name = f't_{ctrl.jobid}_{ctrl.language[:2]}'  # Each task has a unique image name
    os.system(f'docker build -t {image_name} {ctrl.path}')
    if ctrl.language == 'python':
        os.system(f'docker run -it --name embed_{ctrl.jobid} --rm --volume $(pwd):/usr/src/app --net=host {image_name}:latest python {ctrl.path}/src/embed.py --input {ctrl.coarsen_path} --output {ctrl.coarsen_embed} --embed-dim {ctrl.embed_dim} --workers {ctrl.workers} {ctrl.command}')
    elif ctrl.language == 'java':
        os.system(f'docker run -it --name embed_{ctrl.jobid} --rm --volume $(pwd):/usr/src/app --net=host {image_name}:latest java -jar {ctrl.path}/src/embed.jar --input {ctrl.coarsen_path} --output {ctrl.coarsen_embed} --embed-dim {ctrl.embed_dim} --workers {ctrl.workers} {ctrl.command}')

if __name__ == '__main__':
    seed = 123
    np.random.seed(seed)

    ctrl = MILEAPIControl()
    args = parse_args(useEmbed=True)
    graph, mapping = updateCtrl(ctrl, args, useEmbed=True)

    base_embed(ctrl)
