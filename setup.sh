#!/usr/bin/env bash
# -*- coding: utf-8 -*-

conda create -p /tmp/bootstrap -c conda-forge mamba conda-lock poetry python=3.11
conda activate /tmp/bootstrap

conda-lock -k explicit --conda mamba
poetry init --python=~3.11
poetry add --lock conda-lock

conda deactivate
rm -rf /tmp/bootstrap

conda create --name mlo --file conda-linux-64.lock
conda activate mlo
poetry install
