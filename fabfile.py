from sys import prefix

from fabric import task
from invoke import env, run
from invoke.util import cd


@task
def hello(ctx):
    print("Hello World")

@task
def pipinstall(ctx):
    with ctx.cd(env.path):
        with prefix('source venv/bin/activate'):
            ctx.run('pip install -r requirements.txt')
