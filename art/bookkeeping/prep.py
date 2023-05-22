import os,shutil,json
import sys
from jax.config import config as jconf


###########################################################################
# hacks to avoid GPU OOM
###########################################################################

#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

###########################################################################
# setup config and directories
###########################################################################

#if not os.path.exists('config.json'):
#    shutil.copyfile('defaultconfig.json','config.json')

def getfromconfig(key):
    conf=json.load(open('config.json'))
    return conf[key]


def mk_output_dir(outputdir='outputs'):
    os.makedirs(outputdir,exist_ok=True)

mk_output_dir()

###########################################################################
# debug and parallelzation test
###########################################################################

if 'tl' in sys.argv:
    jconf.update('jax_check_tracer_leaks',True)

if '64' in sys.argv:
    jconf.update('jax_enable_x64',True)

if 'p' in sys.argv:
    import os
    os.environ['XLA_FLAGS'] = "--xla_force_host_platform_device_count=2"

debug=False

if 'db' in sys.argv:
    jconf.update('jax_disable_jit', True)
    debug=True


from art.bookkeeping import session
session.debug=debug