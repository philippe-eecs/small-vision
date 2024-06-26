# pylint: disable=consider-using-from-import
# pylint: disable=logging-fstring-interpolation

from absl import app
from absl import flags
from ml_collections import config_flags
import jax
import wandb
import tensorflow as tf

flags.DEFINE_string("main", default="ae", help="What train main to use")
flags.DEFINE_string("key", default=None, help="Wandb key")
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)

flags.DEFINE_string("workdir", default=None, help="Work unit directory.")
flags.DEFINE_boolean("cleanup", default=False,
                     help="Delete workdir (only) after successful completion.")

# Adds jax flags to the program.
jax.config.parse_flags_with_absl()
# Transfer guard will fail the program whenever that data between a host and
# a device is transferred implicitly. This often catches subtle bugs that
# cause slowdowns and memory fragmentation. Explicit transfers are done
# with jax.device_put and jax.device_get.
#jax.config.update("jax_transfer_guard", "disallow")
# Fixes design flaw in jax.random that may cause unnecessary d2d comms.
jax.config.update("jax_threefry_partitionable", True)

def main(argv):
    jax.distributed.initialize()
    tf.config.set_visible_devices([], "GPU")
    print(f"Jax process index: {jax.process_index()}, Jax process count: {jax.process_count()}")
    print(f"Jax local device count: {jax.local_device_count()}, Jax device count: {jax.device_count()}")

    config = flags.FLAGS.config
    workdir = flags.FLAGS.workdir
    if jax.process_index() == 0:
        if flags.FLAGS.key is not None:
            wandb.login(key=flags.FLAGS.key) #Need to manually pass in key for TPU Cloud pods
        wandb.init(project=config.get("wandb_project", None),
                id=config.get("wandb_id", None),
                entity=config.get("entity", None),
                mode=config.get("wandb_mode", "online"), 
                config=config, dir=workdir)

    if flags.FLAGS.main == 'ae':
        import big_vision.trainers.train_ae as main_loop
        main_loop.main(argv, flags)
    elif flags.FLAGS.main == 'lp_ae':
        import big_vision.trainers.linear_ae as main_loop
        main_loop.main(argv, flags)
    else:
       raise NotImplementedError

if __name__ == "__main__":
  app.run(main)