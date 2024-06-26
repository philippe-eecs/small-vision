# pylint: disable=consider-using-from-import
# pylint: disable=logging-fstring-interpolation

from mpi4py import MPI
comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
SIZE = comm.Get_size()
from absl import app
from absl import flags
from ml_collections import config_flags
import jax
import wandb

flags.DEFINE_string("address", default=None, help="Coordinator Address")
flags.DEFINE_string("port", default="1234", help="Coordinator Port")
flags.DEFINE_string("main", default="ae", help="What train main to use")

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)

flags.DEFINE_string("workdir", default=None, help="Work unit directory.")
flags.DEFINE_integer("num_devices", default=3, help="Number of GPUs to use.")
flags.DEFINE_boolean("cleanup", default=False,
                     help="Delete workdir (only) after successful completion.")

jax.config.parse_flags_with_absl()
jax.config.update("jax_threefry_partitionable", True)

def main(argv):
    jax.distributed.initialize(coordinator_address=flags.FLAGS.address + ':' + flags.FLAGS.port,
                                num_processes=SIZE,
                                process_id=RANK,
                                local_device_ids=list(range(flags.FLAGS.num_devices)))

    print(f"Jax process index: {jax.process_index()}, Jax process count: {jax.process_count()}")
    print(f"Jax local device count: {jax.local_device_count()}, Jax device count: {jax.device_count()}")

    config = flags.FLAGS.config
    workdir = flags.FLAGS.workdir
    if jax.process_index() == 0:
        wandb.init(project=config.get("wandb_project", None),
                id=config.get("wandb_id", None),
                entity=config.get("entity", None),
                mode=config.get("wandb_mode", "online"), 
                config=config, dir=workdir)

    if flags.FLAGS.main == 'ae':
       import big_vision.trainers.train_ae as main_loop
       main_loop.main(argv, flags)
    else:
       raise NotImplementedError

if __name__ == "__main__":
  app.run(main)