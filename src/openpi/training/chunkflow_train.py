"""Pure supervised-objective composition for ChunkFlow training."""

from openpi.models.chunkflow_objectives import combine_supervised_losses
from openpi.models.chunkflow_objectives import coordinate_aligned_predicted_history
from openpi.models.chunkflow_objectives import share_boundary_noise
from openpi.training.chunkflow_batch import ChunkFlowTrainBatch
from openpi.training.chunkflow_batch import PairedChunkBatch

__all__ = [
    "batch_observation",
    "batch_with_step",
    "combine_supervised_losses",
    "coordinate_aligned_predicted_history",
    "share_boundary_noise",
]


def batch_observation(batch):
    """Return the current observation from legacy or paired training batches."""

    if isinstance(batch, ChunkFlowTrainBatch):
        return batch.supervised.observation
    if isinstance(batch, PairedChunkBatch):
        return batch.observation
    return batch[0]


def batch_with_step(batch, step):
    """Attach the scalar optimizer step to paired batches."""

    if isinstance(batch, ChunkFlowTrainBatch):
        return batch.replace(supervised=batch.supervised.replace(step=step))
    if isinstance(batch, PairedChunkBatch):
        return batch.replace(step=step)
    return batch
