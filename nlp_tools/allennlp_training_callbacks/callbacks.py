import logging
from typing import Dict, Any

import torch
import wandb as wdb
from allennlp.common import Tqdm
from allennlp.training import EpochCallback, GradientDescentTrainer
from allennlp.training.util import get_metrics, description_from_metrics
from tqdm import tqdm

logger = logging.getLogger(__name__)


class OutputWriter():
    def __init__(self, output_file, labeldict, write_labels=False):
        self.epoch = 0
        self.outfile = output_file + ".{}"
        self.writer = open(output_file.format(self.epoch), "w")
        self.labeldict = labeldict
        self.write_labels = write_labels

    def write(self, outs):
        """

        :param outs: a dictionary containing at least the keys "predictions" and "labels". If it also contains
        the key "ids" then they will used to indicise the prediction-label pair in the output file.
        """
        predictions = outs["predictions"]
        labels = outs["labels"]
        ids = outs.get("ids", None)
        if type(predictions) is torch.Tensor:
            predictions = predictions.flatten().tolist()
        else:
            predictions = torch.cat(predictions).tolist()
        if type(labels) is torch.Tensor:
            labels = labels.flatten().tolist()
        else:
            labels = torch.cat(labels).tolist()
        if ids is not None and type(ids) is torch.Tensor:
            ids = ids.flatten().tolist()
        elif ids is not None:
            if type(ids[0]) == list:
                ids = [x for i in ids for x in i]
        assert len(predictions) == len(labels)
        if ids is not None:
            assert len(predictions) == len(ids)
        for i in range(len(predictions)):  # p, l in zip(predictions, labels):
            p, l = predictions[i], labels[i]
            if ids is not None:
                id = ids[i]
            out_str = (id if ids is not None else "") + "\t" + self.labeldict[p]
            if self.write_labels:
                out_str += "\t" + self.labeldict[l]
            out_str += "\n"
            self.writer.write(out_str)

    def close(self):
        self.writer.close()

    def reset(self):
        self.writer.close()
        self.writer = open(self.outfile.format(self.epoch), "w")

    def increment_epoch(self):
        self.epoch += 1

    def set_epoch(self, epoch):
        self.epoch = epoch


@EpochCallback.register("output_writer_callback")
class OutputWriterCallback(EpochCallback):
    def __init__(self, owriter: OutputWriter):
        self.owriter = owriter

    def __call__(self, trainer: GradientDescentTrainer, metrics: Dict[str, Any], epoch: int):
        self.owriter.set_epoch(epoch + 1)
        self.owriter.reset()


class WanDBLogger():
    def __init__(self, metrics_to_report, soft_match=True):
        self.metrics_to_report = metrics_to_report
        self.soft_match = soft_match

    def __call__(self, metrics: Dict[str, Any], epoch:int, prefix:str = None):
        metrics_keys = set()
        if self.soft_match:
            for k in self.metrics_to_report:
                for mk in metrics.keys():
                    if mk.endswith(k):
                        metrics_keys.add(mk)
        else:
            metrics_keys = self.metrics_to_report
        report = {(prefix + "_" if prefix is not None else "") + k: metrics[k] for k in metrics_keys if k in metrics}
        report["step"] = epoch
        wdb.log(report, step=epoch)


@EpochCallback.register("wandbn_training")
class WanDBTrainingCallback(EpochCallback):
    def __init__(self, wandb_logger: WanDBLogger):
        self.wandb_logger = wandb_logger

    def __call__(self, trainer: GradientDescentTrainer, metrics: Dict[str, Any], epoch: int):
        metrics["step"] = epoch
        self.wandb_logger(metrics, epoch)




@EpochCallback.register("test_and_write")
class TestAndWrite(EpochCallback):
    def __init__(self, test_iterator, output_writer: OutputWriter = None, name: str = None,
                 wandb_logger:WanDBLogger = None, is_dev=False, metric_to_track=None):
        self.test_iterator = test_iterator
        self.metric_to_track = metric_to_track
        self.name = name
        self.writer = output_writer
        self.wandb_logger = wandb_logger
        self.is_dev = is_dev

    def __call__(self, trainer: GradientDescentTrainer, metrics: Dict[str, Any], epoch: int):

        trainer.model.get_metrics(True)
        if epoch < 0:
            return
        # for moving_average in self.moving_averages:
        #     moving_average.assign_average_value()

        with torch.no_grad():
            logger.info("Testing")
            trainer.model.eval()
            batches_this_epoch = 0
            val_loss = 0
            bar = tqdm(self.test_iterator, desc="testing")
            for batch_group in bar:
                outs = trainer.batch_outputs(batch_group, for_training=False)
                loss = outs["loss"]
                if self.writer is not None:
                    self.writer.write(outs)

                if loss is not None:
                    # You shouldn't necessarily have to compute a loss for validation, so we allow for
                    # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                    # currently only used as the divisor for the loss function, so we can safely only
                    # count those batches for which we actually have a loss.  If this variable ever
                    # gets used for something else, we might need to change things around a bit.
                    batches_this_epoch += 1
                    val_loss += loss.detach().cpu().numpy()

                # Update the description with the latest metrics
                val_metrics = get_metrics(trainer.model, val_loss, val_loss, batches_this_epoch)
                description = description_from_metrics(val_metrics)
                if self.name is not None:
                    description = "epoch: %d, dataset: %s, %s" % (epoch, self.name, description)
                bar.set_description(description, refresh=False)

            trainer.val_metrics = get_metrics(trainer.model,
                                              val_loss,
                                              val_loss,
                                              batches_this_epoch,
                                              reset=False)
            if self.wandb_logger is not None:
                self.wandb_logger(trainer.val_metrics, epoch, prefix = self.name)
        # If the trainer has a moving average, restore
        # for moving_average in self.moving_averages:
        #     moving_average.restore()

        self.writer.set_epoch(epoch + 1)
        self.writer.reset()
        trainer.model.get_metrics(True)
