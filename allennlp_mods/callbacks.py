import logging
import math
from typing import Union, List

import torch
from allennlp.common import Tqdm
from allennlp.common.util import lazy_groups_of
from allennlp.training.callbacks import Callback, Events, handle_event, Validate
from allennlp.training.util import get_metrics, description_from_metrics
import wandb as wdb

from allennlp_mods.callback_trainer import MyCallbackTrainer

logger = logging.getLogger(__name__)


class OutputWriter():
    def __init__(self, output_file, labeldict):
        self.epoch = 0
        self.outfile = output_file + ".{}"
        self.writer = open(output_file.format(self.epoch), "w")
        self.labeldict = labeldict

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
            ids = [x for i in ids for x in i]
        assert len(predictions) == len(labels)
        if ids is not None:
            assert len(predictions) == len(ids)
        for i in range(len(predictions)):# p, l in zip(predictions, labels):
            p,l = predictions[i],labels[i]
            if ids is not None:
                id = ids[i]
            out_str = (id if ids is not None else "") + "\t" + self.labeldict[p] + "\t" + self.labeldict[l] + "\n"
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


class OutputWriterCallback(Callback):
    def __init__(self, owriter: OutputWriter):
        self.owriter = owriter

    @handle_event(Events.EPOCH_END)
    def on_epoch_end(self, trainer: MyCallbackTrainer):
        self.owriter.set_epoch(trainer.epoch_number + 1)
        self.owriter.reset()


@Callback.register("wandbn_training")
class WanDBTrainingCallback(Callback):
    @handle_event(Events.EPOCH_END)
    def on_validation(self, trainer: 'MyCallbackTrainer'):
        train_metrics = trainer.train_metrics
        wdb.log(train_metrics)


@Callback.register("validate_and_write")
class ValidateAndWrite(Validate):
    def __init__(self, validation_data, validation_iterator, output_writer: OutputWriter = None, name: str = None,
                 wandb: bool = False):
        super().__init__(validation_data, validation_iterator)
        self.name = name
        self.writer = output_writer
        self.wandb = wandb

    @handle_event(Events.VALIDATE)
    def validate(self, trainer: 'MyCallbackTrainer'):
        trainer.model.get_metrics(True)
        # If the trainer has MovingAverage objects, use their weights for validation.
        for moving_average in self.moving_averages:
            moving_average.assign_average_value()

        with torch.no_grad():
            # We have a validation set, so compute all the metrics on it.
            logger.info("Validating")

            trainer.model.eval()

            num_gpus = len(trainer._cuda_devices)  # pylint: disable=protected-access

            raw_val_generator = self.iterator(self.instances,
                                              num_epochs=1,
                                              shuffle=False)
            val_generator = lazy_groups_of(raw_val_generator, num_gpus)
            num_validation_batches = math.ceil(
                self.iterator.get_num_batches(self.instances) / num_gpus)
            val_generator_tqdm = Tqdm.tqdm(val_generator,
                                           total=num_validation_batches)
            batches_this_epoch = 0
            val_loss = 0
            for batch_group in val_generator_tqdm:
                outs, loss = trainer.batch_outs_and_loss(batch_group, for_training=False)
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
                val_metrics = get_metrics(trainer.model, val_loss, batches_this_epoch)
                description = description_from_metrics(val_metrics)
                if self.name is not None:
                    description = "epoch: %d, dataset: %s, %s" % (trainer.epoch_number, self.name, description)
                val_generator_tqdm.set_description(description, refresh=False)

            trainer.val_metrics = get_metrics(trainer.model,
                                              val_loss,
                                              batches_this_epoch,
                                              reset=False)
            if self.wandb:
                metrics = trainer.val_metrics
                if self.name is not None:
                    metrics = {self.name + "_" + k: v for k, v in metrics.items()}
                metrics["step"] = trainer.epoch_number
                wdb.log(metrics, step=trainer.epoch_number)
        # If the trainer has a moving average, restore
        for moving_average in self.moving_averages:
            moving_average.restore()

        self.writer.set_epoch(trainer.epoch_number + 1)
        self.writer.reset()
        trainer.model.get_metrics(True)
