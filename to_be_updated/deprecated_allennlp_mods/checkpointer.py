# import traceback
#
# from allennlp.common.checks import ConfigurationError
# from allennlp.training.checkpointer import Checkpointer
#
#
# class MyCheckpoint(Checkpointer):
#     def __init__(self, checkpointer: Checkpointer, autoload_last_checkpoint=False):
#         super().__init__(checkpointer)
#         self.autoload_last_checkpoint = autoload_last_checkpoint
#
#     def restore_checkpoint(self):
#         # Restores the model and training state from the last saved checkpoint if self.autoload_last_checkpoint is set
#         # to True.
#         # This includes an epoch count and optimizer state, which is serialized separately
#         # from model parameters. This function should only be used to continue training -
#         # if you wish to load a model for inference/load parts of a model into a new
#         # computation graph, you should use the native Pytorch functions:
#         # `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``
#
#         # If ``self._serialization_dir`` does not exist or does not contain any checkpointed weights,
#         # this will do nothing.
#         if not self.autoload_last_checkpoint:
#             return
#
#         try:
#             model_state, training_state = self.checkpointer.restore_checkpoint()
#         except RuntimeError:
#             traceback.print_exc()
#             raise ConfigurationError(
#                 "Could not recover training from the checkpoint.  "
#                 "Did you mean to output to a different serialization directory "
#                 "or delete the existing serialization directory?"
#             )
#
#         if not training_state:
#             # No checkpoint to restore, start at 0
#             return
#
#         trainer.model.load_state_dict(model_state)
#
#         # Restore state_dict attrs
#         for attr in self.state_dict_attrs:
#             state_attr = getattr(trainer, attr)
#             if state_attr is not None:
#                 state_attr.load_state_dict(training_state[attr])
#
#         # Restore other attrs
#         for attr in self.other_attrs:
#             setattr(trainer, attr, training_state[attr])
#
#         # Restore callback attrs
#         for callback in trainer.handler.callbacks():
#             callback.restore_training_state(training_state)
#
#         if isinstance(training_state["epoch"], int):
#             trainer.epoch_number = training_state["epoch"] + 1
#         else:
#             trainer.epoch_number = int(training_state["epoch"].split(".")[0]) + 1
