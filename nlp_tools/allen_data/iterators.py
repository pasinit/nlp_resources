from allennlp.data import Batch, DataLoader
from allennlp.data.samplers import BucketBatchSampler, MaxTokensBatchSampler
from allennlp.nn.util import move_to_device
from torchtext.data import BucketIterator, batch, pool
import torch

def sorting_key(x):
    return len(x.fields["tokens"]._indexed_tokens["tokens"]["token_ids"])


def get_bucket_iterator(dataset, max_tokens_in_batch,
                        sort_key=("tokens",),
                        sort=False,
                        sort_within_batch=False,
                        repeat=False, is_trainingset=True,
                        device:torch.device = torch.device("cuda", 0),
                        **kwargs):
    batch_sampler = MaxTokensBatchSampler(dataset, max_tokens_in_batch, sort_key)

    return DataLoader(dataset, batch_sampler=batch_sampler)
    # return AllenWSDDatasetBucketIterator(dataset, max_tokens_in_batch, device=device,
    #                                      sort_key=sort_key,
    #                                      sort=sort,
    #                                      sort_within_batch=sort_within_batch,
    #                                      batch_size_fn=get_batch_size,
    #                                      repeat=repeat,
    #                                      train=is_trainingset, **kwargs)


def get_batch_size(ex, num_ex, curr_size):
    ids = ex.fields["tokens"]._indexed_tokens["tokens"]["token_ids"]
    max_len_so_far = curr_size / max(num_ex, 1)
    if max_len_so_far == 0:
        return len(ids) + curr_size
    if len(ids) > max_len_so_far:
        return len(ids) * (num_ex + 1)
    return curr_size + max_len_so_far


class AllenWSDDatasetBucketIterator(BucketIterator):

    def create_batches(self):
        if self.sort:
            self.batches = batch(self.data(), self.batch_size,
                                 self.batch_size_fn)
        else:
            data = sorted(self.dataset, key=self.sort_key)
            self.batches = pool(data, self.batch_size,
                                self.sort_key, self.batch_size_fn,
                                random_shuffler=self.random_shuffler,
                                shuffle=self.shuffle,
                                sort_within_batch=self.sort_within_batch)



    def __len__(self):
        if not hasattr(self, "batches"):
            self.create_batches()
            self.batches_len = len(list(self.batches))
        return self.batches_len

    def __iter__(self):
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a minibatch
                    # be sorted by decreasing order, which requires reversing
                    # relative to typical sort keys
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)
                batch = Batch(minibatch)
                if self.device == 'cuda' or self.device.type == "cuda":
                    batch = move_to_device(batch, self.device.index if self.device.index is not None else 0)
                yield batch.as_tensor_dict(batch.get_padding_lengths())
            if not self.repeat:
                return
