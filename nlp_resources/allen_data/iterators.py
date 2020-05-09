from allennlp.data import Batch
from torchtext.data import BucketIterator

class AllenWSDDatasetBucketIterator(BucketIterator):
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
                yield batch.as_tensor_dict(batch.get_padding_lengths())
            if not self.repeat:
                return

    @classmethod
    def get_bucket_iterator(cls, dataset, max_tokens_in_batch,
                            sort_key=lambda x: len(x.fields["tokens"]._indexed_tokens["tokens"]["token_ids"]),
                            sort=True,
                            sort_within_batch=False,
                            repeat=False):
        return cls(dataset, max_tokens_in_batch, device="cuda",
                   sort_key=sort_key,
                   sort=sort,
                   sort_within_batch=sort_within_batch, batch_size_fn=AllenWSDDatasetBucketIterator.__get_batch_size,
                   repeat=repeat)

    @staticmethod
    def __get_batch_size(ex, num_ex, curr_size):
        ids = ex.fields["tokens"]._indexed_tokens["tokens"]["token_ids"]
        max_len_so_far = curr_size / max(num_ex, 1)
        if max_len_so_far == 0:
            return len(ids) + curr_size
        if len(ids) > max_len_so_far:
            return len(ids) * (num_ex + 1)
        return curr_size + max_len_so_far
