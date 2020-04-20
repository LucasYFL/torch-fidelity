import torch.nn as nn


class FeatureExtractorBase(nn.Module):
    def __init__(self, features_list):
        super(FeatureExtractorBase, self).__init__()
        assert type(features_list) in (list, tuple)
        assert all((a in self.get_provided_features_list() for a in features_list))
        assert len(features_list) == len(set(features_list))
        self.features_list = features_list

    def get_provided_features_list(self):
        raise NotImplementedError

    def convert_features_tuple_to_dict(self, features):
        """
        The only compound return type of the forward function amenable to JIT tracing is tuple.
        This function simply helps to recover the mapping.
        """
        assert type(features) is tuple and len(features) == len(self.features_list)
        return dict(((name, feature) for name, feature  in zip(self.features_list, features)))

    def forward(self, x):
        # do stuff and return a tuple of features in the same order as they appear in self.features_list
        raise NotImplementedError
