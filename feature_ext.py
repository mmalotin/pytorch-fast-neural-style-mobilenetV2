from functools import partial


class FeatureExtractor():
    def __init__(self, model, idxs):
        self.__list = [0] * len(idxs)
        self.__hooks = []
        self.__modules = []
        self.__model_to_list(model)
        self.__create_hooks(idxs)

    def __create_hooks(self, idxs):
        help_idxs = list(range(len(idxs)))
        for i, idx in zip(idxs, help_idxs):
            fun = partial(self.__hook_fn, idx=idx)
            hook = self.__modules[i].register_forward_hook(fun)
            self.__hooks.append(hook)

    def __model_to_list(self, model):
        if list(model.children()) == []:
            self.__modules.append(model)
        for ch in model.children():
            self.__model_to_list(ch)

    def __hook_fn(self, module, input, output, idx):
        self.__list[idx] = output

    @property
    def features(self):
        return self.__list

    def remove_hooks(self):
        [x.remove() for x in self.__hooks]
