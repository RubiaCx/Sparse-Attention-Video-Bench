from sparse_bench.morpher import Strategy


class VideoGenStrategy(Strategy):
    def __init__(self, pipe):
        super().__init__()
        self.pipe = pipe

    def get_module_transformation(self):
        return {}
