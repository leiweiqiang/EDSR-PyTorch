import os
from data import srdata

class DIV2K(srdata.SRData):
    def __init__(self, args, name='DIV2K', train=True, benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(DIV2K, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr = super(DIV2K, self)._scan()
        begin = self.begin
        end = self.end
        if (not self.train) and len(names_hr) < end and begin >= 801:
            begin -= 800
            end -= 800
        names_hr = names_hr[begin - 1:end]
        names_lr = [n[begin - 1:end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(DIV2K, self)._set_filesystem(dir_data)
        if self.train:
            hr_dir = 'DIV2K_train_HR'
            lr_dir = 'DIV2K_train_LR_bicubic'
        else:
            hr_dir = 'DIV2K_valid_HR'
            lr_dir = 'DIV2K_valid_LR_bicubic'
            if not os.path.isdir(os.path.join(self.apath, hr_dir)):
                hr_dir = 'DIV2K_train_HR'
                lr_dir = 'DIV2K_train_LR_bicubic'
        self.dir_hr = os.path.join(self.apath, hr_dir)
        self.dir_lr = os.path.join(self.apath, lr_dir)
        if self.input_large: self.dir_lr += 'L'

