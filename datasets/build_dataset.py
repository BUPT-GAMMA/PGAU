from torch.utils.data import DataLoader

from .npy_dataset import NpyMapDataset


def build_dataset(args, train=True):
    args = dict(args)
    ann_file = args["train_csv"] if train else args.get("val_csv") or args.get("test_csv")
    if ann_file is None:
        raise ValueError("annotation CSV path is required")
    dataset = NpyMapDataset(
        ann_file=ann_file,
        dataroot=args.get("dataroot"),
        require_label=args.get("require_label", True),
    )
    return DataLoader(
        dataset,
        batch_size=int(args.get("batch_size", 1)),
        shuffle=bool(args.get("shuffle", train)),
        num_workers=int(args.get("num_workers", 0)),
        pin_memory=bool(args.get("pin_memory", True)),
        drop_last=bool(args.get("drop_last", False)) if train else False,
    )
