from util.data_loader import BreakoutDataset


if __name__ == "__main__":
    ds = BreakoutDataset()
    print(len(ds))
    print(ds[247][0][0].shape)