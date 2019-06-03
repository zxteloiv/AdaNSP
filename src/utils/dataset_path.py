# DataSet class

import dataclasses


@dataclasses.dataclass()
class DatasetPath:
    train_path: str
    dev_path: str
    test_path: str
