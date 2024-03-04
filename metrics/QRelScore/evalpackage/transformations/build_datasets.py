from transform_utils import SQUADTransformer
import numpy as np
np.random.seed(32767)


def sampling():
    pass


def main():
    dataset_filename = '/path/to/home/to/your/project/squad1/dev-v1.1-toy.json'
    target_dirname = '/path/to/home/to/your/project/squad1'

    transformer = SQUADTransformer(dataset_filename, target_dirname)

    transformer.transform()


if __name__ == '__main__':
    main()