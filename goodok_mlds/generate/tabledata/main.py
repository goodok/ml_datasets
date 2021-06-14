import hashlib
import os
from pathlib import Path
import numpy as np
import warnings
import tarfile
import sklearn

# import hydra.utils
from hydra.experimental import compose, initialize
from hydra.core.global_hydra import GlobalHydra
# from omegaconf import OmegaConf, open_dict

from .config import CONFIGS_PATH


def _get_configs_path(config_path):
    if config_path is None:
        # config_path must be relative for hydra
        config_path = Path(os.path.relpath(CONFIGS_PATH, Path(os.getcwd()).parent))           # .parent  for Path(os.getcwd()).parent  - bug ???
    # assert config_path.exists(), config_path
    return config_path


def _get_cfg(config_name='blobs', config_path=None, destination_dir=None):
    config_path = _get_configs_path(config_path)

    GlobalHydra.instance().clear()
    initialize(config_path=config_path, job_name="test_app")
    cfg = compose(config_name=config_name, overrides=[f'datasets.destination_dir="{destination_dir}"'])
    # print(cfg)
    return cfg


def md5_file(filename):
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def create_tar(config_name, destination_dir, files=["train/x.npy", "train/y.npy", "test/x.npy", "test/y.npy", ]):
    fn_tar = Path(destination_dir) / f"{config_name}.tar.gz"
    tar = tarfile.open(fn_tar, "w:gz")
    for name in files:
        tar.add(destination_dir / name, arcname=name)
    tar.close()

    print(f'{fn_tar} is saved')

    md5_hex = md5_file(fn_tar)
    print(md5_hex)

    fn_md5 = Path(destination_dir) / f"{config_name}.tar.gz.md5"
    with open(fn_md5, "w") as f:
        f.write(md5_hex)


def blobs_generate(config_name='blobs', destination_dir=None, config_path=None):
    assert destination_dir is not None
    assert Path(destination_dir).exists(), destination_dir

    cfg = _get_cfg(config_name=config_name, config_path=config_path, destination_dir=destination_dir)

    BlobsGenerate.from_cfg('train', cfg.datasets.train)
    BlobsGenerate.from_cfg('test', cfg.datasets.test)

    create_tar(config_name, destination_dir, files=["train/x.npy", "train/y.npy", "test/x.npy", "test/y.npy", ])


def two_moons_generate(config_name='two_moons', destination_dir=None, config_path=None):
    assert destination_dir is not None
    assert Path(destination_dir).exists(), destination_dir
    cfg = _get_cfg(config_name=config_name, config_path=config_path, destination_dir=destination_dir)

    TwoMoonsGenerate.from_cfg('train', cfg.datasets.train)
    TwoMoonsGenerate.from_cfg('test', cfg.datasets.test)

    create_tar(config_name, destination_dir, files=["train/x.npy", "train/y.npy", "test/x.npy", "test/y.npy", ])


class BlobsGenerate():

    @classmethod
    def from_cfg(cls, name, cfg):
        shared_params = cfg['shared_params']

        fn_X = Path(cfg['path']) / 'x.npy'
        fn_y = Path(cfg['path']) / 'y.npy'

        if not Path(cfg['path']).exists():
            os.makedirs(cfg['path'])

        assert cfg['seed'] is not None, 'seed is None'

        centers = [v['center'] for v in shared_params['blobs']]
        stds = [v['std'] for v in shared_params['blobs']]
        heights = [v['height'] for v in shared_params['blobs']]
        x_bounds = shared_params['x_bounds']
        dimension = shared_params.get('dimension', 2)
        num_samples = cfg['num_samples']
        seed = cfg['seed']
        X, y = BlobsGenerate.generate_xy(
            cfg=shared_params, centers=centers, clusters_std=stds, heights=heights, n_samples=num_samples, dimension=dimension,
            x_bounds=x_bounds, seed=seed)

        np.save(fn_X, X)
        np.save(fn_y, y)

        print(f'{fn_X} is saved')
        print(f'{fn_y} is saved')

    @classmethod
    def generate_xy(cls, cfg={}, centers=[[2, 2], [-2, -2]], clusters_std=[0.5, 0.5], heights=[1, -1], n_samples=100, dimension=2, x_bounds=[-2, 2], seed=None):
        if seed is None:
            warnings.warn('seed is None', UserWarning)

        rng = np.random.RandomState(seed)
        # rng = np.random.default_rng(seed)

        for center in centers:
            assert len(center) == dimension, f"len{center} != {dimension}"
        assert len(centers) == len(clusters_std)
        assert len(centers) == len(heights)

        points = rng.uniform(x_bounds[0], x_bounds[1], (n_samples, dimension))
        points = points.astype(np.float32)

        yy = cls.function(points, centers, clusters_std, heights)
        yy = yy.astype(np.float32)

        points = cls.x_scale_points(points, cfg)

        return points, yy

    @classmethod
    def function(cls, points, centers, clusters_std, heights):
        y_sum = []
        for center, std, height in zip(centers, clusters_std, heights):
            dr = np.array(points) - np.array(center)
            dr2 = (dr ** 2).sum(axis=1)
            y = np.exp(- dr2 / (2 * std ** 2))
            y = y * height
            y_sum.append(y)

        y = np.stack(y_sum).T
        y = np.sum(y, axis=1)
        return y

    @classmethod
    def x_scale_points(cls, points, cfg):
        scale = cfg.get('x_scale', None)
        if scale is not None:
            points = points * scale
        return points


class TwoMoonsGenerate():

    @classmethod
    def from_cfg(cls, name, cfg):
        shared_params = cfg['shared_params']

        fn_X = Path(cfg['path']) / 'x.npy'
        fn_y = Path(cfg['path']) / 'y.npy'

        if not Path(cfg['path']).exists():
            os.makedirs(cfg['path'])

        assert cfg['seed'] is not None, 'seed is None'

        x_bounds = shared_params['x_bounds']
        dimension = shared_params.get('dimension', 2)
        noise = shared_params['noise']
        x_scale = shared_params['x_scale']

        num_samples = cfg['num_samples']
        seed = cfg['seed']
        X, y = TwoMoonsGenerate.generate_xy(
            cfg=shared_params,
            noise=noise, x_scale=x_scale,
            n_samples=num_samples, dimension=dimension,
            x_bounds=x_bounds, seed=seed)

        np.save(fn_X, X)
        np.save(fn_y, y)

        print(f'{fn_X} is saved')
        print(f'{fn_y} is saved')

    @classmethod
    def generate_xy(cls, cfg={}, noise=0.1, x_scale=2,
                    shifts=[-1, -0.2],   # TODO: params
                    n_samples=100, dimension=2, x_bounds=[-2, 2], seed=None):

        if seed is None:
            warnings.warn('seed is None', UserWarning)

        rng = np.random.RandomState(seed)
        # rng = np.random.default_rng(seed)

        xx, yy = sklearn.datasets.make_moons(n_samples=n_samples, noise=noise, random_state=rng)
        xx = xx.astype("float32")
        xx = xx * x_scale + np.array(shifts)
        xx = xx.astype("float32")

        yy = yy.astype(np.float32)

        # points = cls.x_scale_points(xx, cfg)
        points = xx

        return points, yy

    @classmethod
    def function(cls, points, centers, clusters_std, heights):
        y_sum = []
        for center, std, height in zip(centers, clusters_std, heights):
            dr = np.array(points) - np.array(center)
            dr2 = (dr ** 2).sum(axis=1)
            y = np.exp(- dr2 / (2 * std ** 2))
            y = y * height
            y_sum.append(y)

        y = np.stack(y_sum).T
        y = np.sum(y, axis=1)
        return y

    # @classmethod
    # def x_scale_points(cls, points, cfg):
    #     scale = cfg.get('x_scale', None)
    #     if scale is not None:
    #         points = points * scale
    #     return points