import numpy as np
import warnings
import sklearn
import os
from pathlib import Path


class BlobGenerate():

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
        X, y = BlobGenerate.generate_xy(
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


class BlobsGenerate(BlobGenerate):

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


class EightGaussiansGenerate(BlobGenerate):

    """
        https://github.com/nicola-decao/BNAF/blob/master/data/generate2d.py
        elif data == "8gaussians":
            scale = 4.
            centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                    (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                            1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
            centers = [(scale * x, scale * y) for x, y in centers]

            dataset = []
            for i in range(batch_size):
                point = rng.randn(2) * 0.5
                idx = rng.randint(8)
                center = centers[idx]
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = np.array(dataset, dtype="float32")
            dataset /= 1.414
            return dataset
    """
    @classmethod
    def from_cfg(cls, name, cfg):
        shared_params = cfg['shared_params']

        fn_X = Path(cfg['path']) / 'x.npy'
        fn_y = Path(cfg['path']) / 'y.npy'

        if not Path(cfg['path']).exists():
            os.makedirs(cfg['path'])

        assert cfg['seed'] is not None, 'seed is None'

        blob_num = shared_params['blob_num']

        centers = []
        for i in range(blob_num):
            angle = 2 * np.pi / blob_num
            cx = np.cos(angle)
            cy = np.sin(angle)
            centers.append([cx, cy])

        stds = [shared_params['blob_std']] * blob_num
        heights = [shared_params['blob_height']] * blob_num
        x_bounds = shared_params['x_bounds']
        dimension = shared_params.get('dimension', 2)
        num_samples = cfg['num_samples']
        seed = cfg['seed']
        X, y = cls.generate_xy(
            cfg=shared_params, centers=centers, clusters_std=stds, heights=heights, n_samples=num_samples, dimension=dimension,
            x_bounds=x_bounds, seed=seed)

        np.save(fn_X, X)
        np.save(fn_y, y)

        print(f'{fn_X} is saved')
        print(f'{fn_y} is saved')


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


class CirclesGenerate():

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
        factor = shared_params['factor']

        num_samples = cfg['num_samples']
        seed = cfg['seed']
        X, y = cls.generate_xy(
            cfg=shared_params,
            noise=noise, x_scale=x_scale,
            factor=factor,
            n_samples=num_samples, dimension=dimension,
            x_bounds=x_bounds, seed=seed)

        np.save(fn_X, X)
        np.save(fn_y, y)

        print(f'{fn_X} is saved')
        print(f'{fn_y} is saved')

    @classmethod
    def generate_xy(cls, cfg={}, noise=0.1, x_scale=2,
                    shifts=[-1, -0.2],   # TODO: params
                    factor=0.5,
                    n_samples=100, dimension=2, x_bounds=[-2, 2], seed=None):

        if seed is None:
            warnings.warn('seed is None', UserWarning)

        rng = np.random.RandomState(seed)
        # rng = np.random.default_rng(seed)

        xx, yy = sklearn.datasets.make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=rng)
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
