import hashlib
import os
from pathlib import Path
import tarfile
import time

# import hydra.utils
from hydra.experimental import compose, initialize
from hydra.core.global_hydra import GlobalHydra
# from omegaconf import OmegaConf, open_dict

from .config import CONFIGS_PATH, DATETIME_OF_FILES
from .classes import BlobGenerate, BlobsGenerate, TwoMoonsGenerate, CirclesGenerate, EightGaussiansGenerate, TwoSpiralsGenerate


def _get_configs_path(config_path):
    if config_path is None:
        # config_path must be relative for hydra
        config_path = Path(os.path.relpath(CONFIGS_PATH, Path(os.getcwd())))           # .parent  for Path(os.getcwd()).parent  - bug ???
    # assert config_path.exists(), config_path
    return config_path


def _get_cfg(config_name='blobs', config_path=None, destination_dir=None):
    config_path = _get_configs_path(config_path)
    print("config_path:", config_path)

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


def reset(tarinfo):
    tarinfo.uid = tarinfo.gid = 0
    tarinfo.uname = tarinfo.gname = "root"
    # print(tarinfo, tarinfo.mtime)
    return tarinfo


def create_tar(config_name, destination_dir, files=["train/x.npy", "train/y.npy", "test/x.npy", "test/y.npy", ]):

    fn_tar = Path(destination_dir) / f"{config_name}.tar.gz"
    tar = tarfile.open(fn_tar, "w:gz")
    mod_time = time.mktime(DATETIME_OF_FILES.timetuple())
    for name in files:
        fn = destination_dir / name
        os.utime(fn, (mod_time, mod_time))
        # print(fn, mod_time)
        tar.add(fn, arcname=name, filter=reset)
    tar.close()

    os.utime(fn_tar, (mod_time, mod_time))

    print(f'{fn_tar} is saved')

    md5_hex = md5_file(fn_tar)
    print(md5_hex)

    fn_md5 = Path(destination_dir) / f"{config_name}.tar.gz.md5"
    with open(fn_md5, "w") as f:
        f.write(md5_hex)


def blob_generate(config_name='blob', destination_dir=None, config_path=None):
    assert destination_dir is not None
    assert Path(destination_dir).exists(), destination_dir

    cfg = _get_cfg(config_name=config_name, config_path=config_path, destination_dir=destination_dir)

    BlobGenerate.from_cfg('train', cfg.datasets.train)
    BlobGenerate.from_cfg('test', cfg.datasets.test)

    create_tar(config_name, destination_dir, files=["train/x.npy", "train/y.npy", "test/x.npy", "test/y.npy", ])


def blobs_generate(config_name='blobs', destination_dir=None, config_path=None):
    assert destination_dir is not None
    assert Path(destination_dir).exists(), destination_dir

    cfg = _get_cfg(config_name=config_name, config_path=config_path, destination_dir=destination_dir)

    BlobsGenerate.from_cfg('train', cfg.datasets.train)
    BlobsGenerate.from_cfg('test', cfg.datasets.test)

    create_tar(config_name, destination_dir, files=["train/x.npy", "train/y.npy", "test/x.npy", "test/y.npy", ])


def eight_gaussians_generate(config_name='eight_gaussians', destination_dir=None, config_path=None):
    assert destination_dir is not None
    assert Path(destination_dir).exists(), destination_dir

    cfg = _get_cfg(config_name=config_name, config_path=config_path, destination_dir=destination_dir)

    EightGaussiansGenerate.from_cfg('train', cfg.datasets.train)
    EightGaussiansGenerate.from_cfg('test', cfg.datasets.test)

    create_tar(config_name, destination_dir, files=["train/x.npy", "train/y.npy", "test/x.npy", "test/y.npy", ])


def two_moons_generate(config_name='two_moons', destination_dir=None, config_path=None):
    assert destination_dir is not None
    assert Path(destination_dir).exists(), destination_dir
    cfg = _get_cfg(config_name=config_name, config_path=config_path, destination_dir=destination_dir)

    TwoMoonsGenerate.from_cfg('train', cfg.datasets.train)
    TwoMoonsGenerate.from_cfg('test', cfg.datasets.test)

    create_tar(config_name, destination_dir, files=["train/x.npy", "train/y.npy", "test/x.npy", "test/y.npy", ])


def circles_generate(config_name='circles', destination_dir=None, config_path=None):
    assert destination_dir is not None
    assert Path(destination_dir).exists(), destination_dir
    cfg = _get_cfg(config_name=config_name, config_path=config_path, destination_dir=destination_dir)

    CirclesGenerate.from_cfg('train', cfg.datasets.train)
    CirclesGenerate.from_cfg('test', cfg.datasets.test)

    create_tar(config_name, destination_dir, files=["train/x.npy", "train/y.npy", "test/x.npy", "test/y.npy", ])


def two_spirals_generate(config_name='two_spirals', destination_dir=None, config_path=None):
    assert destination_dir is not None
    assert Path(destination_dir).exists(), destination_dir
    cfg = _get_cfg(config_name=config_name, config_path=config_path, destination_dir=destination_dir)

    TwoSpiralsGenerate.from_cfg('train', cfg.datasets.train)
    TwoSpiralsGenerate.from_cfg('test', cfg.datasets.test)

    create_tar(config_name, destination_dir, files=["train/x.npy", "train/y.npy", "test/x.npy", "test/y.npy", ])
