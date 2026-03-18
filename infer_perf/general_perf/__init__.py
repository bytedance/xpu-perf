import sys
from packaging.version import parse
import warnings

from .version import __version__

def digit_version(version_str: str, length: int = 4):
    """Convert a version string into a tuple of integers.

    This method is usually used for comparing two versions. For pre-release
    versions: alpha < beta < rc.

    Args:
        version_str (str): The version string.
        length (int): The maximum number of version levels. Defaults to 4.

    Returns:
        tuple[int]: The version info in digits (integers).
    """
    assert 'parrots' not in version_str
    version = parse(version_str)
    assert version.release, f'failed to parse version {version_str}'
    release = list(version.release)
    release = release[:length]
    if len(release) < length:
        release = release + [0] * (length - len(release))
    if version.is_prerelease:
        mapping = {'a': -3, 'b': -2, 'rc': -1}
        val = -4
        # version.pre can be None
        if version.pre:
            if version.pre[0] not in mapping:
                warnings.warn(f'unknown prerelease version {version.pre[0]}, '
                              'version checking may go wrong')
            else:
                val = mapping[version.pre[0]]
            release.extend([val, version.pre[-1]])
        else:
            release.extend([val, 0])

    elif version.is_postrelease:
        release.extend([1, version.post])  # type: ignore
    else:
        release.extend([0, 0])
    return tuple(release)

python3_minimum_version = '3.6.0'
python_version = digit_version(sys.version.split()[0])

assert (python_version >= digit_version(python3_minimum_version)), \
    f'PYTHON=={sys.version.split()[0]} is used but incompatible. ' \
    f'Please install python>={python3_minimum_version}.'

__all__ = ['__version__']