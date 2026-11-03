"""Explicit operator credentials, held only in the host process memory."""

from rescopebench.portable_provider_config import ProviderConfig, load_credentials as read_pool
from rescopebench.portable_provider_runtime import provider_from_transport, validate_transport

_pools = {}


def load_credentials(path=None, *, transport=None):
    config = (ProviderConfig.from_record(validate_transport(transport)) if transport is not None
              else ProviderConfig.from_environment())
    pool = read_pool(config, path=path)
    _pools[config] = pool
    return len(pool.keys)


def provider(directory, transport):
    config = ProviderConfig.from_record(validate_transport(transport))
    pool = _pools.get(config)
    if pool is None:
        pool = read_pool(config)
    return provider_from_transport(directory, transport, pool=pool)
