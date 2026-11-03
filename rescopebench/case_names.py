"""Short installation names with stable identifiers for frozen results."""

_IDENTITIES = {
    "cinema": "cinema_c_20_04_03",
    "filesystem": "filesystem_c_14_01_54",
    "railway_ticketing": "railway_ticketing_c_04_06_01",
    "supply_chain_finance": "supply_chain_finance_c_05_10_53",
}
_NAMES = {identity: name for name, identity in _IDENTITIES.items()}


def case_identity(value: str) -> str:
    """Accept a public name or an existing result identifier."""
    if (not isinstance(value, str) or not value
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in value)):
        raise ValueError("invalid case identifier")
    return _IDENTITIES.get(value, value)


def case_name(value: str) -> str:
    """Return the directory and CLI name without rewriting frozen task inputs."""
    identity = case_identity(value)
    return _NAMES.get(identity, identity)
