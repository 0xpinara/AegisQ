"""Post-quantum secure job submission.

Primitives come from [liboqs](https://openquantumsafe.org/) (ML-KEM-768,
FIPS 203; ML-DSA-65, FIPS 204) and the `cryptography` package (HKDF-SHA256,
AES-256-GCM). Nothing cryptographic is implemented here — this layer handles
identities, canonical serialisation, envelopes and replay state.
"""

from aegisq.secure.canonical import canonical_bytes, canonical_hash, parse_canonical
from aegisq.secure.kem import CryptoError
from aegisq.secure.keys import (
    KEM_ALGORITHM,
    SIGNATURE_ALGORITHM,
    Identity,
    IdentityError,
    PublicIdentity,
    SecretIdentity,
    generate_client_identity,
    generate_cluster_identity,
    load_identity,
    load_public_identity,
    load_trusted_directory,
)
from aegisq.secure.signatures import SignatureError, sign_payload, verify_payload

__all__ = [
    "KEM_ALGORITHM",
    "SIGNATURE_ALGORITHM",
    "Identity",
    "PublicIdentity",
    "SecretIdentity",
    "IdentityError",
    "CryptoError",
    "SignatureError",
    "generate_client_identity",
    "generate_cluster_identity",
    "load_identity",
    "load_public_identity",
    "load_trusted_directory",
    "sign_payload",
    "verify_payload",
    "canonical_bytes",
    "canonical_hash",
    "parse_canonical",
]
