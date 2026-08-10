from app.services.auth_service import hash_password, hash_token, verify_password


def test_password_hash_roundtrip() -> None:
    password_hash = hash_password("s3cret-pass")

    assert verify_password("s3cret-pass", password_hash) is True
    assert verify_password("wrong-pass", password_hash) is False


def test_token_hash_is_stable() -> None:
    token = "demo-token"

    assert hash_token(token) == hash_token(token)
